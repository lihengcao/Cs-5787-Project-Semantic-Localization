"""Shared stdlib-only pipeline orchestration for VM and Colab runners."""
from __future__ import annotations

import argparse
import codecs
import csv
import ctypes
from datetime import datetime, timezone
import fcntl
import getpass
import glob
import hashlib
import inspect
import json
import math
import os
import pty
import re
import shutil
import signal
import struct
import subprocess
import sys
import tarfile
import tempfile
import termios
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Sequence
from typing import Any

RUNNERS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(RUNNERS_DIR)

DEFAULT_GCS_BACKUP_URI = "gs://scenealigner"
METRICS_ORDER = ("Pos@0.1m", "Pos@0.5m", "Pos@1.0m", "Ang&Pos@(30,1m)")
PHOTO_SEM_SOURCE_CHOICES = ("gt", "none")
SEM_MODE_CHOICES = ("none", "uniform", "consensus")
STAGE_CHOICES = ("smoke", "prep", "train", "eval", "backup")
TRAIN_VALID_CHUNKS = tuple(c for c in range(15) if c != 8)
TEST_SCENE_RANGE = range(3250, 3500)
TEST_SCENES_TAR = "test_scenes_3250_3500.tar"
RECON_CACHE_PREFIX = "s3d_eval/recon_cache_pi3_orig"
RECON_CACHE_SHARDS = tuple((s, s + 50) for s in range(3250, 3500, 50))
MIN_RECON_CACHE_FREE_BYTES = 22 * 1024**3
VIEW_PATTERNS = [
    "Structured3D/*/2D_rendering/*/perspective/full/*/rgb_rawlight.png",
    "Structured3D/*/2D_rendering/*/perspective/full/*/depth.png",
    "Structured3D/*/2D_rendering/*/perspective/full/*/semantic.png",
    "Structured3D/*/2D_rendering/*/perspective/full/*/camera_pose.txt",
]
EVAL_CHUNK_SENTINELS = ((16, "scene_03250"), (17, "scene_03400"))
_EVAL_CHUNK_RANGES: tuple[tuple[int, range], ...] = (
    (16, range(3250, 3400)),
    (17, range(3400, 3500)),
)
_S3D_DATA_SUBDIRS = ("floorplan", "floorplan_sem", "depth", "pi3", "depth_sem", "pi3_sem")
_COLAB_BASE_PACKAGES = frozenset({"numpy", "pillow", "opencv-python", "pandas", "matplotlib"})
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")
_INLINE_CR_ANSI_RE = re.compile(r"\x1b\[(?:2?K|[01]?G)")
_TRAILING_INT_RE = re.compile(r"(\d+)$")
_SLICE_TAR_RE = re.compile(r"^scenes_(\d+)_(\d+)\.tar$")
_SLICE_CSV_RE = re.compile(r"^scenes_(\d+)_(\d+)\.csv$")
_RECON_SHARD_TAR_RE = re.compile(r"^recon_(\d+)_(\d+)\.tar$")


def _strip_prefix(text: str, prefix: str) -> str:
    return text[len(prefix) :] if text.startswith(prefix) else text


def _inter_eval_enabled(args: argparse.Namespace) -> bool:
    return (
        int(getattr(args, "eval_every_epochs", 1)) > 0
        and int(getattr(args, "inter_eval_scenes", 50)) > 0
    )


def _parse_scene_int(scene_id: str) -> int | None:
    match = _TRAILING_INT_RE.search(str(scene_id).strip())
    return int(match.group(1)) if match else None


def _normalize_gcs_uri(raw: str) -> str:
    cleaned = raw.strip()
    if not cleaned:
        return ""
    return f"gs://{_strip_prefix(cleaned, 'gs://').strip('/')}"


def _normalize_s3d_url(raw: str) -> str:
    cleaned = _ANSI_RE.sub("", raw)
    cleaned = re.sub(r"[\x00-\x1f\x7f\u200b\ufeff]", "", cleaned).strip()
    cleaned = re.sub(r"^(?:export\s+)?S3D_BASE_URL\s*=\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.strip().strip("\"'`").strip()
    parts = cleaned.split()
    if len(parts) >= 2 and len(set(parts)) == 1:
        cleaned = parts[0]
    if (
        len(cleaned) % 2 == 0
        and cleaned[: len(cleaned) // 2] == cleaned[len(cleaned) // 2 :]
        and cleaned.lower().startswith(("http://", "https://"))
    ):
        cleaned = cleaned[: len(cleaned) // 2]
    cleaned = cleaned.rstrip("/")
    if cleaned.endswith("/README.txt"):
        cleaned = cleaned[: -len("/README.txt")]
    if re.search(r"/[^/]+\.zip$", cleaned, flags=re.IGNORECASE):
        cleaned = cleaned.rsplit("/", 1)[0]
    return cleaned.rstrip("/")


def _redact_secret(text: str, secret: str | None, placeholder: str = "<REDACTED>") -> str:
    if not secret or not text:
        return text
    out = text.replace(secret, placeholder)
    trimmed = secret.rstrip("/")
    if trimmed and trimmed != secret:
        out = out.replace(trimmed, placeholder)
    parsed = urllib.parse.urlsplit(trimmed or secret)
    if parsed.scheme in ("http", "https") and parsed.netloc:
        out = out.replace(f"{parsed.scheme}://{parsed.netloc}", placeholder)
        out = out.replace(parsed.netloc, placeholder)
        if parsed.hostname:
            out = out.replace(parsed.hostname, placeholder)
    return out


def _verify_s3d_base_url(base_url: str, timeout: float = 8.0) -> None:
    """Validate URL syntax and confirm Structured3D_annotation_3d.zip responds with 2xx/3xx."""
    parsed = urllib.parse.urlsplit(base_url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc or any(c.isspace() for c in base_url):
        raise ValueError("Invalid S3D_BASE_URL: expected an http(s):// base URL.")

    probe_url = f"{base_url.rstrip('/')}/Structured3D_annotation_3d.zip"

    def _safe_err(exc: BaseException) -> str:
        return _redact_secret(str(exc), base_url, "<S3D_BASE_URL>")

    def _probe(method: str, headers: dict[str, str]) -> int:
        req = urllib.request.Request(probe_url, method=method, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return int(getattr(resp, "status", 200))

    headers = {"User-Agent": "aria2/1.37.0"}
    try:
        status = _probe("HEAD", headers)
        if 200 <= status < 400:
            return
        raise RuntimeError(
            f"S3D_BASE_URL verification failed (HTTP {status}) for <S3D_BASE_URL>/Structured3D_annotation_3d.zip."
        )
    except urllib.error.HTTPError as err:
        code, reason = err.code, err.reason
        err.close()
        if code in (403, 405, 501):
            try:
                status = _probe("GET", {**headers, "Range": "bytes=0-0"})
                if 200 <= status < 400:
                    return
                raise RuntimeError(
                    f"S3D_BASE_URL verification failed (HTTP {status}) "
                    "for <S3D_BASE_URL>/Structured3D_annotation_3d.zip."
                )
            except urllib.error.HTTPError as get_err:
                get_code, get_reason = get_err.code, get_err.reason
                get_err.close()
                raise RuntimeError(
                    f"S3D_BASE_URL verification failed (HTTP {get_code} {get_reason}) "
                    "for <S3D_BASE_URL>/Structured3D_annotation_3d.zip."
                ) from None
            except (urllib.error.URLError, TimeoutError, OSError) as get_err:
                raise RuntimeError(
                    f"S3D_BASE_URL verification failed ({_safe_err(get_err)}) "
                    "for <S3D_BASE_URL>/Structured3D_annotation_3d.zip."
                ) from None
        raise RuntimeError(
            f"S3D_BASE_URL verification failed (HTTP {code} {reason}) "
            "for <S3D_BASE_URL>/Structured3D_annotation_3d.zip."
        ) from None
    except (urllib.error.URLError, TimeoutError, OSError) as err:
        raise RuntimeError(
            f"S3D_BASE_URL verification failed ({_safe_err(err)}) "
            "for <S3D_BASE_URL>/Structured3D_annotation_3d.zip."
        ) from None


def _requirements_hash(repo_dir: str | None = None) -> str:
    base_dir = repo_dir or REPO_DIR
    req_path = os.path.join(base_dir, "requirements.txt")
    with open(req_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _find_7z(required: bool = True) -> str | None:
    for candidate in ("7z", "7zz", "7za"):
        found = shutil.which(candidate)
        if found:
            return found
    if required:
        raise FileNotFoundError("Missing 7z/7zz CLI. Run ./runners/setup_vm.sh to install p7zip-full.")
    return None


def _resolve_chunks(args: argparse.Namespace) -> list[int]:
    chunks = list(TRAIN_VALID_CHUNKS) if getattr(args, "all_train_chunks", False) else list(args.chunks)
    return [c for c in dict.fromkeys(chunks) if c != 8]


def _chunks_to_scene_ranges(chunks: Sequence[int]) -> list[tuple[int, int]]:
    return [(int(c) * 200, (int(c) + 1) * 200) for c in chunks]


def compute_prep_profile(
    photo_sem_source: str = "gt",
    include_rooms: bool = False,
    structural_photo_semantics: bool = False,
) -> str:
    parts = [f"photo-{photo_sem_source}"]
    if photo_sem_source != "none" and structural_photo_semantics:
        parts.append("struct")
    if include_rooms:
        parts.append("rooms")
    return "_".join(parts)


def format_chunks_slug(chunks: Sequence[int], all_train_chunks: bool = False) -> str:
    valid = [c for c in dict.fromkeys(chunks) if c != 8]
    if all_train_chunks or set(valid) == set(TRAIN_VALID_CHUNKS):
        return "c-all"
    if not valid:
        return "c-none"
    sorted_c = sorted(valid)
    spans: list[str] = []
    start = prev = sorted_c[0]
    for c in sorted_c[1:]:
        if c == prev + 1:
            prev = c
        else:
            spans.append(f"{start}-{prev}" if prev > start else str(start))
            start = prev = c
    spans.append(f"{start}-{prev}" if prev > start else str(start))
    return "c" + ".".join(spans)


def _sanitize_tag(raw: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", raw.strip()).strip("-.")
    return cleaned[:48]


def build_run_slug(args: argparse.Namespace) -> str:
    prep_profile = compute_prep_profile(
        args.photo_sem_source, args.include_rooms, args.structural_photo_semantics
    )
    parts = ["s3d", prep_profile]
    if "train" in args.stages:
        chunks_slug = format_chunks_slug(_resolve_chunks(args), getattr(args, "all_train_chunks", False))
        parts.extend([args.sem_mode, chunks_slug, f"ep{args.epochs}"])
    elif "eval" in args.stages:
        parts.append("eval")
    else:
        parts.append("-".join(args.stages))
    if getattr(args, "tag", None):
        tag = _sanitize_tag(str(args.tag))
        if tag:
            parts.append(tag)
    return "_".join(parts)


def compute_run_fingerprint(args: argparse.Namespace) -> dict[str, Any]:
    chunks = _resolve_chunks(args)
    return {
        "prep_profile": compute_prep_profile(
            args.photo_sem_source, args.include_rooms, args.structural_photo_semantics
        ),
        "photo_sem_source": str(args.photo_sem_source),
        "include_rooms": bool(args.include_rooms),
        "structural_photo_semantics": bool(args.structural_photo_semantics),
        "sem_mode": str(args.sem_mode),
        "lambda_sem": float(args.lambda_sem),
        "alpha_sem": None if args.alpha_sem is None else float(args.alpha_sem),
        "init_lora": str(args.init_lora),
        "freeze_lora": bool(args.freeze_lora),
        "epochs": int(args.epochs),
        "resolutions": [int(r) for r in args.resolutions],
        "min_points": int(args.min_points),
        "chunks": chunks if ("train" in args.stages or "prep" in args.stages) else None,
        "eval_checkpoint_dir": args.eval_checkpoint_dir,
        "tag": getattr(args, "tag", None) or None,
    }


def _atomic_write_text(path: str, content: str, mode: int | None = None) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(content)
    if mode is not None:
        os.chmod(tmp_path, mode)
    os.replace(tmp_path, path)


def _same_open_file(stream: Any, path: str) -> bool:
    if not os.path.exists(path):
        return False
    try:
        fd = stream.fileno()
        st_stream = os.fstat(fd)
        st_path = os.stat(path)
        return (st_stream.st_dev, st_stream.st_ino) == (st_path.st_dev, st_path.st_ino)
    except Exception:
        return False


class _CleanLogWriter:
    """Line-buffered log writer that redacts secrets, strips ANSI codes, and collapses \\r progress."""

    def __init__(
        self,
        log_path: str,
        *,
        redact: str | None = None,
        echo_to_stdout: bool = False,
        progress_interval_sec: float = 15.0,
    ) -> None:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.log_path = log_path
        self.redact = redact or None
        self.echo_to_stdout = echo_to_stdout and not _same_open_file(sys.stdout, log_path)
        self.progress_interval_sec = progress_interval_sec
        self._fp = open(log_path, "a", encoding="utf-8", buffering=1)
        self._partial = ""
        self._pending_cr = False
        self._last_progress_ts = time.monotonic()

    def _clean(self, text: str) -> str:
        text = _redact_secret(text, self.redact, "<REDACTED>")
        return _ANSI_RE.sub("", text)

    def _emit_line(self, line: str) -> None:
        self._fp.write(line + "\n")
        self._fp.flush()
        if self.echo_to_stdout:
            sys.stdout.write(line + "\n")
            sys.stdout.flush()

    def log_line(self, msg: str) -> None:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        line = f"[{ts}] {self._clean(msg)}"
        self._fp.write(line + "\n")
        self._fp.flush()
        if not _same_open_file(sys.stdout, self.log_path):
            sys.stdout.write(line + "\n")
            sys.stdout.flush()

    def write_chunk(self, text: str) -> None:
        cleaned = self._clean(text)
        if not cleaned:
            return
        if self._pending_cr:
            self._pending_cr = False
            if cleaned.startswith("\n"):
                self._emit_line(self._partial)
                self._partial = ""
                cleaned = cleaned[1:]
            else:
                self._partial = ""
        if cleaned.endswith("\r"):
            self._pending_cr = True
            cleaned = cleaned[:-1]

        cleaned = cleaned.replace("\r\n", "\n")
        parts = cleaned.split("\n")
        for seg in parts[:-1]:
            if "\r" in seg:
                self._partial = seg.rsplit("\r", 1)[-1]
            else:
                self._partial += seg
            self._emit_line(self._partial)
            self._partial = ""

        tail = parts[-1]
        if "\r" in tail:
            self._partial = tail.rsplit("\r", 1)[-1]
        else:
            self._partial += tail

        now = time.monotonic()
        if (
            self._partial.strip()
            and self.progress_interval_sec > 0
            and now - self._last_progress_ts >= self.progress_interval_sec
        ):
            self._emit_line(f"[progress] {self._partial}")
            self._last_progress_ts = now

    def flush_partial(self) -> None:
        if self._partial:
            self._emit_line(self._partial)
            self._partial = ""
        self._pending_cr = False
        self._fp.flush()

    def close(self) -> None:
        self.flush_partial()
        self._fp.close()


def _kill_process_group(proc: subprocess.Popen[bytes] | subprocess.Popen[str]) -> None:
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            break
        try:
            proc.wait(timeout=2)
            return
        except subprocess.TimeoutExpired:
            continue
    try:
        proc.wait(timeout=1)
    except Exception:
        pass


def _is_interactive_stdout() -> bool:
    """Return True for a real TTY or notebook OutStream (Colab/Jupyter) that handles \\r in-place."""
    try:
        if sys.stdout.isatty():
            return True
    except Exception:
        pass
    mod = getattr(type(sys.stdout), "__module__", "")
    return isinstance(mod, str) and mod.startswith(("ipykernel.", "google.colab."))


def run_live(
    cmd: list[str],
    log_writer: _CleanLogWriter | None = None,
    *,
    redact: str | None = None,
    cwd: str | None = None,
) -> None:
    """Run a subprocess attached to a PTY so progress bars stream cleanly and signals reap child groups."""
    master_fd, slave_fd = pty.openpty()
    try:
        fcntl.ioctl(slave_fd, termios.TIOCSWINSZ, struct.pack("HHHH", 24, 120, 0, 0))
    except OSError:
        pass
    parent_pid = os.getpid()
    work_dir = cwd or REPO_DIR

    def _child_setup() -> None:
        os.setsid()
        try:
            libc = ctypes.CDLL("libc.so.6")
            libc.prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG = 1 on Linux
        except OSError:
            pass
        if os.getppid() != parent_pid:
            os._exit(1)

    proc = subprocess.Popen(
        cmd,
        cwd=work_dir,
        stdin=subprocess.DEVNULL,
        stdout=slave_fd,
        stderr=slave_fd,
        close_fds=True,
        preexec_fn=_child_setup,
    )
    os.close(slave_fd)
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    real_tty = False
    try:
        real_tty = sys.stdout.isatty()
    except Exception:
        real_tty = False
    stream_stdout = real_tty or _is_interactive_stdout()
    needs_newline = False

    def _emit_stdout(raw_text: str) -> None:
        nonlocal needs_newline
        if not stream_stdout:
            return
        out_text = raw_text if real_tty else _ANSI_RE.sub("", _INLINE_CR_ANSI_RE.sub("\r", raw_text))
        if not out_text:
            return
        sys.stdout.write(out_text)
        sys.stdout.flush()
        needs_newline = not out_text.endswith("\n")

    try:
        while True:
            try:
                chunk = os.read(master_fd, 4096)
            except OSError:
                break
            if not chunk:
                break
            text = _redact_secret(decoder.decode(chunk), redact, "<REDACTED>")
            _emit_stdout(text)
            if log_writer is not None:
                log_writer.write_chunk(text)
        rest = decoder.decode(b"", final=True)
        if rest:
            rest = _redact_secret(rest, redact, "<REDACTED>")
            _emit_stdout(rest)
            if log_writer is not None:
                log_writer.write_chunk(rest)
        if stream_stdout and not real_tty and needs_newline:
            sys.stdout.write("\n")
            sys.stdout.flush()
        if log_writer is not None:
            log_writer.flush_partial()
    except BaseException:
        _kill_process_group(proc)
        raise
    finally:
        os.close(master_fd)

    rc = proc.wait()
    if rc != 0:
        msg = _redact_secret(f"Command failed (exit {rc}): {' '.join(cmd)}", redact, "<REDACTED>")
        raise RuntimeError(msg)


def build_workspace_paths(
    workspace: str,
    run_id: str,
    prep_profile: str = "photo-gt",
) -> dict[str, str]:
    ws = os.path.abspath(os.path.expanduser(workspace))
    runs_dir = os.path.join(ws, "runs")
    run_out = os.path.join(runs_dir, run_id)
    s3d_data_root = os.path.join(ws, "s3d_data")
    return {
        "workspace": ws,
        "env_file": os.path.join(ws, ".env"),
        "raw_zips": os.path.join(ws, "raw_zips"),
        "s3d_raw": os.path.join(ws, "Structured3D"),
        "s3d_data_root": s3d_data_root,
        "s3d_data": os.path.join(s3d_data_root, prep_profile),
        "s3d_recon_cache": os.path.join(ws, "s3d_recon_cache"),
        "runs_dir": runs_dir,
        "active_run_file": os.path.join(runs_dir, "ACTIVE_RUN"),
        "index_jsonl": os.path.join(runs_dir, "INDEX.jsonl"),
        "run_out": run_out,
        "manifest_json": os.path.join(run_out, "run_manifest.json"),
        "hf_cache": os.path.join(ws, "hf_cache"),
        "log_path": os.path.join(run_out, "pipeline.log"),
        "pid_path": os.path.join(run_out, "run.pid"),
        "summary_json": os.path.join(run_out, "results_summary.json"),
        "summary_txt": os.path.join(run_out, "results_summary.txt"),
        "latest_json": os.path.join(ws, "latest_results.json"),
        "latest_txt": os.path.join(ws, "latest_results.txt"),
    }


def migrate_local_legacy_s3d_data(paths: dict[str, str], prep_profile: str = "photo-gt") -> None:
    """Move legacy flat workspace/s3d_data/scenes_*.csv + folders into workspace/s3d_data/photo-gt/."""
    if prep_profile != "photo-gt":
        return
    root_dir = paths["s3d_data_root"]
    target_dir = paths["s3d_data"]
    if not os.path.isdir(root_dir):
        return
    flat_csvs = glob.glob(os.path.join(root_dir, "scenes_*.csv"))
    target_csvs = glob.glob(os.path.join(target_dir, "scenes_*.csv")) if os.path.isdir(target_dir) else []
    if not flat_csvs or target_csvs:
        return
    os.makedirs(target_dir, exist_ok=True)
    for pattern in ("scenes_*.csv", "scenes_*.csv.jsonl.part"):
        for src in glob.glob(os.path.join(root_dir, pattern)):
            dst = os.path.join(target_dir, os.path.basename(src))
            if not os.path.exists(dst):
                shutil.move(src, dst)
    for sub in _S3D_DATA_SUBDIRS:
        src_sub = os.path.join(root_dir, sub)
        dst_sub = os.path.join(target_dir, sub)
        if not os.path.isdir(src_sub):
            continue
        if not os.path.exists(dst_sub):
            shutil.move(src_sub, dst_sub)
        elif os.path.isdir(dst_sub) and not os.listdir(dst_sub):
            os.rmdir(dst_sub)
            shutil.move(src_sub, dst_sub)


def ensure_workspace_dirs(paths: dict[str, str], prep_profile: str = "photo-gt") -> None:
    for key in (
        "workspace",
        "raw_zips",
        "s3d_raw",
        "s3d_data_root",
        "s3d_data",
        "runs_dir",
        "run_out",
        "hf_cache",
    ):
        os.makedirs(paths[key], exist_ok=True)
    migrate_local_legacy_s3d_data(paths, prep_profile)
    os.environ.setdefault("HF_HOME", paths["hf_cache"])


def _parse_env_file(path: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    if not os.path.exists(path):
        return parsed
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            val = val.strip()
            if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
                val = val[1:-1]
            parsed[key.strip()] = val
    return parsed


def _load_env_file(path: str) -> None:
    for key, val in _parse_env_file(path).items():
        os.environ.setdefault(key, val)


def _has_env_key(path: str, target_key: str) -> bool:
    return target_key in _parse_env_file(path)


def _upsert_env_key(path: str, target_key: str, value: str) -> None:
    lines: list[str] = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            lines = f.read().splitlines()
    entry = f"{target_key}={value}"
    updated = False
    new_lines: list[str] = []
    for raw in lines:
        stripped = raw.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key, _ = stripped.split("=", 1)
            if key.strip() == target_key:
                if not updated:
                    new_lines.append(entry)
                    updated = True
                continue
        new_lines.append(raw)
    if not updated:
        new_lines.append(entry)
    _atomic_write_text(path, "\n".join(new_lines) + "\n", mode=0o600)


def _colab_userdata_get(key: str) -> str:
    try:
        from google.colab import userdata  # type: ignore[import-not-found]

        return str(userdata.get(key) or "").strip()
    except Exception:
        return ""


def gcs_exists(uri: str) -> bool:
    return (
        subprocess.run(
            ["gcloud", "storage", "ls", uri],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode
        == 0
    )


def gcs_ls(pattern: str) -> list[str]:
    res = subprocess.run(
        ["gcloud", "storage", "ls", pattern],
        capture_output=True,
        text=True,
        check=False,
    )
    if res.returncode != 0:
        return []
    return [line.strip() for line in res.stdout.splitlines() if line.strip()]


def gcs_cp(src: str, dst: str, *, check: bool = True) -> bool:
    res = subprocess.run(
        ["gcloud", "storage", "cp", "--quiet", src, dst],
        check=check,
    )
    return res.returncode == 0


def gcs_rm(uri: str, *, check: bool = False) -> bool:
    res = subprocess.run(
        ["gcloud", "storage", "rm", "--quiet", uri],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=check,
    )
    return res.returncode == 0


def gcs_rsync(src: str, dst: str, *, check: bool = True) -> bool:
    res = subprocess.run(
        ["gcloud", "storage", "rsync", "--recursive", "--quiet", src, dst],
        check=check,
    )
    return res.returncode == 0


def _gcs_read_text(uri: str) -> str | None:
    fd, tmp_path = tempfile.mkstemp(prefix="scenealigner_gcs_")
    os.close(fd)
    try:
        res = subprocess.run(
            ["gcloud", "storage", "cp", "--quiet", uri, tmp_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if res.returncode != 0:
            return None
        with open(tmp_path, encoding="utf-8") as f:
            return f.read()
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _gcs_write_text(uri: str, content: str, *, check: bool = False) -> bool:
    fd, tmp_path = tempfile.mkstemp(prefix="scenealigner_gcs_up_")
    os.close(fd)
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(content)
        res = subprocess.run(
            ["gcloud", "storage", "cp", "--quiet", tmp_path, uri],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=check,
        )
        return res.returncode == 0
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def ensure_gcp_auth(gcs_uri: str) -> None:
    if not gcs_uri:
        return
    if not shutil.which("gcloud"):
        raise RuntimeError(
            "GCS_BACKUP_URI is configured, but `gcloud` was not found on PATH. "
            "Install Google Cloud CLI or pass --no-gcs."
        )

    def _active_account() -> str:
        return subprocess.run(
            ["gcloud", "auth", "list", "--filter=status:ACTIVE", "--format=value(account)"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()

    if _active_account():
        return
    if os.environ.get("_SCENEALIGNER_BOOTSTRAPPED") != "1" and sys.stdin.isatty():
        subprocess.run(["gcloud", "auth", "login", "--no-launch-browser"], check=True)
    if not _active_account():
        try:
            from google.colab import auth  # type: ignore[import-not-found]

            auth.authenticate_user()
        except ImportError:
            pass
    if not _active_account():
        raise RuntimeError("No active gcloud account found. Run `gcloud auth login` first or pass --no-gcs.")


def _is_nonempty_file(path: str) -> bool:
    return os.path.isfile(path) and os.path.getsize(path) > 0


def _has_local_zip(raw_zips_dir: str, filename: str) -> bool:
    local_zip = os.path.join(raw_zips_dir, filename)
    return _is_nonempty_file(local_zip) and not os.path.exists(f"{local_zip}.aria2")


def _ranges_cover(ranges: Sequence[tuple[int, int]], start_scene: int, end_scene: int) -> bool:
    pos = start_scene
    for r_start, r_end in sorted(ranges):
        if r_start <= pos < r_end:
            pos = max(pos, r_end)
            if pos >= end_scene:
                return True
    return False


def covered_scene_ranges(s3d_data_dir: str) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for path in glob.glob(os.path.join(s3d_data_dir, "scenes_*.csv")):
        m = _SLICE_CSV_RE.match(os.path.basename(path))
        if m:
            ranges.append((int(m.group(1)), int(m.group(2))))
    return sorted(ranges)


def is_slice_covered(s3d_data_dir: str, start_scene: int, end_scene: int) -> bool:
    return _ranges_cover(covered_scene_ranges(s3d_data_dir), start_scene, end_scene)


def _gcs_slice_ranges(gcs_uri: str, prep_profile: str) -> list[tuple[int, int]]:
    if not gcs_uri:
        return []
    base = gcs_uri.rstrip("/")
    uris = gcs_ls(f"{base}/s3d_data/{prep_profile}/scenes_*.tar")
    if prep_profile == "photo-gt":
        uris.extend(gcs_ls(f"{base}/s3d_data/scenes_*.tar"))
    ranges: list[tuple[int, int]] = []
    for uri in uris:
        m = _SLICE_TAR_RE.match(os.path.basename(uri))
        if m:
            ranges.append((int(m.group(1)), int(m.group(2))))
    return sorted(set(ranges))


def _eval_chunk_views_ready(s3d_raw_dir: str, scene_range: range) -> bool:
    return all(
        os.path.isdir(os.path.join(s3d_raw_dir, f"scene_{s:05d}", "2D_rendering"))
        for s in scene_range
    )


def _eval_annos_ready(s3d_raw_dir: str) -> bool:
    return all(
        os.path.exists(os.path.join(s3d_raw_dir, f"scene_{s:05d}", "annotation_3d.json"))
        for s in TEST_SCENE_RANGE
    )


def are_test_scenes_ready(s3d_raw_dir: str) -> bool:
    return all(
        os.path.exists(os.path.join(s3d_raw_dir, f"scene_{s:05d}", "annotation_3d.json"))
        and os.path.isdir(os.path.join(s3d_raw_dir, f"scene_{s:05d}", "2D_rendering"))
        for s in TEST_SCENE_RANGE
    )


def _missing_floorplan_sem_map(
    s3d_data_dir: str,
    target_ranges: Sequence[tuple[int, int]] | None = None,
) -> dict[str, str]:
    """Return {scene_id: floorplan_sem_rel} for prepared CSV rows missing floorplan_sem files on disk."""
    missing: dict[str, str] = {}
    for csv_path in sorted(glob.glob(os.path.join(s3d_data_dir, "scenes_*.csv"))):
        m = _SLICE_CSV_RE.match(os.path.basename(csv_path))
        if m and target_ranges is not None:
            s0, s1 = int(m.group(1)), int(m.group(2))
            if not any(s0 < hi and s1 > lo for lo, hi in target_ranges):
                continue
        with open(csv_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                sem_rel = (row.get("floorplan_sem") or "").strip()
                sid = (row.get("scene_id") or "").strip()
                if not sem_rel or not sid or sid in missing:
                    continue
                idx_p = os.path.join(s3d_data_dir, f"{sem_rel}_sem_idx.png")
                pal_p = os.path.join(s3d_data_dir, f"{sem_rel}_palette.npy")
                if not (os.path.exists(idx_p) and os.path.exists(pal_p)):
                    missing[sid] = sem_rel
    return missing


def _first_unmigrated_legacy_gcs_scene(
    gcs_uri: str,
    prep_profile: str,
    target_ranges: Sequence[tuple[int, int]] | None = None,
) -> str | None:
    """Return the first scene ID in a legacy flat GCS slice tar not yet archived under photo-gt/."""
    if not gcs_uri or prep_profile != "photo-gt":
        return None
    base = gcs_uri.rstrip("/")
    legacy_uris = [u for u in gcs_ls(f"{base}/s3d_data/scenes_*.tar") if u.endswith(".tar")]
    if not legacy_uris:
        return None
    migrated_names = {
        os.path.basename(u)
        for u in gcs_ls(f"{base}/s3d_data/{prep_profile}/scenes_*.tar")
        if u.endswith(".tar")
    }
    for uri in sorted(legacy_uris):
        fname = os.path.basename(uri)
        if fname in migrated_names:
            continue
        m = _SLICE_TAR_RE.match(fname)
        if not m:
            continue
        s0, s1 = int(m.group(1)), int(m.group(2))
        if target_ranges is not None and not any(s0 < hi and s1 > lo for lo, hi in target_ranges):
            continue
        return f"scene_{s0:05d}"
    return None


def needs_external_s3d_download(
    args: argparse.Namespace,
    paths: dict[str, str],
    gcs_uri: str,
    prep_profile: str,
) -> bool:
    """True only when prep/eval/floorplan_sem repair requires fetching a zip from S3D_BASE_URL."""
    base = gcs_uri.rstrip("/")
    anno_in_cache = _has_local_zip(paths["raw_zips"], "Structured3D_annotation_3d.zip") or bool(
        base and gcs_exists(f"{base}/raw_zips/Structured3D_annotation_3d.zip")
    )

    if any(s in args.stages for s in ("prep", "train", "backup")):
        heal_ranges = None if "backup" in args.stages else _chunks_to_scene_ranges(_resolve_chunks(args))
        missing_fp = _missing_floorplan_sem_map(paths["s3d_data"], target_ranges=heal_ranges)
        if missing_fp:
            first_sid = next(iter(missing_fp))
            if not os.path.exists(os.path.join(paths["s3d_raw"], first_sid, "annotation_3d.json")) and not anno_in_cache:
                return True
        legacy_sid = _first_unmigrated_legacy_gcs_scene(gcs_uri, prep_profile, target_ranges=heal_ranges)
        if legacy_sid is not None:
            if not os.path.exists(os.path.join(paths["s3d_raw"], legacy_sid, "annotation_3d.json")) and not anno_in_cache:
                return True

    if "prep" in args.stages:
        covered = covered_scene_ranges(paths["s3d_data"]) + _gcs_slice_ranges(gcs_uri, prep_profile)
        for chunk_idx in _resolve_chunks(args):
            c_start, c_end = chunk_idx * 200, (chunk_idx + 1) * 200
            slices = [(s, min(s + args.slice_scenes, c_end)) for s in range(c_start, c_end, args.slice_scenes)]
            pending = [sl for sl in slices if not _ranges_cover(covered, *sl)]
            if not pending:
                continue
            anno_expected = os.path.join(paths["s3d_raw"], f"scene_{pending[0][0]:05d}", "annotation_3d.json")
            if not os.path.exists(anno_expected) and not anno_in_cache:
                return True
            zip_name = f"Structured3D_perspective_full_{chunk_idx:02d}.zip"
            views_ready = all(
                os.path.isdir(os.path.join(paths["s3d_raw"], f"scene_{s0:05d}", "2D_rendering"))
                for s0, _ in pending
            )
            zip_cached = _has_local_zip(paths["raw_zips"], zip_name) or bool(
                base and gcs_exists(f"{base}/raw_zips/{zip_name}")
            )
            if not views_ready and not zip_cached:
                return True

    need_eval_scenes = "eval" in args.stages or ("train" in args.stages and _inter_eval_enabled(args))
    if need_eval_scenes and not are_test_scenes_ready(paths["s3d_raw"]):
        test_tar_in_gcs = bool(base and gcs_exists(f"{base}/s3d_eval/{TEST_SCENES_TAR}"))
        if not test_tar_in_gcs:
            for chunk_idx, scene_range in _EVAL_CHUNK_RANGES:
                if _eval_chunk_views_ready(paths["s3d_raw"], scene_range):
                    continue
                zip_name = f"Structured3D_perspective_full_{chunk_idx:02d}.zip"
                if not _has_local_zip(paths["raw_zips"], zip_name) and not (
                    base and gcs_exists(f"{base}/raw_zips/{zip_name}")
                ):
                    return True
            if not _eval_annos_ready(paths["s3d_raw"]) and not anno_in_cache:
                return True

    return False


def _sync_env_with_gcs(env_file: str, gcs_uri: str) -> None:
    if not gcs_uri:
        return
    gcs_env = f"{gcs_uri.rstrip('/')}/config/.env"
    local_env = _parse_env_file(env_file)
    if not local_env.get("S3D_BASE_URL", "").strip() and gcs_exists(gcs_env):
        text = _gcs_read_text(gcs_env)
        if text:
            for raw in text.splitlines():
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                k = key.strip()
                v = val.strip().strip("\"'")
                if v and not local_env.get(k, "").strip():
                    _upsert_env_key(env_file, k, v)
                    local_env[k] = v


def _upload_env_if_needed(
    env_file: str,
    gcs_uri: str,
    *,
    updated: bool,
    verified: bool = False,
) -> None:
    if not gcs_uri or not os.path.exists(env_file):
        return
    local_env = _parse_env_file(env_file)
    raw_url = local_env.get("S3D_BASE_URL", "").strip()
    if not raw_url:
        return
    gcs_env = f"{gcs_uri.rstrip('/')}/config/.env"
    if not updated and gcs_exists(gcs_env):
        return
    if not verified:
        try:
            _verify_s3d_base_url(_normalize_s3d_url(raw_url))
        except (ValueError, RuntimeError):
            return
    gcs_cp(env_file, gcs_env, check=False)


def resolve_config(
    args: argparse.Namespace,
    paths: dict[str, str],
    *,
    verify_gcp_auth: bool = False,
) -> dict[str, str]:
    env_file = paths["env_file"]
    had_proc_s3d = bool(os.environ.get("S3D_BASE_URL", "").strip())
    pre_local_env = _parse_env_file(env_file)
    repo_env = _parse_env_file(os.path.join(REPO_DIR, ".env"))

    if args.no_gcs:
        gcs_uri = ""
    elif args.gcs_uri is not None:
        gcs_uri = _normalize_gcs_uri(args.gcs_uri)
    elif os.environ.get("GCS_BACKUP_URI", "").strip():
        gcs_uri = _normalize_gcs_uri(os.environ["GCS_BACKUP_URI"])
    elif "GCS_BACKUP_URI" in pre_local_env:
        gcs_uri = _normalize_gcs_uri(pre_local_env["GCS_BACKUP_URI"])
    elif "GCS_BACKUP_URI" in repo_env:
        gcs_uri = _normalize_gcs_uri(repo_env["GCS_BACKUP_URI"])
    else:
        gcs_uri = DEFAULT_GCS_BACKUP_URI
    os.environ["GCS_BACKUP_URI"] = gcs_uri

    if verify_gcp_auth and gcs_uri:
        ensure_gcp_auth(gcs_uri)
    if gcs_uri and (verify_gcp_auth or os.environ.get("_SCENEALIGNER_SYNC_GCS_ENV") == "1"):
        _sync_env_with_gcs(env_file, gcs_uri)

    _load_env_file(env_file)
    _load_env_file(os.path.join(REPO_DIR, ".env"))
    for extra_key in ("HF_TOKEN", "GITHUB_TOKEN", "S3D_BASE_URL"):
        if not os.environ.get(extra_key, "").strip():
            secret_val = _colab_userdata_get(extra_key)
            if secret_val:
                os.environ[extra_key] = secret_val

    bootstrapped = os.environ.get("_SCENEALIGNER_BOOTSTRAPPED") == "1"
    is_tty = sys.stdin.isatty() and sys.stdout.isatty()
    prep_profile = compute_prep_profile(
        args.photo_sem_source, args.include_rooms, args.structural_photo_semantics
    )
    needs_external = False
    if not bootstrapped and (verify_gcp_auth or not gcs_uri) and any(
        s in args.stages for s in ("prep", "train", "eval", "backup")
    ):
        needs_external = needs_external_s3d_download(args, paths, gcs_uri, prep_profile)

    s3d_url = ""
    env_updated = False
    s3d_url_verified = False
    had_invalid_saved_url = False
    if args.s3d_base_url is not None:
        s3d_url = _normalize_s3d_url(args.s3d_base_url)
        if needs_external and not bootstrapped:
            _verify_s3d_base_url(s3d_url)
            s3d_url_verified = True
            if s3d_url and s3d_url != pre_local_env.get("S3D_BASE_URL", ""):
                _upsert_env_key(env_file, "S3D_BASE_URL", s3d_url)
                env_updated = True
    elif os.environ.get("S3D_BASE_URL", "").strip():
        raw_env_val = os.environ["S3D_BASE_URL"]
        s3d_url = _normalize_s3d_url(raw_env_val)
        if needs_external and not bootstrapped and not had_proc_s3d:
            try:
                _verify_s3d_base_url(s3d_url)
            except (ValueError, RuntimeError) as err:
                colab_url = _normalize_s3d_url(_colab_userdata_get("S3D_BASE_URL"))
                if colab_url and colab_url != s3d_url:
                    try:
                        _verify_s3d_base_url(colab_url)
                    except (ValueError, RuntimeError):
                        colab_url = ""
                else:
                    colab_url = ""
                if colab_url:
                    s3d_url = colab_url
                    s3d_url_verified = True
                    _upsert_env_key(env_file, "S3D_BASE_URL", s3d_url)
                    env_updated = True
                elif is_tty:
                    sys.stderr.write(
                        f"[run_vm] Saved S3D_BASE_URL in {env_file} is invalid ({err}); please re-enter.\n"
                    )
                    sys.stderr.flush()
                    had_invalid_saved_url = True
                    s3d_url = ""
                else:
                    raise
            else:
                s3d_url_verified = True
                if _has_env_key(env_file, "S3D_BASE_URL") and s3d_url != raw_env_val.strip():
                    _upsert_env_key(env_file, "S3D_BASE_URL", s3d_url)
                    env_updated = True

    if not s3d_url and not bootstrapped and needs_external and is_tty:
        for attempt in range(3):
            entered = getpass.getpass("S3D_BASE_URL (masked, saved to .env; Enter to skip): ")
            candidate = _normalize_s3d_url(entered)
            if not candidate:
                break
            try:
                _verify_s3d_base_url(candidate)
            except (ValueError, RuntimeError) as err:
                if attempt == 2:
                    raise RuntimeError(str(err)) from None
                sys.stderr.write(f"[run_vm] {err} Try again (or press Enter to skip).\n")
                sys.stderr.flush()
            else:
                s3d_url = candidate
                s3d_url_verified = True
                _upsert_env_key(env_file, "S3D_BASE_URL", s3d_url)
                env_updated = True
                break
        if had_invalid_saved_url and not s3d_url and _has_env_key(env_file, "S3D_BASE_URL"):
            _upsert_env_key(env_file, "S3D_BASE_URL", "")

    os.environ["S3D_BASE_URL"] = s3d_url
    if gcs_uri and verify_gcp_auth:
        _upload_env_if_needed(env_file, gcs_uri, updated=env_updated, verified=s3d_url_verified)

    return {"gcs_backup_uri": gcs_uri, "s3d_base_url": s3d_url}


def _parse_manifest_json(raw_text: str | None) -> dict[str, Any] | None:
    if not raw_text:
        return None
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _read_local_and_remote_manifests(
    run_id: str, workspace: str, gcs_uri: str
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    local_manifest_path = os.path.join(workspace, "runs", run_id, "run_manifest.json")
    local_m: dict[str, Any] | None = None
    if os.path.exists(local_manifest_path):
        try:
            with open(local_manifest_path, encoding="utf-8") as f:
                local_m = _parse_manifest_json(f.read())
        except OSError:
            local_m = None
    remote_m: dict[str, Any] | None = None
    if gcs_uri:
        remote_m = _parse_manifest_json(_gcs_read_text(f"{gcs_uri.rstrip('/')}/runs/{run_id}/run_manifest.json"))
    return local_m, remote_m


def _read_manifest_dict(run_id: str, workspace: str, gcs_uri: str) -> dict[str, Any] | None:
    local_m, remote_m = _read_local_and_remote_manifests(run_id, workspace, gcs_uri)
    if local_m is not None and remote_m is not None:
        local_done = local_m.get("status") == "completed"
        remote_done = remote_m.get("status") == "completed"
        if remote_done != local_done:
            return remote_m if remote_done else local_m
        local_ts = str(local_m.get("updated_at_utc") or "")
        remote_ts = str(remote_m.get("updated_at_utc") or "")
        return local_m if local_ts > remote_ts else remote_m
    return local_m if local_m is not None else remote_m


def resolve_run_id(args: argparse.Namespace, workspace: str, gcs_uri: str) -> tuple[str, bool]:
    explicit = getattr(args, "run_id", None)
    if explicit:
        cleaned = str(explicit).strip()
        if not cleaned or "/" in cleaned or "\\" in cleaned or ".." in cleaned:
            raise ValueError(f"Invalid --run-id {explicit!r}")
        manifest = _read_manifest_dict(cleaned, workspace, gcs_uri)
        resumed = bool(manifest is not None and manifest.get("status") == "in_progress")
        return cleaned, resumed

    track_active = "train" in args.stages or "eval" in args.stages
    if track_active and not getattr(args, "new_run", False):
        active_path = os.path.join(workspace, "runs", "ACTIVE_RUN")
        local_id = ""
        if os.path.exists(active_path):
            with open(active_path, encoding="utf-8") as f:
                local_id = f.read().strip()
        remote_id = ""
        if gcs_uri:
            remote_active = _gcs_read_text(f"{gcs_uri.rstrip('/')}/runs/ACTIVE_RUN")
            if remote_active:
                remote_id = remote_active.strip()

        def _is_valid_id(cid: str) -> bool:
            return bool(cid and "/" not in cid and "\\" not in cid and ".." not in cid)

        if _is_valid_id(local_id):
            local_manifest = _read_manifest_dict(local_id, workspace, gcs_uri)
            if local_manifest is not None and local_manifest.get("status") == "completed":
                os.remove(active_path)
                local_id = ""

        candidates: list[str] = []
        for cid in (remote_id, local_id):
            if _is_valid_id(cid) and cid not in candidates:
                candidates.append(cid)

        target_fp = compute_run_fingerprint(args)
        for cid in candidates:
            manifest = _read_manifest_dict(cid, workspace, gcs_uri)
            if (
                manifest is not None
                and manifest.get("status") == "in_progress"
                and manifest.get("config_fingerprint") == target_fp
            ):
                return cid, True

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%SZ")
    return f"{ts}_{build_run_slug(args)}", False


def restore_run_from_gcs(
    paths: dict[str, str],
    gcs_uri: str,
    *,
    allow_log_overwrite: bool = False,
) -> bool:
    """Restore persisted checkpoints, eval progress, and logs from GCS into run_out."""
    base = gcs_uri.rstrip("/")
    if not base:
        return False
    run_out = paths["run_out"]
    run_id = os.path.basename(run_out)
    remote_dir = f"{base}/runs/{run_id}"
    remote_files = [u for u in gcs_ls(f"{remote_dir}/*") if u and not u.endswith(("/", ":"))]
    epochs_remote = gcs_exists(f"{remote_dir}/epochs/*")
    if not remote_files and not epochs_remote:
        return False
    os.makedirs(run_out, exist_ok=True)
    if allow_log_overwrite and not os.path.exists(paths["log_path"]) and not os.path.exists(paths["manifest_json"]):
        return gcs_rsync(remote_dir, run_out, check=False)

    local_m, remote_m = _read_local_and_remote_manifests(run_id, paths["workspace"], base)
    remote_newer = remote_m is not None and (
        local_m is None
        or (remote_m.get("status") == "completed" and local_m.get("status") != "completed")
        or str(remote_m.get("updated_at_utc") or "") > str(local_m.get("updated_at_utc") or "")
    )
    mutable_exact = {
        "checkpoint.pt",
        "adapter_config.json",
        "adapter_model.safetensors",
        "semantic_adapter.pt",
        "log.csv",
        "train_state.json",
        "epoch_evals.jsonl",
    }
    skip_names = {"pipeline.log", "run_manifest.json"}
    restored_any = False
    remote_prefix = f"{remote_dir}/"
    for uri in sorted(remote_files):
        rel = _strip_prefix(uri.rstrip("/"), remote_prefix)
        if not rel or rel == uri.rstrip("/") or rel in skip_names:
            continue
        if rel == "epochs" or rel.startswith("epochs/"):
            continue
        dst = os.path.join(run_out, rel)
        fname = os.path.basename(rel)
        if not fname:
            continue
        is_mutable = rel in mutable_exact or fname.startswith(("eval_progress_", "eval_s3d_"))
        if not os.path.exists(dst) or (remote_newer and is_mutable):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if gcs_cp(uri, dst, check=False):
                restored_any = True
    epochs_dir = os.path.join(run_out, "epochs")
    if epochs_remote and (not os.path.isdir(epochs_dir) or remote_newer):
        if gcs_rsync(f"{remote_dir}/epochs", epochs_dir, check=False):
            restored_any = True
    return restored_any


def init_run_manifest(
    args: argparse.Namespace,
    paths: dict[str, str],
    cfg: dict[str, str],
    *,
    resumed: bool,
) -> dict[str, Any]:
    run_id = os.path.basename(paths["run_out"])
    prep_profile = compute_prep_profile(
        args.photo_sem_source, args.include_rooms, args.structural_photo_semantics
    )
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    existing = _read_manifest_dict(run_id, paths["workspace"], cfg["gcs_backup_uri"])
    started_at = (
        str(existing["started_at_utc"])
        if isinstance(existing, dict) and existing.get("started_at_utc")
        else now_utc
    )
    eff_resumed = resumed or bool(isinstance(existing, dict) and existing.get("resumed"))
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "status": "in_progress",
        "resumed": eff_resumed,
        "started_at_utc": started_at,
        "updated_at_utc": now_utc,
        "git_commit": _git_short_head(),
        "photo_sem_source": str(args.photo_sem_source),
        "floorplan_sem_source": "s3d_annos",
        "prep_profile": prep_profile,
        "stages": list(args.stages),
        "config_fingerprint": compute_run_fingerprint(args),
    }
    _atomic_write_text(paths["manifest_json"], json.dumps(manifest, indent=2) + "\n")
    track_active = "train" in args.stages or "eval" in args.stages
    gcs_uri = cfg["gcs_backup_uri"].rstrip("/")
    if track_active:
        _atomic_write_text(paths["active_run_file"], f"{run_id}\n")
        if gcs_uri:
            gcs_cp(paths["manifest_json"], f"{gcs_uri}/runs/{run_id}/run_manifest.json", check=False)
            gcs_cp(paths["active_run_file"], f"{gcs_uri}/runs/ACTIVE_RUN", check=False)
    return manifest


def preflight_data_sources(
    args: argparse.Namespace,
    cfg: dict[str, str],
    paths: dict[str, str],
) -> None:
    """Fail fast upfront if required Structured3D downloads or GCS archives are unavailable."""
    if int(getattr(args, "eval_every_epochs", 1)) < 0:
        raise ValueError(f"--eval-every-epochs must be non-negative, got {args.eval_every_epochs}")
    if int(getattr(args, "inter_eval_scenes", 50)) < 0:
        raise ValueError(f"--inter-eval-scenes must be non-negative, got {args.inter_eval_scenes}")

    if "backup" in args.stages and not cfg["gcs_backup_uri"]:
        raise RuntimeError("The `backup` stage requires GCS_BACKUP_URI (--gcs-uri or .env).")

    if "train" in args.stages and args.freeze_lora and args.sem_mode == "none" and (args.alpha_sem or 0.0) == 0.0:
        raise ValueError("Cannot use --freeze-lora with --sem-mode none and alpha_sem=0 (pass --no-freeze-lora).")

    if "train" in args.stages and args.photo_sem_source == "none" and (
        args.sem_mode != "none" or (args.alpha_sem or 0.0) > 0.0
    ):
        raise ValueError(
            "Training with --photo-sem-source none has no 2D photo semantics; "
            "pass --sem-mode none --no-freeze-lora (or run --stages eval only)."
        )

    gcs_uri = cfg["gcs_backup_uri"].rstrip("/")
    prep_profile = compute_prep_profile(
        args.photo_sem_source, args.include_rooms, args.structural_photo_semantics
    )
    if "train" in args.stages and "prep" not in args.stages:
        chunks = _resolve_chunks(args)
        available = covered_scene_ranges(paths["s3d_data"]) + _gcs_slice_ranges(gcs_uri, prep_profile)
        missing_chunks = [
            c
            for c, (s0, s1) in zip(chunks, _chunks_to_scene_ranges(chunks))
            if not _ranges_cover(available, s0, s1)
        ]
        if missing_chunks:
            raise RuntimeError(
                f"Train stage requires prepared slices for chunks {missing_chunks} (profile={prep_profile}), "
                "but they were not found locally or in GCS. Include `prep` in --stages first."
            )

    s3d_url = cfg["s3d_base_url"]
    if s3d_url:
        return

    if any(s in args.stages for s in ("prep", "train", "backup")):
        heal_ranges = None if "backup" in args.stages else _chunks_to_scene_ranges(_resolve_chunks(args))
        missing_fp = _missing_floorplan_sem_map(paths["s3d_data"], target_ranges=heal_ranges)
        legacy_sid = _first_unmigrated_legacy_gcs_scene(gcs_uri, prep_profile, target_ranges=heal_ranges)
        heal_sid = next(iter(missing_fp)) if missing_fp else legacy_sid
        if heal_sid is not None and not os.path.exists(os.path.join(paths["s3d_raw"], heal_sid, "annotation_3d.json")):
            anno_zip = "Structured3D_annotation_3d.zip"
            in_gcs = bool(gcs_uri and gcs_exists(f"{gcs_uri}/raw_zips/{anno_zip}"))
            if not _has_local_zip(paths["raw_zips"], anno_zip) and not in_gcs:
                raise RuntimeError(
                    f"Floorplan semantic healing requires {anno_zip}, which is not in {paths['raw_zips']} or GCS; provide S3D_BASE_URL."
                )

    if "prep" in args.stages and not gcs_uri:
        for chunk_idx in _resolve_chunks(args):
            c_start, c_end = chunk_idx * 200, (chunk_idx + 1) * 200
            slices = [(s, min(s + args.slice_scenes, c_end)) for s in range(c_start, c_end, args.slice_scenes)]
            pending = [sl for sl in slices if not is_slice_covered(paths["s3d_data"], *sl)]
            if not pending:
                continue
            zip_name = f"Structured3D_perspective_full_{chunk_idx:02d}.zip"
            views_ready = all(
                os.path.isdir(os.path.join(paths["s3d_raw"], f"scene_{s0:05d}", "2D_rendering"))
                for s0, _ in pending
            )
            if not views_ready and not _has_local_zip(paths["raw_zips"], zip_name):
                raise RuntimeError(
                    f"Prep requires {zip_name} (chunk {chunk_idx}), but S3D_BASE_URL and GCS_BACKUP_URI are empty."
                )

    need_eval_scenes = "eval" in args.stages or ("train" in args.stages and _inter_eval_enabled(args))
    if need_eval_scenes and not are_test_scenes_ready(paths["s3d_raw"]):
        if not (gcs_uri and gcs_exists(f"{gcs_uri}/s3d_eval/{TEST_SCENES_TAR}")):
            missing_chunks = [
                c
                for c, scene_range in _EVAL_CHUNK_RANGES
                if not _eval_chunk_views_ready(paths["s3d_raw"], scene_range)
            ]
            for chunk_idx in missing_chunks:
                zip_name = f"Structured3D_perspective_full_{chunk_idx:02d}.zip"
                in_gcs = bool(gcs_uri and gcs_exists(f"{gcs_uri}/raw_zips/{zip_name}"))
                if not _has_local_zip(paths["raw_zips"], zip_name) and not in_gcs:
                    raise RuntimeError(
                        f"Eval requires {zip_name} (or s3d_eval/{TEST_SCENES_TAR} in GCS), which is unavailable; provide S3D_BASE_URL."
                    )
            if not _eval_annos_ready(paths["s3d_raw"]):
                anno_zip = "Structured3D_annotation_3d.zip"
                in_gcs = bool(gcs_uri and gcs_exists(f"{gcs_uri}/raw_zips/{anno_zip}"))
                if not _has_local_zip(paths["raw_zips"], anno_zip) and not in_gcs:
                    raise RuntimeError(
                        f"Eval requires {anno_zip}, which is not in {paths['raw_zips']} or GCS; provide S3D_BASE_URL."
                    )


def fetch_zip(
    filename: str,
    paths: dict[str, str],
    cfg: dict[str, str],
    log_writer: _CleanLogWriter,
) -> str:
    local_zip = os.path.join(paths["raw_zips"], filename)
    gcs_uri = cfg["gcs_backup_uri"].rstrip("/")
    gcs_zip = f"{gcs_uri}/raw_zips/{filename}" if gcs_uri else ""

    if _has_local_zip(paths["raw_zips"], filename):
        if filename == "Structured3D_annotation_3d.zip" and gcs_zip and not gcs_exists(gcs_zip):
            log_writer.log_line(f"Persisting {filename} to {gcs_zip}...")
            gcs_cp(local_zip, gcs_zip, check=False)
        return local_zip

    if gcs_zip and gcs_exists(gcs_zip):
        log_writer.log_line(f"Fetching {filename} from {gcs_zip}...")
        try:
            gcs_cp(gcs_zip, local_zip, check=True)
            return local_zip
        except (subprocess.CalledProcessError, OSError) as exc:
            if os.path.exists(local_zip):
                os.remove(local_zip)
            if not cfg["s3d_base_url"].strip():
                raise
            log_writer.log_line(
                f"[warn] GCS fetch failed for {filename} ({exc}); falling back to S3D_BASE_URL."
            )

    base = cfg["s3d_base_url"].rstrip("/")
    if not base:
        raise RuntimeError(f"Missing {local_zip} and S3D_BASE_URL is not set.")
    url = f"{base}/{filename}"
    log_writer.log_line(f"Downloading {filename} via aria2c...")
    run_live(
        [
            "aria2c",
            "-c",
            "-x",
            "16",
            "-s",
            "16",
            "--summary-interval=0",
            "-d",
            paths["raw_zips"],
            "-o",
            filename,
            url,
        ],
        log_writer,
        redact=base,
    )
    if filename == "Structured3D_annotation_3d.zip" and gcs_zip and os.path.exists(local_zip):
        log_writer.log_line(f"Persisting {filename} to {gcs_zip}...")
        gcs_cp(local_zip, gcs_zip, check=False)
    return local_zip


def extract_7z(
    zip_path: str,
    patterns: list[str],
    expected_path: str,
    workspace: str,
    *,
    delete_zip: bool,
    log_writer: _CleanLogWriter,
) -> None:
    bin_7z = _find_7z(required=True)
    assert bin_7z is not None
    log_writer.log_line(f"Extracting {os.path.basename(zip_path)}...")
    try:
        subprocess.run(
            [bin_7z, "x", "-y", zip_path, f"-o{workspace}", *patterns],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        if not os.path.exists(expected_path):
            raise RuntimeError(f"Expected {expected_path} after extracting {zip_path}")
    except (subprocess.CalledProcessError, RuntimeError):
        if os.path.exists(zip_path):
            os.remove(zip_path)
        raise
    if delete_zip and os.path.exists(zip_path):
        os.remove(zip_path)


def _safe_extract_tar(tar_path: str, dest_dir: str) -> None:
    os.makedirs(dest_dir, exist_ok=True)
    supports_filter = "filter" in inspect.signature(tarfile.TarFile.extractall).parameters
    with tarfile.open(tar_path, "r:*") as tf:
        for member in tf.getmembers():
            parts = member.name.split("/")
            if member.name.startswith("/") or ".." in parts or member.issym() or member.islnk():
                raise ValueError(f"Unsafe tar member: {member.name}")
        if supports_filter:
            tf.extractall(dest_dir, filter="data")
        else:
            tf.extractall(dest_dir)


def archive_s3d_slice(
    paths: dict[str, str],
    gcs_uri: str,
    start_scene: int,
    end_scene: int,
    *,
    prep_profile: str = "photo-gt",
    overwrite: bool = False,
) -> None:
    if not gcs_uri:
        return
    base = gcs_uri.rstrip("/")
    tar_name = f"scenes_{start_scene}_{end_scene}.tar"
    gcs_tar = f"{base}/s3d_data/{prep_profile}/{tar_name}"
    if not overwrite and gcs_exists(gcs_tar):
        return
    s3d_data_dir = paths["s3d_data"]
    csv_name = f"scenes_{start_scene}_{end_scene}.csv"
    csv_path = os.path.join(s3d_data_dir, csv_name)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Cannot archive missing slice CSV {csv_path}")
    local_tar = os.path.join(paths["workspace"], tar_name)
    try:
        with tarfile.open(local_tar, "w") as tf:
            tf.add(csv_path, arcname=csv_name)
            for s in range(start_scene, end_scene):
                sid = f"scene_{s:05d}"
                fp = os.path.join(s3d_data_dir, "floorplan", f"{sid}.png")
                if os.path.exists(fp):
                    tf.add(fp, arcname=f"floorplan/{sid}.png")
                for p in glob.glob(os.path.join(s3d_data_dir, "floorplan_sem", f"{sid}_*")):
                    tf.add(p, arcname=os.path.relpath(p, s3d_data_dir))
                for sub in ("depth", "pi3", "depth_sem", "pi3_sem"):
                    sdir = os.path.join(s3d_data_dir, sub, sid)
                    if os.path.isdir(sdir):
                        tf.add(sdir, arcname=f"{sub}/{sid}")
        gcs_cp(local_tar, gcs_tar, check=True)
    finally:
        if os.path.exists(local_tar):
            os.remove(local_tar)


def restore_s3d_tars(
    paths: dict[str, str],
    gcs_uri: str,
    log_writer: _CleanLogWriter,
    *,
    prep_profile: str = "photo-gt",
    target_ranges: Sequence[tuple[int, int]] | None = None,
) -> None:
    if not gcs_uri:
        return
    base = gcs_uri.rstrip("/")
    s3d_data_dir = paths["s3d_data"]

    profile_uris = [u for u in gcs_ls(f"{base}/s3d_data/{prep_profile}/scenes_*.tar") if u.endswith(".tar")]
    uris = list(profile_uris)
    if prep_profile == "photo-gt":
        profile_names = {os.path.basename(u) for u in profile_uris}
        legacy_uris = [u for u in gcs_ls(f"{base}/s3d_data/scenes_*.tar") if u.endswith(".tar")]
        for u in legacy_uris:
            if os.path.basename(u) not in profile_names:
                uris.append(u)

    if not uris:
        if not glob.glob(os.path.join(s3d_data_dir, "scenes_*.csv")) and gcs_exists(
            f"{base}/s3d_data/{prep_profile}/scenes_*.csv"
        ):
            gcs_rsync(f"{base}/s3d_data/{prep_profile}", s3d_data_dir)
        return

    for uri in sorted(uris, key=os.path.basename):
        fname = os.path.basename(uri)
        m = _SLICE_TAR_RE.match(fname)
        s0 = s1 = None
        if m:
            s0, s1 = int(m.group(1)), int(m.group(2))
            if target_ranges is not None and not any(s0 < hi and s1 > lo for lo, hi in target_ranges):
                continue
            if is_slice_covered(s3d_data_dir, s0, s1):
                continue
        local_tar = os.path.join(paths["workspace"], fname)
        expected_csv = os.path.join(s3d_data_dir, f"scenes_{s0}_{s1}.csv") if s0 is not None else ""
        try:
            log_writer.log_line(f"Restoring {prep_profile}/{fname} from GCS...")
            gcs_cp(uri, local_tar, check=True)
            _safe_extract_tar(local_tar, s3d_data_dir)
            if expected_csv and not os.path.exists(expected_csv):
                raise ValueError(f"Missing expected {os.path.basename(expected_csv)} inside {fname}")
        except (tarfile.TarError, subprocess.CalledProcessError, OSError, EOFError, ValueError) as exc:
            if expected_csv and os.path.exists(expected_csv):
                os.remove(expected_csv)
            log_writer.log_line(
                f"[warn] Corrupt or unreadable GCS slice archive {uri} ({exc}); removing broken remote archive."
            )
            gcs_rm(uri, check=False)
        finally:
            if os.path.exists(local_tar):
                os.remove(local_tar)


def _finalize_legacy_gcs_s3d_tars(
    paths: dict[str, str],
    gcs_uri: str,
    log_writer: _CleanLogWriter,
    *,
    prep_profile: str,
) -> None:
    """Promote verified legacy flat GCS slice tars into photo-gt/ only once floorplan_sem is intact."""
    if not gcs_uri or prep_profile != "photo-gt":
        return
    base = gcs_uri.rstrip("/")
    s3d_data_dir = paths["s3d_data"]
    legacy_uris = [u for u in gcs_ls(f"{base}/s3d_data/scenes_*.tar") if u.endswith(".tar")]
    for src in sorted(legacy_uris):
        fname = os.path.basename(src)
        m = _SLICE_TAR_RE.match(fname)
        dst = f"{base}/s3d_data/{prep_profile}/{fname}"
        if gcs_exists(dst):
            gcs_rm(src, check=False)
            continue
        if not m:
            continue
        s0, s1 = int(m.group(1)), int(m.group(2))
        if not is_slice_covered(s3d_data_dir, s0, s1):
            continue
        if _missing_floorplan_sem_map(s3d_data_dir, target_ranges=[(s0, s1)]):
            continue
        log_writer.log_line(f"Migrating verified legacy GCS slice {fname} -> s3d_data/{prep_profile}/{fname}...")
        archive_s3d_slice(paths, base, s0, s1, prep_profile=prep_profile, overwrite=False)
        if gcs_exists(dst):
            gcs_rm(src, check=False)


def ensure_floorplan_sem(
    args: argparse.Namespace,
    paths: dict[str, str],
    cfg: dict[str, str],
    log_writer: _CleanLogWriter,
    *,
    prep_profile: str | None = None,
    target_ranges: Sequence[tuple[int, int]] | None = None,
) -> None:
    s3d_data_dir = paths["s3d_data"]
    s3d_raw_dir = paths["s3d_raw"]
    eff_profile = prep_profile or compute_prep_profile(
        args.photo_sem_source, args.include_rooms, args.structural_photo_semantics
    )
    gcs_uri = cfg["gcs_backup_uri"].rstrip("/")
    missing = _missing_floorplan_sem_map(s3d_data_dir, target_ranges=target_ranges)
    if not missing:
        _finalize_legacy_gcs_s3d_tars(paths, gcs_uri, log_writer, prep_profile=eff_profile)
        return

    first_sid = next(iter(missing))
    anno_expected = os.path.join(s3d_raw_dir, first_sid, "annotation_3d.json")
    if not os.path.exists(anno_expected):
        extract_7z(
            fetch_zip("Structured3D_annotation_3d.zip", paths, cfg, log_writer),
            ["Structured3D/*/annotation_3d.json"],
            anno_expected,
            paths["workspace"],
            delete_zip=not args.keep_zips,
            log_writer=log_writer,
        )

    todo_path = os.path.join(paths["workspace"], "_floorplan_sem_todo.json")
    script = "\n".join(
        [
            "import json, os, sys",
            "from tqdm import tqdm",
            f"sys.path.insert(0, {REPO_DIR!r})",
            "from scenealigner import structured3d as s3d",
            "from scenealigner.semantics import ClipTextEncoder, save_semantic_map",
            "items = json.load(open(sys.argv[1], encoding='utf-8'))",
            "enc = ClipTextEncoder()",
            f"for sid, sem_rel in tqdm(items, desc='floorplan_sem ({len(missing)})'):",
            f"    sdir = os.path.join({s3d_raw_dir!r}, sid)",
            "    idx, pal, lbls = s3d.load_semantic_floorplan(",
            f"        sdir, text_encoder=enc, include_rooms={args.include_rooms!r}",
            "    )",
            f"    out = os.path.join({s3d_data_dir!r}, sem_rel)",
            "    save_semantic_map(out, idx, pal, labels=lbls)",
        ]
    )
    try:
        _atomic_write_text(todo_path, json.dumps(sorted(missing.items())))
        run_live([sys.executable, "-c", script, todo_path], log_writer)
    finally:
        if os.path.exists(todo_path):
            os.remove(todo_path)

    if gcs_uri:
        missing_nums = [n for sid in missing if (n := _parse_scene_int(sid)) is not None]
        for s_start, s_end in covered_scene_ranges(s3d_data_dir):
            if not missing_nums or any(s_start <= n < s_end for n in missing_nums):
                log_writer.log_line(
                    f"Healing GCS archive s3d_data/{eff_profile}/scenes_{s_start}_{s_end}.tar with floorplan_sem..."
                )
                archive_s3d_slice(
                    paths,
                    gcs_uri,
                    s_start,
                    s_end,
                    prep_profile=eff_profile,
                    overwrite=True,
                )
        _finalize_legacy_gcs_s3d_tars(paths, gcs_uri, log_writer, prep_profile=eff_profile)


def archive_eval_test_scenes(
    paths: dict[str, str],
    gcs_uri: str,
    log_writer: _CleanLogWriter,
    *,
    overwrite: bool = False,
) -> None:
    if not gcs_uri:
        return
    base = gcs_uri.rstrip("/")
    gcs_tar = f"{base}/s3d_eval/{TEST_SCENES_TAR}"
    if not overwrite and gcs_exists(gcs_tar):
        return
    if not are_test_scenes_ready(paths["s3d_raw"]):
        raise RuntimeError("Cannot archive test_scenes_3250_3500.tar before all test scenes are ready locally.")
    local_tar = os.path.join(paths["workspace"], TEST_SCENES_TAR)
    try:
        log_writer.log_line(f"Archiving Structured3D test scenes (3250..3499) -> {gcs_tar}...")
        with tarfile.open(local_tar, "w") as tf:
            for s in TEST_SCENE_RANGE:
                sid = f"scene_{s:05d}"
                sdir = os.path.join(paths["s3d_raw"], sid)
                anno = os.path.join(sdir, "annotation_3d.json")
                views_dir = os.path.join(sdir, "2D_rendering")
                if os.path.exists(anno):
                    tf.add(anno, arcname=f"{sid}/annotation_3d.json")
                if os.path.isdir(views_dir):
                    tf.add(views_dir, arcname=f"{sid}/2D_rendering")
        gcs_cp(local_tar, gcs_tar, check=True)
    finally:
        if os.path.exists(local_tar):
            os.remove(local_tar)


def ensure_eval_test_scenes(
    args: argparse.Namespace,
    paths: dict[str, str],
    cfg: dict[str, str],
    log_writer: _CleanLogWriter,
) -> None:
    gcs_uri = cfg["gcs_backup_uri"].rstrip("/")
    gcs_tar = f"{gcs_uri}/s3d_eval/{TEST_SCENES_TAR}" if gcs_uri else ""

    if are_test_scenes_ready(paths["s3d_raw"]):
        if gcs_tar and not gcs_exists(gcs_tar):
            archive_eval_test_scenes(paths, gcs_uri, log_writer)
        return

    had_broken_gcs_tar = False
    if gcs_tar and gcs_exists(gcs_tar):
        local_tar = os.path.join(paths["workspace"], TEST_SCENES_TAR)
        try:
            log_writer.log_line(f"Restoring {TEST_SCENES_TAR} from {gcs_tar}...")
            gcs_cp(gcs_tar, local_tar, check=True)
            _safe_extract_tar(local_tar, paths["s3d_raw"])
        except (tarfile.TarError, subprocess.CalledProcessError, OSError, EOFError, ValueError) as exc:
            had_broken_gcs_tar = True
            log_writer.log_line(
                f"[warn] Restoring {TEST_SCENES_TAR} from GCS failed ({exc}); falling back to zip extraction."
            )
        finally:
            if os.path.exists(local_tar):
                os.remove(local_tar)
        if not had_broken_gcs_tar and are_test_scenes_ready(paths["s3d_raw"]):
            return
        if not had_broken_gcs_tar:
            had_broken_gcs_tar = True
            log_writer.log_line(f"[warn] Restored {TEST_SCENES_TAR} was incomplete; falling back to zip extraction.")

    anno_test = os.path.join(paths["s3d_raw"], "scene_03250", "annotation_3d.json")
    if not _eval_annos_ready(paths["s3d_raw"]):
        extract_7z(
            fetch_zip("Structured3D_annotation_3d.zip", paths, cfg, log_writer),
            ["Structured3D/*/annotation_3d.json"],
            anno_test,
            paths["workspace"],
            delete_zip=not args.keep_zips,
            log_writer=log_writer,
        )
    for chunk_idx, scene_range in _EVAL_CHUNK_RANGES:
        if _eval_chunk_views_ready(paths["s3d_raw"], scene_range):
            continue
        zip_name = f"Structured3D_perspective_full_{chunk_idx:02d}.zip"
        extract_7z(
            fetch_zip(zip_name, paths, cfg, log_writer),
            VIEW_PATTERNS,
            os.path.join(paths["s3d_raw"], f"scene_{chunk_idx * 200:05d}"),
            paths["workspace"],
            delete_zip=not args.keep_zips,
            log_writer=log_writer,
        )
    if not args.keep_raw_views:
        for val_scene in range(3200, 3250):
            rdir = os.path.join(paths["s3d_raw"], f"scene_{val_scene:05d}", "2D_rendering")
            if os.path.isdir(rdir):
                shutil.rmtree(rdir)
    if gcs_uri and are_test_scenes_ready(paths["s3d_raw"]):
        archive_eval_test_scenes(paths, gcs_uri, log_writer, overwrite=had_broken_gcs_tar)


def _expected_scene_recon_npz(scene_dir: str) -> list[str]:
    rendering_dir = os.path.join(scene_dir, "2D_rendering")
    if not os.path.isdir(rendering_dir):
        return []
    expected: list[str] = []
    for room_id in sorted(os.listdir(rendering_dir)):
        room_dir = os.path.join(rendering_dir, room_id, "perspective", "full")
        if not os.path.isdir(room_dir):
            continue
        for pos_id in sorted(os.listdir(room_dir)):
            view_dir = os.path.join(room_dir, pos_id)
            if all(
                os.path.exists(os.path.join(view_dir, name))
                for name in ("camera_pose.txt", "depth.png", "rgb_rawlight.png")
            ):
                rel = os.path.normpath(os.path.relpath(view_dir, scene_dir))
                expected.append(rel.replace(os.sep, "__") + ".npz")
    return expected


def _is_recon_shard_complete(
    recon_cache_dir: str,
    s0: int,
    s1: int,
    s3d_raw_dir: str | None = None,
) -> bool:
    if s3d_raw_dir:
        per_scene = [
            (s, _expected_scene_recon_npz(os.path.join(s3d_raw_dir, f"scene_{s:05d}")))
            for s in range(s0, s1)
        ]
        if sum(len(v) for _, v in per_scene) > 0:
            for s, expected_files in per_scene:
                if not expected_files:
                    continue
                sdir = os.path.join(recon_cache_dir, f"scene_{s:05d}")
                if not all(_is_nonempty_file(os.path.join(sdir, fname)) for fname in expected_files):
                    return False
            return True
    for s in range(s0, s1):
        sdir = os.path.join(recon_cache_dir, f"scene_{s:05d}")
        if not os.path.isdir(sdir):
            return False
        has_npz = any(name.endswith(".npz") and _is_nonempty_file(os.path.join(sdir, name)) for name in os.listdir(sdir))
        if not has_npz:
            return False
    return True


def restore_eval_recon_cache(paths: dict[str, str], gcs_uri: str, log_writer: _CleanLogWriter) -> None:
    if not gcs_uri:
        return
    base = gcs_uri.rstrip("/")
    recon_dir = paths["s3d_recon_cache"]
    os.makedirs(recon_dir, exist_ok=True)
    uris = [u for u in gcs_ls(f"{base}/{RECON_CACHE_PREFIX}/recon_*.tar") if u.endswith(".tar")]
    for uri in sorted(uris):
        fname = os.path.basename(uri)
        m = _RECON_SHARD_TAR_RE.match(fname)
        if m and _is_recon_shard_complete(recon_dir, int(m.group(1)), int(m.group(2)), s3d_raw_dir=paths["s3d_raw"]):
            continue
        local_tar = os.path.join(paths["workspace"], fname)
        try:
            log_writer.log_line(f"Restoring eval recon cache {fname} from GCS...")
            gcs_cp(uri, local_tar, check=True)
            _safe_extract_tar(local_tar, recon_dir)
        except (tarfile.TarError, subprocess.CalledProcessError, OSError, EOFError, ValueError) as exc:
            log_writer.log_line(
                f"[warn] Corrupt or unreadable recon cache shard {uri} ({exc}); removing broken remote archive."
            )
            gcs_rm(uri, check=False)
        finally:
            if os.path.exists(local_tar):
                os.remove(local_tar)


def archive_eval_recon_cache(paths: dict[str, str], gcs_uri: str, log_writer: _CleanLogWriter) -> None:
    if not gcs_uri:
        return
    base = gcs_uri.rstrip("/")
    recon_dir = paths["s3d_recon_cache"]
    if not os.path.isdir(recon_dir):
        return
    for s0, s1 in RECON_CACHE_SHARDS:
        if not _is_recon_shard_complete(recon_dir, s0, s1, s3d_raw_dir=paths["s3d_raw"]):
            continue
        fname = f"recon_{s0}_{s1}.tar"
        gcs_tar = f"{base}/{RECON_CACHE_PREFIX}/{fname}"
        if gcs_exists(gcs_tar):
            continue
        local_tar = os.path.join(paths["workspace"], fname)
        try:
            log_writer.log_line(f"Uploading eval recon cache shard {fname} -> {gcs_tar}...")
            with tarfile.open(local_tar, "w") as tf:
                for s in range(s0, s1):
                    sid = f"scene_{s:05d}"
                    sdir = os.path.join(recon_dir, sid)
                    if os.path.isdir(sdir):
                        tf.add(sdir, arcname=sid)
            gcs_cp(local_tar, gcs_tar, check=False)
        finally:
            if os.path.exists(local_tar):
                os.remove(local_tar)


def run_smoke_stage(paths: dict[str, str], log_writer: _CleanLogWriter) -> None:
    log_writer.log_line("=== Stage: smoke ===")
    run_live([sys.executable, "-m", "pytest", "-q"], log_writer)
    demo_out = os.path.join(paths["workspace"], "demo_S221")
    has_playwright = (
        subprocess.run(
            [sys.executable, "-c", "import playwright"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode
        == 0
    )
    if not has_playwright:
        run_live([sys.executable, "-m", "pip", "install", "-q", "playwright"], log_writer)
        run_live([sys.executable, "-m", "playwright", "install", "--with-deps", "chromium"], log_writer)
    run_live(
        [
            sys.executable,
            os.path.join(REPO_DIR, "demo.py"),
            "--images",
            "examples/S221/interior",
            "examples/S221/exterior",
            "--floorplan",
            "examples/S221/floorplan.jpg",
            "--output",
            demo_out,
            "--render",
        ],
        log_writer,
    )


def run_prep_stage(
    args: argparse.Namespace,
    paths: dict[str, str],
    cfg: dict[str, str],
    log_writer: _CleanLogWriter,
) -> None:
    chunks = _resolve_chunks(args)
    prep_profile = compute_prep_profile(
        args.photo_sem_source, args.include_rooms, args.structural_photo_semantics
    )
    target_ranges = _chunks_to_scene_ranges(chunks)
    log_writer.log_line(f"=== Stage: prep (profile={prep_profile}, chunks={chunks}, slice_scenes={args.slice_scenes}) ===")
    if not chunks:
        return
    restore_s3d_tars(
        paths,
        cfg["gcs_backup_uri"],
        log_writer,
        prep_profile=prep_profile,
        target_ranges=target_ranges,
    )
    ensure_floorplan_sem(
        args,
        paths,
        cfg,
        log_writer,
        prep_profile=prep_profile,
        target_ranges=target_ranges,
    )

    for chunk_idx in chunks:
        c_start, c_end = chunk_idx * 200, (chunk_idx + 1) * 200
        slices = [(s, min(s + args.slice_scenes, c_end)) for s in range(c_start, c_end, args.slice_scenes)]
        pending = [sl for sl in slices if not is_slice_covered(paths["s3d_data"], *sl)]
        if not pending:
            log_writer.log_line(f"Chunk {chunk_idx} ({c_start}..{c_end}) already prepared; skipping.")
            continue

        anno_sid = f"scene_{pending[0][0]:05d}"
        anno_expected = os.path.join(paths["s3d_raw"], anno_sid, "annotation_3d.json")
        if not os.path.exists(anno_expected):
            extract_7z(
                fetch_zip("Structured3D_annotation_3d.zip", paths, cfg, log_writer),
                ["Structured3D/*/annotation_3d.json"],
                anno_expected,
                paths["workspace"],
                delete_zip=not args.keep_zips,
                log_writer=log_writer,
            )

        if not all(
            os.path.isdir(os.path.join(paths["s3d_raw"], f"scene_{s0:05d}", "2D_rendering"))
            for s0, _ in pending
        ):
            zip_name = f"Structured3D_perspective_full_{chunk_idx:02d}.zip"
            extract_7z(
                fetch_zip(zip_name, paths, cfg, log_writer),
                VIEW_PATTERNS,
                os.path.join(paths["s3d_raw"], f"scene_{c_start:05d}"),
                paths["workspace"],
                delete_zip=not args.keep_zips,
                log_writer=log_writer,
            )

        for s_start, s_end in pending:
            cmd = [
                sys.executable,
                os.path.join(REPO_DIR, "prepare_s3d.py"),
                "--s3d_root",
                paths["s3d_raw"],
                "--output",
                paths["s3d_data"],
                "--scenes",
                str(s_start),
                str(s_end),
                "--photo_sem_source",
                args.photo_sem_source,
                "--resume",
                *(["--include_rooms"] if args.include_rooms else []),
                *(["--structural_photo_semantics"] if args.structural_photo_semantics else []),
            ]
            run_live(cmd, log_writer)
            if cfg["gcs_backup_uri"]:
                archive_s3d_slice(
                    paths,
                    cfg["gcs_backup_uri"],
                    s_start,
                    s_end,
                    prep_profile=prep_profile,
                    overwrite=True,
                )

            if not args.keep_raw_views:
                for s in range(s_start, s_end):
                    rdir = os.path.join(paths["s3d_raw"], f"scene_{s:05d}", "2D_rendering")
                    if os.path.isdir(rdir):
                        shutil.rmtree(rdir)


def run_train_stage(
    args: argparse.Namespace,
    paths: dict[str, str],
    cfg: dict[str, str],
    log_writer: _CleanLogWriter,
) -> None:
    run_id = os.path.basename(paths["run_out"])
    chunks = _resolve_chunks(args)
    target_ranges = _chunks_to_scene_ranges(chunks)
    prep_profile = compute_prep_profile(
        args.photo_sem_source, args.include_rooms, args.structural_photo_semantics
    )
    log_writer.log_line(
        f"=== Stage: train (run={run_id}, profile={prep_profile}, chunks={chunks}, epochs={args.epochs}) ==="
    )
    restore_s3d_tars(
        paths,
        cfg["gcs_backup_uri"],
        log_writer,
        prep_profile=prep_profile,
        target_ranges=target_ranges,
    )
    missing_after_restore = [
        c for c, (s0, s1) in zip(chunks, target_ranges) if not is_slice_covered(paths["s3d_data"], s0, s1)
    ]
    if missing_after_restore:
        raise RuntimeError(
            f"Prepared slices for chunks {missing_after_restore} are missing locally after GCS restore; run `prep` stage first."
        )
    ensure_floorplan_sem(
        args,
        paths,
        cfg,
        log_writer,
        prep_profile=prep_profile,
        target_ranges=target_ranges,
    )

    gcs_uri = cfg["gcs_backup_uri"].rstrip("/")
    sync_uri = f"{gcs_uri}/runs/{run_id}" if gcs_uri else ""
    run_out = paths["run_out"]
    if gcs_uri and not os.path.exists(os.path.join(run_out, "checkpoint.pt")):
        restore_run_from_gcs(paths, gcs_uri, allow_log_overwrite=False)

    ckpt_path = os.path.join(run_out, "checkpoint.pt")
    if not _inter_eval_enabled(args):
        train_cmd = _build_train_cmd(args, paths, target_ranges, run_out, sync_uri)
        run_live(train_cmd, log_writer)
        if not os.path.exists(ckpt_path):
            raise RuntimeError(f"Expected {ckpt_path} after training")
        if sync_uri:
            gcs_rsync(run_out, sync_uri, check=False)
        return

    eval_every = int(getattr(args, "eval_every_epochs", 1))
    inter_scenes = int(getattr(args, "inter_eval_scenes", 50))
    milestones = list(range(eval_every, args.epochs, eval_every))
    if not milestones or milestones[-1] != args.epochs:
        milestones.append(args.epochs)
    milestone_set = set(milestones)

    init_state = _read_train_state(run_out)
    if init_state is not None:
        init_completed = min(init_state["completed_epochs"], args.epochs)
        if init_completed >= 1 and init_completed in milestone_set:
            _run_inter_epoch_eval(
                args,
                paths,
                cfg,
                log_writer,
                epoch=init_completed,
                step=init_state["step"],
                inter_scenes=inter_scenes,
            )

    for target_epoch in milestones:
        cur_state = _read_train_state(run_out)
        if cur_state is not None and cur_state["completed_epochs"] >= target_epoch:
            continue
        log_writer.log_line(f"--- Training chunk through epoch {target_epoch}/{args.epochs} ---")
        train_cmd = _build_train_cmd(
            args,
            paths,
            target_ranges,
            run_out,
            sync_uri,
            stop_after_epoch=target_epoch,
        )
        run_live(train_cmd, log_writer)
        if not os.path.exists(ckpt_path):
            raise RuntimeError(f"Expected {ckpt_path} after training")
        after_state = _read_train_state(run_out)
        eval_epoch = (
            min(after_state["completed_epochs"], args.epochs)
            if after_state is not None and after_state["completed_epochs"] >= 1
            else target_epoch
        )
        eval_step = (
            after_state["step"]
            if after_state is not None
            else _read_last_training_log(os.path.join(run_out, "log.csv"))[0]
        )
        if eval_epoch in milestone_set:
            _run_inter_epoch_eval(
                args,
                paths,
                cfg,
                log_writer,
                epoch=eval_epoch,
                step=eval_step,
                inter_scenes=inter_scenes,
            )

    if not os.path.exists(ckpt_path):
        raise RuntimeError(f"Expected {ckpt_path} after training")
    if sync_uri:
        gcs_rsync(run_out, sync_uri, check=False)


def _build_train_cmd(
    args: argparse.Namespace,
    paths: dict[str, str],
    target_ranges: list[tuple[int, int]],
    run_out: str,
    sync_uri: str,
    *,
    stop_after_epoch: int | None = None,
) -> list[str]:
    scene_range_specs = [f"{s0}:{s1}" for s0, s1 in target_ranges]
    return [
        sys.executable,
        os.path.join(REPO_DIR, "train.py"),
        "--data",
        paths["s3d_data"],
        "--output",
        run_out,
        *(["--scene_ranges", *scene_range_specs] if scene_range_specs else []),
        "--init_lora",
        args.init_lora,
        *(["--freeze_lora"] if args.freeze_lora else []),
        "--sem_mode",
        args.sem_mode,
        "--lambda_sem",
        str(args.lambda_sem),
        *(["--alpha_sem", str(args.alpha_sem)] if args.alpha_sem is not None else []),
        "--epochs",
        str(args.epochs),
        *(["--stop_after_epoch", str(stop_after_epoch)] if stop_after_epoch is not None else []),
        "--resolutions",
        *(str(r) for r in args.resolutions),
        "--min_points",
        str(args.min_points),
        "--save_every",
        str(args.save_every),
        "--resume",
        *(["--sync_gcs_uri", sync_uri] if sync_uri else []),
    ]


def _is_valid_eval_metrics_file(path: str) -> dict[str, float] | None:
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    out: dict[str, float] = {}
    for k in METRICS_ORDER:
        v = data.get(k)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
            return None
        out[k] = float(v)
    return out


def _read_train_state(run_out: str) -> dict[str, int] | None:
    path = os.path.join(run_out, "train_state.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    out: dict[str, int] = {}
    for k in ("step", "steps_per_epoch", "completed_epochs", "total_epochs"):
        v = data.get(k)
        if isinstance(v, bool) or not isinstance(v, int):
            return None
        out[k] = v
    return out


def _read_epoch_evals(run_out: str) -> list[dict[str, Any]]:
    path = os.path.join(run_out, "epoch_evals.jsonl")
    if not os.path.exists(path):
        return []
    by_epoch: dict[int, dict[str, Any]] = {}
    try:
        with open(path, encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(rec, dict) and isinstance(rec.get("epoch"), int) and not isinstance(rec.get("epoch"), bool):
                    by_epoch[int(rec["epoch"])] = rec
    except OSError:
        return []
    return [by_epoch[ep] for ep in sorted(by_epoch)]


def _upsert_epoch_eval_jsonl(jsonl_path: str, record: dict[str, Any]) -> None:
    by_epoch: dict[int, dict[str, Any]] = {
        int(rec["epoch"]): rec
        for rec in _read_epoch_evals(os.path.dirname(jsonl_path))
    }
    by_epoch[int(record["epoch"])] = record
    content = "".join(json.dumps(by_epoch[ep]) + "\n" for ep in sorted(by_epoch))
    _atomic_write_text(jsonl_path, content)


def _snapshot_epoch_checkpoint(run_out: str, epoch_dir: str) -> None:
    os.makedirs(epoch_dir, exist_ok=True)
    for fname in (
        "adapter_config.json",
        "adapter_model.safetensors",
        "adapter_model.bin",
        "semantic_adapter.pt",
        "train_state.json",
    ):
        src = os.path.join(run_out, fname)
        if not os.path.exists(src):
            continue
        dst = os.path.join(epoch_dir, fname)
        tmp = f"{dst}.tmp"
        shutil.copy2(src, tmp)
        os.replace(tmp, dst)


def _inter_eval_meta(args: argparse.Namespace, epoch: int, inter_scenes: int) -> dict[str, Any]:
    return {
        "epoch": int(epoch),
        "max_scenes": int(inter_scenes),
        "init_lora": str(args.init_lora),
        "photo_sem_source": str(args.photo_sem_source),
        "include_rooms": bool(args.include_rooms),
        "structural_photo_semantics": bool(args.structural_photo_semantics),
    }


def _meta_matches_eval_config(meta: dict[str, Any], expected: dict[str, Any]) -> bool:
    keys = ("max_scenes", "init_lora", "photo_sem_source", "include_rooms", "structural_photo_semantics")
    return all(meta.get(k) == expected.get(k) for k in keys)


def _find_cached_none_eval_json(epochs_root: str, target_epoch: int, expected_meta: dict[str, Any]) -> str | None:
    if not os.path.isdir(epochs_root):
        return None
    target_name = f"epoch_{target_epoch:02d}"
    for name in sorted(os.listdir(epochs_root)):
        if name == target_name or not name.startswith("epoch_"):
            continue
        ep_dir = os.path.join(epochs_root, name)
        if not os.path.isdir(ep_dir):
            continue
        meta_path = os.path.join(ep_dir, "inter_eval_meta.json")
        if not os.path.exists(meta_path):
            continue
        try:
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
        except (OSError, json.JSONDecodeError, ValueError):
            continue
        if not isinstance(meta, dict) or not _meta_matches_eval_config(meta, expected_meta):
            continue
        none_path = os.path.join(ep_dir, "eval_s3d_none.json")
        if _is_valid_eval_metrics_file(none_path) is not None:
            return none_path
    return None


def _prepare_eval_recon_cache(
    args: argparse.Namespace,
    paths: dict[str, str],
    cfg: dict[str, str],
    log_writer: _CleanLogWriter,
) -> tuple[bool, str]:
    gcs_uri = cfg["gcs_backup_uri"].rstrip("/")
    use_recon_cache = bool(getattr(args, "use_recon_cache", True))
    recon_cache_dir = paths["s3d_recon_cache"]
    if use_recon_cache:
        cache_empty = not os.path.isdir(recon_cache_dir) or not os.listdir(recon_cache_dir)
        if cache_empty and shutil.disk_usage(paths["workspace"]).free < MIN_RECON_CACHE_FREE_BYTES:
            log_writer.log_line("[warn] Free disk below 22 GiB; disabling local --recon_cache for this run.")
            use_recon_cache = False
        elif gcs_uri:
            restore_eval_recon_cache(paths, gcs_uri, log_writer)
    return use_recon_cache, recon_cache_dir


def _build_eval_cmd(
    args: argparse.Namespace,
    paths: dict[str, str],
    *,
    ckpt_dir: str,
    output_dir: str,
    eval_modes: Sequence[str],
    use_recon_cache: bool,
    recon_cache_dir: str,
    sync_uri: str,
    max_scenes: int | None = None,
) -> list[str]:
    return [
        sys.executable,
        os.path.join(REPO_DIR, "evaluate_s3d.py"),
        "--s3d_root",
        paths["s3d_raw"],
        "--checkpoint_dir",
        ckpt_dir,
        "--sem_modes",
        *eval_modes,
        "--photo_sem_source",
        args.photo_sem_source,
        "--output_dir",
        output_dir,
        "--resume",
        *(["--max_scenes", str(max_scenes)] if max_scenes is not None else []),
        *(["--recon_cache", recon_cache_dir] if use_recon_cache else []),
        *(["--include_rooms"] if args.include_rooms else []),
        *(["--structural_photo_semantics"] if args.structural_photo_semantics else []),
        *(["--sync_gcs_uri", sync_uri] if sync_uri else []),
    ]


def _run_inter_epoch_eval(
    args: argparse.Namespace,
    paths: dict[str, str],
    cfg: dict[str, str],
    log_writer: _CleanLogWriter,
    *,
    epoch: int,
    step: int | None,
    inter_scenes: int,
) -> None:
    run_out = paths["run_out"]
    run_id = os.path.basename(run_out)
    epochs_root = os.path.join(run_out, "epochs")
    epoch_dir = os.path.join(epochs_root, f"epoch_{epoch:02d}")
    eval_modes = _resolve_eval_modes(args, run_out)
    expected_meta = _inter_eval_meta(args, epoch, inter_scenes)
    meta_path = os.path.join(epoch_dir, "inter_eval_meta.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, encoding="utf-8") as f:
                old_meta = json.load(f)
        except (OSError, json.JSONDecodeError, ValueError):
            old_meta = None
        if not isinstance(old_meta, dict) or not _meta_matches_eval_config(old_meta, expected_meta):
            for fname in os.listdir(epoch_dir):
                if fname.startswith(("eval_s3d_", "eval_progress_")) or fname == "inter_eval_meta.json":
                    os.remove(os.path.join(epoch_dir, fname))

    existing_records = {int(r["epoch"]): r for r in _read_epoch_evals(run_out) if "epoch" in r}
    existing_rec = existing_records.get(epoch)
    need_eval = any(
        _is_valid_eval_metrics_file(os.path.join(epoch_dir, f"eval_s3d_{m}.json")) is None
        for m in eval_modes
    )
    if (
        not need_eval
        and existing_rec is not None
        and existing_rec.get("max_scenes") == int(inter_scenes)
        and isinstance(existing_rec.get("metrics"), dict)
        and all(m in existing_rec["metrics"] for m in eval_modes)
    ):
        return

    _snapshot_epoch_checkpoint(run_out, epoch_dir)
    if args.freeze_lora and "none" in eval_modes:
        none_dst = os.path.join(epoch_dir, "eval_s3d_none.json")
        if _is_valid_eval_metrics_file(none_dst) is None:
            cached_none = _find_cached_none_eval_json(epochs_root, epoch, expected_meta)
            if cached_none is not None:
                log_writer.log_line(
                    f"Reusing frozen-LoRA baseline `none` metrics for epoch {epoch} from {cached_none}"
                )
                with open(cached_none, encoding="utf-8") as f:
                    _atomic_write_text(none_dst, f.read())

    gcs_uri = cfg["gcs_backup_uri"].rstrip("/")
    need_eval = any(
        _is_valid_eval_metrics_file(os.path.join(epoch_dir, f"eval_s3d_{m}.json")) is None
        for m in eval_modes
    )
    if need_eval:
        log_writer.log_line(
            f"=== Inter-eval: epoch {epoch}/{args.epochs} (scenes={inter_scenes}, modes={eval_modes}) ==="
        )
        ensure_eval_test_scenes(args, paths, cfg, log_writer)
        use_recon_cache, recon_cache_dir = _prepare_eval_recon_cache(args, paths, cfg, log_writer)
        epoch_sync_uri = f"{gcs_uri}/runs/{run_id}/epochs/epoch_{epoch:02d}" if gcs_uri else ""
        eval_cmd = _build_eval_cmd(
            args,
            paths,
            ckpt_dir=epoch_dir,
            output_dir=epoch_dir,
            eval_modes=eval_modes,
            use_recon_cache=use_recon_cache,
            recon_cache_dir=recon_cache_dir,
            sync_uri=epoch_sync_uri,
            max_scenes=inter_scenes,
        )
        try:
            run_live(eval_cmd, log_writer)
        finally:
            if use_recon_cache and gcs_uri:
                archive_eval_recon_cache(paths, gcs_uri, log_writer)

    epoch_metrics: dict[str, dict[str, float]] = {}
    for m in eval_modes:
        parsed = _is_valid_eval_metrics_file(os.path.join(epoch_dir, f"eval_s3d_{m}.json"))
        if parsed is None:
            raise RuntimeError(f"Expected valid inter-eval metrics for mode {m!r} in {epoch_dir}")
        epoch_metrics[m] = parsed

    _atomic_write_text(meta_path, json.dumps(expected_meta, indent=2) + "\n")
    record = {
        "epoch": int(epoch),
        "total_epochs": int(args.epochs),
        "step": int(step) if step is not None else None,
        "max_scenes": int(inter_scenes),
        "metrics": epoch_metrics,
    }
    _upsert_epoch_eval_jsonl(os.path.join(run_out, "epoch_evals.jsonl"), record)
    log_writer.log_line(f"=== Epoch {epoch}/{args.epochs} inter-eval ({inter_scenes} scenes) ===")
    for tbl_line in format_results_table(epoch_metrics).splitlines():
        log_writer.log_line(tbl_line)
    if gcs_uri:
        gcs_rsync(run_out, f"{gcs_uri}/runs/{run_id}", check=False)


def _resolve_eval_modes(args: argparse.Namespace, ckpt_dir: str) -> list[str]:
    if args.eval_modes:
        return list(dict.fromkeys(args.eval_modes))
    if args.photo_sem_source == "none":
        return ["none"]
    has_adapter = os.path.exists(os.path.join(ckpt_dir, "semantic_adapter.pt"))
    if args.sem_mode == "none" and not has_adapter:
        return ["none"]
    return ["none", "uniform", "consensus"]


def run_eval_stage(
    args: argparse.Namespace,
    paths: dict[str, str],
    cfg: dict[str, str],
    log_writer: _CleanLogWriter,
) -> list[str]:
    run_out = paths["run_out"]
    run_id = os.path.basename(run_out)
    ckpt_dir = args.eval_checkpoint_dir or run_out
    gcs_uri = cfg["gcs_backup_uri"].rstrip("/")
    sync_uri = f"{gcs_uri}/runs/{run_id}" if gcs_uri else ""

    if gcs_uri and ckpt_dir == run_out:
        restore_run_from_gcs(paths, gcs_uri, allow_log_overwrite=False)

    eval_modes = _resolve_eval_modes(args, ckpt_dir)
    log_writer.log_line(
        f"=== Stage: eval (checkpoint={ckpt_dir}, modes={eval_modes}, photo_sem={args.photo_sem_source}) ==="
    )

    ensure_eval_test_scenes(args, paths, cfg, log_writer)

    use_recon_cache, recon_cache_dir = _prepare_eval_recon_cache(args, paths, cfg, log_writer)
    eval_cmd = _build_eval_cmd(
        args,
        paths,
        ckpt_dir=ckpt_dir,
        output_dir=run_out,
        eval_modes=eval_modes,
        use_recon_cache=use_recon_cache,
        recon_cache_dir=recon_cache_dir,
        sync_uri=sync_uri,
    )
    try:
        run_live(eval_cmd, log_writer)
    finally:
        if use_recon_cache and gcs_uri:
            archive_eval_recon_cache(paths, gcs_uri, log_writer)
    return eval_modes


def run_backup_stage(
    args: argparse.Namespace,
    paths: dict[str, str],
    cfg: dict[str, str],
    log_writer: _CleanLogWriter,
) -> None:
    log_writer.log_line("=== Stage: backup ===")
    gcs_uri = cfg["gcs_backup_uri"].rstrip("/")
    if not gcs_uri:
        raise RuntimeError("GCS_BACKUP_URI is required for the `backup` stage.")
    prep_profile = compute_prep_profile(
        args.photo_sem_source, args.include_rooms, args.structural_photo_semantics
    )
    ensure_floorplan_sem(args, paths, cfg, log_writer, prep_profile=prep_profile)
    for s_start, s_end in covered_scene_ranges(paths["s3d_data"]):
        archive_s3d_slice(paths, gcs_uri, s_start, s_end, prep_profile=prep_profile, overwrite=True)
    if are_test_scenes_ready(paths["s3d_raw"]):
        archive_eval_test_scenes(paths, gcs_uri, log_writer)
    if bool(getattr(args, "use_recon_cache", True)):
        archive_eval_recon_cache(paths, gcs_uri, log_writer)
    run_id = os.path.basename(paths["run_out"])
    gcs_rsync(paths["run_out"], f"{gcs_uri}/runs/{run_id}", check=True)


def _read_last_training_log(log_csv: str) -> tuple[int | None, float | None]:
    if not os.path.exists(log_csv):
        return None, None
    last_row: dict[str, str] | None = None
    with open(log_csv, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            last_row = row
    if not last_row:
        return None, None
    try:
        step = int(last_row["step"]) if last_row.get("step") else None
    except ValueError:
        step = None
    try:
        loss = float(last_row["loss"]) if last_row.get("loss") else None
    except ValueError:
        loss = None
    return step, loss


def _git_short_head() -> str | None:
    if not shutil.which("git"):
        return None
    try:
        res = subprocess.run(
            ["git", "-C", REPO_DIR, "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    return res.stdout.strip() if res.returncode == 0 else None


def format_results_table(eval_metrics: dict[str, dict[str, float]]) -> str:
    if not eval_metrics:
        return "(no evaluation metrics recorded yet)"
    header = f"{'Mode':<12} | " + " | ".join(f"{k:>15}" for k in METRICS_ORDER)
    sep = "-" * 12 + "-+-" + "-+-".join("-" * 15 for _ in METRICS_ORDER)
    rows = [header, sep]
    for mode, metrics in eval_metrics.items():
        vals = " | ".join(
            f"{metrics[k]:15.2f}" if isinstance(metrics.get(k), (int, float)) else f"{'-':>15}"
            for k in METRICS_ORDER
        )
        rows.append(f"{mode:<12} | {vals}")
    return "\n".join(rows)


def _upsert_index_jsonl(
    paths: dict[str, str],
    record: dict[str, Any],
    gcs_uri: str,
) -> None:
    by_id: dict[str, dict[str, Any]] = {}
    order: list[str] = []

    def _ingest_text(raw: str | None) -> None:
        if not raw:
            return
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            rid = item.get("run_id")
            if not rid:
                continue
            if rid not in by_id:
                order.append(rid)
            by_id[rid] = item

    if gcs_uri and gcs_exists(f"{gcs_uri}/runs/INDEX.jsonl"):
        _ingest_text(_gcs_read_text(f"{gcs_uri}/runs/INDEX.jsonl"))
    if os.path.exists(paths["index_jsonl"]):
        with open(paths["index_jsonl"], encoding="utf-8") as f:
            _ingest_text(f.read())

    rid = str(record["run_id"])
    if rid not in by_id:
        order.append(rid)
    by_id[rid] = record
    content = "\n".join(json.dumps(by_id[k]) for k in order) + "\n"
    _atomic_write_text(paths["index_jsonl"], content)
    if gcs_uri:
        gcs_cp(paths["index_jsonl"], f"{gcs_uri}/runs/INDEX.jsonl", check=False)


def write_final_results(
    args: argparse.Namespace,
    paths: dict[str, str],
    cfg: dict[str, str],
    eval_modes: list[str] | None,
    log_writer: _CleanLogWriter,
    *,
    resumed: bool = False,
) -> dict[str, Any]:
    run_out = paths["run_out"]
    run_id = os.path.basename(run_out)
    prep_profile = compute_prep_profile(
        args.photo_sem_source, args.include_rooms, args.structural_photo_semantics
    )
    modes_to_check: list[str] = []
    for mode in [*(eval_modes or []), *SEM_MODE_CHOICES]:
        if mode not in modes_to_check:
            modes_to_check.append(mode)
    eval_metrics: dict[str, dict[str, float]] = {}
    eval_files: dict[str, str] = {}
    for mode in modes_to_check:
        out_json = os.path.join(run_out, f"eval_s3d_{mode}.json")
        if os.path.exists(out_json):
            try:
                with open(out_json, encoding="utf-8") as f:
                    parsed = json.load(f)
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                log_writer.log_line(f"[warn] Ignoring unreadable eval file {out_json}: {exc}")
                continue
            if not isinstance(parsed, dict):
                log_writer.log_line(f"[warn] Ignoring non-dict eval file {out_json}.")
                continue
            eval_metrics[mode] = parsed
            eval_files[mode] = out_json

    log_csv = os.path.join(run_out, "log.csv")
    last_step, last_loss = _read_last_training_log(log_csv)
    epoch_evals = _read_epoch_evals(run_out)
    ranges = covered_scene_ranges(paths["s3d_data"])
    gcs_uri = cfg["gcs_backup_uri"].rstrip("/")
    finished_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    git_commit = _git_short_head()

    existing_manifest = _read_manifest_dict(run_id, paths["workspace"], gcs_uri)
    started_at = (
        str(existing_manifest["started_at_utc"])
        if isinstance(existing_manifest, dict) and existing_manifest.get("started_at_utc")
        else finished_at
    )
    eff_resumed = resumed or bool(isinstance(existing_manifest, dict) and existing_manifest.get("resumed"))

    summary: dict[str, Any] = {
        "run_id": run_id,
        "run_name": run_id,
        "resumed": eff_resumed,
        "photo_sem_source": str(args.photo_sem_source),
        "floorplan_sem_source": "s3d_annos",
        "prep_profile": prep_profile,
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "git_commit": git_commit,
        "stages": list(args.stages),
        "workspace": paths["workspace"],
        "run_dir": run_out,
        "prepared_scene_ranges": [list(r) for r in ranges],
        "config": {
            "chunks": _resolve_chunks(args),
            "slice_scenes": args.slice_scenes,
            "epochs": args.epochs,
            "resolutions": list(args.resolutions),
            "min_points": args.min_points,
            "save_every": args.save_every,
            "init_lora": args.init_lora,
            "freeze_lora": args.freeze_lora,
            "sem_mode": args.sem_mode,
            "lambda_sem": args.lambda_sem,
            "alpha_sem": args.alpha_sem,
            "photo_sem_source": args.photo_sem_source,
            "include_rooms": args.include_rooms,
            "structural_photo_semantics": args.structural_photo_semantics,
            "use_recon_cache": bool(getattr(args, "use_recon_cache", True)),
            "eval_every_epochs": int(getattr(args, "eval_every_epochs", 1)),
            "inter_eval_scenes": int(getattr(args, "inter_eval_scenes", 50)),
            "gcs_backup_uri": f"{gcs_uri}/runs/{run_id}" if gcs_uri else None,
        },
        "training": {
            "checkpoint_exists": os.path.exists(os.path.join(run_out, "checkpoint.pt")),
            "adapter_exists": os.path.exists(os.path.join(run_out, "adapter_config.json")),
            "semantic_adapter_exists": os.path.exists(os.path.join(run_out, "semantic_adapter.pt")),
            "last_logged_step": last_step,
            "last_logged_loss": last_loss,
        },
        "eval_metrics": eval_metrics,
        "epoch_evals": epoch_evals,
        "artifacts": {
            "summary_json": paths["summary_json"],
            "summary_txt": paths["summary_txt"],
            "latest_json": paths["latest_json"],
            "latest_txt": paths["latest_txt"],
            "manifest_json": paths["manifest_json"],
            "pipeline_log": paths["log_path"],
            "eval_files": eval_files,
        },
    }

    slices_str = ", ".join(f"{s}..{e}" for s, e in ranges) if ranges else "none"
    ckpt_status = "present" if summary["training"]["checkpoint_exists"] else "missing"
    if last_step is not None:
        loss_str = f"{last_loss:.4f}" if last_loss is not None else "n/a"
        ckpt_status += f" (last logged step={last_step}, loss={loss_str})"

    lines = [
        "=" * 72,
        "SceneAligner Structured3D Run Summary",
        "=" * 72,
        f"Run ID           : {run_id}",
        f"Prep Profile     : {prep_profile} (photo={args.photo_sem_source}, floorplan=s3d_annos)",
        f"Started / End UTC: {started_at} -> {finished_at}",
        f"Git Commit       : {git_commit or 'unknown'}",
        f"Stages           : {', '.join(args.stages)}",
        f"Workspace        : {paths['workspace']}",
        f"Run Directory    : {run_out}",
        f"Prepared Slices  : {slices_str}",
        f"Checkpoint       : {ckpt_status}",
        f"GCS Backup URI   : {summary['config']['gcs_backup_uri'] or 'disabled'}",
        "",
        "Structured3D Single-View Localization (Table 3, Recall %):",
        format_results_table(eval_metrics),
    ]
    if epoch_evals:
        lines.extend(["", f"Between-Epoch Subset Evals ({epoch_evals[0].get('max_scenes', '?')} scenes):"])
        for rec in epoch_evals:
            ep = rec.get("epoch")
            tot = rec.get("total_epochs", args.epochs)
            step_val = rec.get("step")
            step_part = f", step {step_val}" if step_val is not None else ""
            rec_metrics = rec.get("metrics") or {}
            lines.append(f"  Epoch {ep}/{tot}{step_part}:")
            for tbl_line in format_results_table(rec_metrics).splitlines():
                lines.append(f"    {tbl_line}")
    lines.extend(
        [
            "",
            "Saved Files:",
            f"  Summary JSON   : {paths['summary_json']}",
            f"  Summary Text   : {paths['summary_txt']}",
            f"  Latest JSON    : {paths['latest_json']}",
            f"  Latest Text    : {paths['latest_txt']}",
            f"  Pipeline Log   : {paths['log_path']}",
            "=" * 72,
        ]
    )
    summary_txt = "\n".join(lines) + "\n"
    summary_json_txt = json.dumps(summary, indent=2) + "\n"

    _atomic_write_text(paths["summary_json"], summary_json_txt)
    _atomic_write_text(paths["summary_txt"], summary_txt)
    _atomic_write_text(paths["latest_json"], summary_json_txt)
    _atomic_write_text(paths["latest_txt"], summary_txt)

    manifest = {
        "run_id": run_id,
        "status": "completed",
        "resumed": eff_resumed,
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "git_commit": git_commit,
        "photo_sem_source": str(args.photo_sem_source),
        "floorplan_sem_source": "s3d_annos",
        "prep_profile": prep_profile,
        "stages": list(args.stages),
        "config_fingerprint": compute_run_fingerprint(args),
        "eval_metrics": eval_metrics,
    }
    _atomic_write_text(paths["manifest_json"], json.dumps(manifest, indent=2) + "\n")

    track_active = "train" in args.stages or "eval" in args.stages
    should_sync_gcs = bool(gcs_uri and any(s in args.stages for s in ("train", "eval", "backup")))

    if track_active:
        active_file = paths["active_run_file"]
        if os.path.exists(active_file):
            with open(active_file, encoding="utf-8") as f:
                if f.read().strip() == run_id:
                    os.remove(active_file)
        if should_sync_gcs:
            remote_active = _gcs_read_text(f"{gcs_uri}/runs/ACTIVE_RUN")
            if remote_active and remote_active.strip() == run_id:
                gcs_rm(f"{gcs_uri}/runs/ACTIVE_RUN", check=False)

        index_rec = {
            "run_id": run_id,
            "started_at_utc": started_at,
            "finished_at_utc": finished_at,
            "git_commit": git_commit,
            "stages": list(args.stages),
            "photo_sem_source": str(args.photo_sem_source),
            "floorplan_sem_source": "s3d_annos",
            "prep_profile": prep_profile,
            "chunks": _resolve_chunks(args),
            "epochs": int(args.epochs),
            "sem_mode": str(args.sem_mode),
            "lambda_sem": float(args.lambda_sem),
            "freeze_lora": bool(args.freeze_lora),
            "eval_metrics": eval_metrics,
        }
        _upsert_index_jsonl(paths, index_rec, gcs_uri if should_sync_gcs else "")

    log_writer.flush_partial()
    log_writer._fp.write(summary_txt)
    log_writer._fp.flush()
    if not _same_open_file(sys.stdout, paths["log_path"]):
        sys.stdout.write(summary_txt)
        sys.stdout.flush()

    if should_sync_gcs:
        ok = gcs_rsync(run_out, f"{gcs_uri}/runs/{run_id}", check=False)
        gcs_cp(paths["summary_txt"], f"{gcs_uri}/runs/LATEST_SUMMARY.txt", check=False)
        if not ok:
            log_writer.log_line("[warn] Final GCS rsync returned non-zero; local results are saved.")
    return summary


def log_system_banner(paths: dict[str, str], log_writer: _CleanLogWriter, *, runner_name: str = "run_vm.py") -> None:
    log_writer.log_line(f"Starting {runner_name} in {REPO_DIR} (workspace={paths['workspace']})")
    if shutil.which("nvidia-smi"):
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
        )
        if smi.returncode == 0 and smi.stdout.strip():
            log_writer.log_line(f"GPU: {smi.stdout.strip().splitlines()[0]}")
    if shutil.which("df"):
        df = subprocess.run(["df", "-h", paths["workspace"]], capture_output=True, text=True, check=False)
        if df.returncode == 0 and len(df.stdout.splitlines()) >= 2:
            log_writer.log_line(f"Disk: {df.stdout.splitlines()[1]}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Persistent-VM runner for SceneAligner (Structured3D).")
    parser.add_argument("--stages", nargs="+", choices=STAGE_CHOICES, default=["prep", "train", "eval"])
    parser.add_argument("--workspace", default=None, help="persistent workspace dir (default: <repo>/outputs)")
    parser.add_argument(
        "--run-id",
        "--run-name",
        dest="run_id",
        default=None,
        help="explicit run ID (default: timestamped ID, auto-resuming matching incomplete runs)",
    )
    parser.add_argument("--new-run", action="store_true", help="ignore ACTIVE_RUN and mint a fresh timestamped run ID")
    parser.add_argument("--tag", default=None, help="optional short tag appended to auto-generated run ID slug")
    parser.add_argument(
        "--gcs-uri",
        default=None,
        help=f"GCS bucket or gs://... URI for slice/run sync (default: {DEFAULT_GCS_BACKUP_URI})",
    )
    parser.add_argument("--no-gcs", action="store_true", help="disable GCS sync and skip GCS prompt")
    parser.add_argument("--s3d-base-url", default=None, help="Structured3D archive base URL")
    parser.add_argument("--chunks", type=int, nargs="+", default=[0, 1, 2], help="Structured3D chunks (0..14, skips 8)")
    parser.add_argument("--all-train-chunks", action="store_true", help="prepare/train all Structured3D train chunks (0..14 except 8)")
    parser.add_argument("--slice-scenes", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--resolutions", type=int, nargs="+", default=[512])
    parser.add_argument("--min-points", type=int, default=1000)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--init-lora", default="jhcho99/SceneAligner/structured3d")
    parser.add_argument("--freeze-lora", dest="freeze_lora", action="store_true", default=True)
    parser.add_argument("--no-freeze-lora", dest="freeze_lora", action="store_false")
    parser.add_argument("--sem-mode", choices=SEM_MODE_CHOICES, default="consensus")
    parser.add_argument("--lambda-sem", type=float, default=0.5)
    parser.add_argument("--alpha-sem", type=float, default=None)
    parser.add_argument("--photo-sem-source", choices=PHOTO_SEM_SOURCE_CHOICES, default="gt")
    parser.add_argument("--eval-modes", nargs="+", choices=SEM_MODE_CHOICES, default=None)
    parser.add_argument("--eval-checkpoint-dir", default=None, help="checkpoint dir or HF repo for eval")
    parser.add_argument(
        "--eval-every-epochs",
        type=int,
        default=1,
        help="run intermediate subset evaluation every N epochs during train (0 disables)",
    )
    parser.add_argument(
        "--inter-eval-scenes",
        type=int,
        default=50,
        help="number of test scenes to evaluate at each intermediate epoch checkpoint (0 disables)",
    )
    parser.add_argument("--recon-cache", dest="use_recon_cache", action="store_true", default=True)
    parser.add_argument("--no-recon-cache", dest="use_recon_cache", action="store_false")
    parser.add_argument("--include-rooms", action="store_true")
    parser.add_argument("--structural-photo-semantics", action="store_true")
    parser.add_argument("--keep-zips", action="store_true", help="keep Structured3D .zip archives after extraction")
    parser.add_argument("--keep-raw-views", action="store_true", help="keep extracted 2D_rendering views after prep")
    parser.add_argument("--skip-setup", action="store_true", help="skip setup_vm.sh / .venv bootstrap")
    parser.add_argument("--detach", action="store_true", help="run in detached background session and tail pipeline.log")
    return parser.parse_args(argv)


def _req_pkg_name(req_line: str) -> str:
    stripped = req_line.strip()
    if " @ " in stripped:
        stripped = stripped.split(" @ ", 1)[0]
    for sep in ("==", ">=", "<=", "!=", "~=", ">", "<", "["):
        if sep in stripped:
            stripped = stripped.split(sep, 1)[0]
    return stripped.strip().lower().replace("_", "-")


def install_colab_deps(*, include_pytest: bool = True) -> list[str]:
    """Install requirements.txt pins for Colab while skipping base scientific packages preloaded in the kernel."""
    req_path = os.path.join(REPO_DIR, "requirements.txt")
    to_install: list[str] = []
    with open(req_path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if _req_pkg_name(line) in _COLAB_BASE_PACKAGES:
                continue
            to_install.append(line)
    if include_pytest and "pytest" not in {_req_pkg_name(x) for x in to_install}:
        to_install.append("pytest")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *to_install], cwd=REPO_DIR, check=True)
    return to_install


class PipelineSession:
    """Interactive notebook/session wrapper sharing the exact VM pipeline logic."""

    def __init__(
        self,
        args: argparse.Namespace,
        paths: dict[str, str],
        cfg: dict[str, str],
        *,
        resumed: bool,
    ) -> None:
        self.args = args
        self.paths = paths
        self.cfg = cfg
        self.resumed = resumed
        self.prep_profile = compute_prep_profile(
            args.photo_sem_source, args.include_rooms, args.structural_photo_semantics
        )
        self.run_id = os.path.basename(paths["run_out"])
        self.eval_modes: list[str] | None = list(args.eval_modes) if args.eval_modes else None

    def _open_log(self) -> _CleanLogWriter:
        return _CleanLogWriter(
            self.paths["log_path"],
            redact=self.cfg["s3d_base_url"],
            echo_to_stdout=not _is_interactive_stdout(),
        )

    def run_smoke(self) -> None:
        lw = self._open_log()
        try:
            run_smoke_stage(self.paths, lw)
        finally:
            lw.close()

    def run_prep(self) -> None:
        lw = self._open_log()
        try:
            run_prep_stage(self.args, self.paths, self.cfg, lw)
        finally:
            lw.close()

    def run_train(self) -> None:
        lw = self._open_log()
        try:
            run_train_stage(self.args, self.paths, self.cfg, lw)
        finally:
            lw.close()

    def run_eval(self) -> list[str]:
        lw = self._open_log()
        try:
            self.eval_modes = run_eval_stage(self.args, self.paths, self.cfg, lw)
            return self.eval_modes
        finally:
            lw.close()

    def run_backup(self) -> None:
        lw = self._open_log()
        try:
            run_backup_stage(self.args, self.paths, self.cfg, lw)
        finally:
            lw.close()

    def write_final_results(self) -> dict[str, Any]:
        lw = self._open_log()
        try:
            return write_final_results(
                self.args, self.paths, self.cfg, self.eval_modes, lw, resumed=self.resumed
            )
        finally:
            lw.close()


def init_session(
    *,
    workspace: str = "/content/scenealigner_ws",
    chunks: Sequence[int] = (0, 1, 2),
    all_train_chunks: bool = False,
    slice_scenes: int = 50,
    epochs: int = 10,
    resolutions: Sequence[int] = (512,),
    min_points: int = 1000,
    save_every: int = 500,
    init_lora: str = "jhcho99/SceneAligner/structured3d",
    freeze_lora: bool = True,
    sem_mode: str = "consensus",
    lambda_sem: float = 0.5,
    alpha_sem: float | None = None,
    photo_sem_source: str = "gt",
    include_rooms: bool = False,
    structural_photo_semantics: bool = False,
    use_recon_cache: bool = True,
    eval_modes: Sequence[str] | None = None,
    eval_checkpoint_dir: str | None = None,
    eval_every_epochs: int = 1,
    inter_eval_scenes: int = 50,
    gcs_uri: str | None = None,
    no_gcs: bool = False,
    s3d_base_url: str | None = None,
    run_id: str | None = None,
    new_run: bool = False,
    tag: str | None = None,
    stages: Sequence[str] = ("prep", "train", "eval"),
    keep_zips: bool = False,
    keep_raw_views: bool = False,
    verify_gcp_auth: bool = True,
) -> PipelineSession:
    """Create a configured PipelineSession for Colab / interactive notebooks."""
    argv: list[str] = ["--workspace", workspace, "--stages", *stages]
    argv.extend(["--chunks", *(str(c) for c in chunks)])
    if all_train_chunks:
        argv.append("--all-train-chunks")
    argv.extend(["--slice-scenes", str(slice_scenes), "--epochs", str(epochs)])
    argv.extend(["--resolutions", *(str(r) for r in resolutions)])
    argv.extend(["--min-points", str(min_points), "--save-every", str(save_every)])
    argv.extend(["--init-lora", init_lora])
    argv.append("--freeze-lora" if freeze_lora else "--no-freeze-lora")
    argv.extend(["--sem-mode", sem_mode, "--lambda-sem", str(lambda_sem)])
    if alpha_sem is not None:
        argv.extend(["--alpha-sem", str(alpha_sem)])
    argv.extend(["--photo-sem-source", photo_sem_source])
    if eval_modes:
        argv.extend(["--eval-modes", *eval_modes])
    if eval_checkpoint_dir:
        argv.extend(["--eval-checkpoint-dir", eval_checkpoint_dir])
    argv.extend(["--eval-every-epochs", str(eval_every_epochs), "--inter-eval-scenes", str(inter_eval_scenes)])
    if include_rooms:
        argv.append("--include-rooms")
    if structural_photo_semantics:
        argv.append("--structural-photo-semantics")
    argv.append("--recon-cache" if use_recon_cache else "--no-recon-cache")
    if no_gcs:
        argv.append("--no-gcs")
    elif gcs_uri is not None:
        argv.extend(["--gcs-uri", gcs_uri])
    if s3d_base_url is not None:
        argv.extend(["--s3d-base-url", s3d_base_url])
    if run_id:
        argv.extend(["--run-id", run_id])
    if new_run:
        argv.append("--new-run")
    if tag:
        argv.extend(["--tag", tag])
    if keep_zips:
        argv.append("--keep-zips")
    if keep_raw_views:
        argv.append("--keep-raw-views")

    args = parse_args(argv)
    prep_profile = compute_prep_profile(
        args.photo_sem_source, args.include_rooms, args.structural_photo_semantics
    )
    resolved_ws = os.path.abspath(os.path.expanduser(workspace))
    probe_paths = build_workspace_paths(resolved_ws, "_init", prep_profile)
    os.makedirs(probe_paths["workspace"], exist_ok=True)
    os.makedirs(probe_paths["s3d_data_root"], exist_ok=True)
    migrate_local_legacy_s3d_data(probe_paths, prep_profile)

    cfg = resolve_config(args, probe_paths, verify_gcp_auth=verify_gcp_auth)
    resolved_run_id, resumed = resolve_run_id(args, resolved_ws, cfg["gcs_backup_uri"])
    paths = build_workspace_paths(resolved_ws, resolved_run_id, prep_profile)
    ensure_workspace_dirs(paths, prep_profile)
    if (resumed or bool(getattr(args, "run_id", None))) and cfg["gcs_backup_uri"]:
        restore_run_from_gcs(paths, cfg["gcs_backup_uri"], allow_log_overwrite=True)
    preflight_data_sources(args, cfg, paths)
    init_run_manifest(args, paths, cfg, resumed=resumed)
    status_word = "Resuming" if resumed else "Initialized"
    print(f"[pipeline] {status_word} run {resolved_run_id} (profile={prep_profile}, workspace={resolved_ws})")
    return PipelineSession(args, paths, cfg, resumed=resumed)

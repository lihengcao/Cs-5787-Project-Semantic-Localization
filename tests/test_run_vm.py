"""Tests for unified Colab/VM pipeline orchestrator and runners/ layout."""
from __future__ import annotations

import ast
import inspect
import json
import os
import shutil
import stat
import subprocess
import sys
import tarfile
import urllib.error
import urllib.request
from collections import namedtuple
from email.message import Message
from pathlib import Path
from typing import Any

import pytest

from runners import pipeline, run_vm

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _isolate_pipeline_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear runner/pipeline env vars so tests never inherit ambient host configuration."""
    for var in (
        "HF_HOME",
        "GCS_BACKUP_URI",
        "S3D_BASE_URL",
        "_SCENEALIGNER_BOOTSTRAPPED",
        "_SCENEALIGNER_DETACHED",
        "_SCENEALIGNER_RUN_RESUMED",
        "_SCENEALIGNER_SYNC_GCS_ENV",
    ):
        monkeypatch.delenv(var, raising=False)


def _populate_test_scenes(s3d_raw: Path, *, with_views: bool = True) -> None:
    for s in range(3250, 3500):
        sdir = s3d_raw / f"scene_{s:05d}"
        sdir.mkdir(parents=True, exist_ok=True)
        (sdir / "annotation_3d.json").write_text("{}", encoding="utf-8")
        if with_views:
            (sdir / "2D_rendering").mkdir(exist_ok=True)


def _install_fake_gcs(monkeypatch: pytest.MonkeyPatch) -> dict[str, bytes]:
    store: dict[str, bytes] = {}

    def _norm(u: str) -> str:
        return u.rstrip("/")

    def _exists(uri: str) -> bool:
        u = _norm(uri)
        if "*" in u:
            prefix, suffix = u.split("*", 1)
            return any(k.startswith(prefix) and k.endswith(suffix) for k in store)
        return u in store

    def _ls(pattern: str) -> list[str]:
        p = _norm(pattern)
        if "*" in p:
            prefix, suffix = p.split("*", 1)
            return sorted(k for k in store if k.startswith(prefix) and k.endswith(suffix))
        return sorted(k for k in store if k == p or k.startswith(p + "/"))

    def _cp(src: str, dst: str, *, check: bool = True) -> bool:
        del check
        if src.startswith("gs://") and dst.startswith("gs://"):
            store[_norm(dst)] = store[_norm(src)]
        elif dst.startswith("gs://"):
            store[_norm(dst)] = Path(src).read_bytes()
        elif src.startswith("gs://"):
            Path(dst).parent.mkdir(parents=True, exist_ok=True)
            Path(dst).write_bytes(store[_norm(src)])
        else:
            shutil.copy2(src, dst)
        return True

    def _rm(uri: str, *, check: bool = False) -> bool:
        del check
        store.pop(_norm(uri), None)
        return True

    def _rsync(src: str, dst: str, *, check: bool = True) -> bool:
        del check
        if dst.startswith("gs://"):
            base_dst = _norm(dst)
            root = Path(src)
            if root.is_dir():
                for p in root.rglob("*"):
                    if p.is_file():
                        rel = p.relative_to(root).as_posix()
                        store[f"{base_dst}/{rel}"] = p.read_bytes()
        elif src.startswith("gs://"):
            base_src = _norm(src) + "/"
            root = Path(dst)
            root.mkdir(parents=True, exist_ok=True)
            for k, val in list(store.items()):
                if k.startswith(base_src):
                    rel = k[len(base_src) :]
                    out_p = root / rel
                    out_p.parent.mkdir(parents=True, exist_ok=True)
                    out_p.write_bytes(val)
        return True

    def _read_text(uri: str) -> str | None:
        raw = store.get(_norm(uri))
        return None if raw is None else raw.decode("utf-8")

    def _write_text(uri: str, content: str, *, check: bool = False) -> bool:
        del check
        store[_norm(uri)] = content.encode("utf-8")
        return True

    monkeypatch.setattr(pipeline, "ensure_gcp_au" \
    "th", lambda _u: None)
    monkeypatch.setattr(pipeline, "gcs_exists", _exists)
    monkeypatch.setattr(pipeline, "gcs_ls", _ls)
    monkeypatch.setattr(pipeline, "gcs_cp", _cp)
    monkeypatch.setattr(pipeline, "gcs_rm", _rm)
    monkeypatch.setattr(pipeline, "gcs_rsync", _rsync)
    monkeypatch.setattr(pipeline, "_gcs_read_text", _read_text)
    monkeypatch.setattr(pipeline, "_gcs_write_text", _write_text)
    return store


def _notebook_cell_to_python(source: str) -> str:
    lines: list[str] = []
    for raw in source.splitlines():
        stripped = raw.lstrip()
        if stripped.startswith(("!", "%")):
            continue
        lines.append(raw)
    return "\n".join(lines)


def test_colab_notebook_contract() -> None:
    nb_path = REPO_ROOT / "runners" / "colab.ipynb"
    assert nb_path.exists()
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    cells = nb["cells"]
    code_cells = [c for c in cells if c.get("cell_type") == "code"]
    cells_by_id = {c.get("id"): "".join(c["source"]) for c in code_cells}

    expected_order = [
        "gcp-auth",
        "clone-install",
        "env-setup",
        "smoke-test",
        "prepare-s3d",
        "train-s3d",
        "eval-s3d",
        "gcs-backup",
    ]
    cell_ids = [c.get("id") for c in cells]
    for cid in expected_order:
        assert cid in cells_by_id, f"Missing cell {cid!r}"
    for earlier, later in zip(expected_order[:-1], expected_order[1:], strict=True):
        assert cell_ids.index(earlier) < cell_ids.index(later)

    parsed_trees: list[ast.Module] = []
    for cid, src in cells_by_id.items():
        assert "!7z x" not in src, f"Unchecked !7z x in cell {cid}"
        tree = ast.parse(_notebook_cell_to_python(src), filename=f"colab.ipynb:{cid}")
        parsed_trees.append(tree)

    init_session_params = set(inspect.signature(pipeline.init_session).parameters)
    init_calls = 0
    for tree in parsed_trees:
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            attr = node.func.attr
            if isinstance(node.func.value, ast.Name) and node.func.value.id == "pipeline":
                assert hasattr(pipeline, attr), f"pipeline.{attr} does not exist"
                if attr == "init_session":
                    init_calls += 1
                    for kw in node.keywords:
                        if kw.arg is not None:
                            assert kw.arg in init_session_params, f"Unknown init_session kwarg: {kw.arg}"
            elif isinstance(node.func.value, ast.Name) and node.func.value.id == "SESSION":
                assert hasattr(pipeline.PipelineSession, attr), f"PipelineSession.{attr} does not exist"
    assert init_calls == 1


def test_sync_sh_strips_outputs_by_default_and_supports_keep_outputs(tmp_path: Path) -> None:
    repo = tmp_path / "sync_repo"
    (repo / "runners").mkdir(parents=True)
    sync_copy = repo / "runners" / "sync.sh"
    shutil.copy2(REPO_ROOT / "runners" / "sync.sh", sync_copy)
    sync_copy.chmod(0o755)
    bash_syntax = subprocess.run(["bash", "-n", str(sync_copy)], capture_output=True, text=True, check=False)
    assert bash_syntax.returncode == 0, bash_syntax.stderr

    nb_path = repo / "runners" / "colab.ipynb"

    def _write_nb(outputs: list[dict[str, Any]], exec_count: int | None) -> None:
        meta = {"outputId": "oid", "executionInfo": {"status": "ok"}} if exec_count is not None else {}
        nb_obj = {
            "cells": [
                {
                    "id": "c1",
                    "cell_type": "code",
                    "execution_count": exec_count,
                    "metadata": meta,
                    "outputs": outputs,
                    "source": ["print('hi')\n"],
                }
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        nb_path.write_text(json.dumps(nb_obj, indent=2) + "\n", encoding="utf-8")

    # 1. Default --check-only strips outputs, execution_count, and execution metadata
    _write_nb([{"output_type": "stream", "text": ["ok\n"]}], 3)
    res_strip = subprocess.run(
        ["bash", str(sync_copy), "--repo-root", str(repo), "--check-only"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res_strip.returncode == 0, res_strip.stderr
    assert "stripped cell outputs" in res_strip.stdout
    stripped = json.loads(nb_path.read_text(encoding="utf-8"))
    assert stripped["cells"][0]["outputs"] == []
    assert stripped["cells"][0]["execution_count"] is None
    assert stripped["cells"][0]["metadata"] == {}

    # 2. Second --check-only on already-clean notebook is a no-op
    res_clean = subprocess.run(
        ["bash", str(sync_copy), "--repo-root", str(repo), "--check-only"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res_clean.returncode == 0, res_clean.stderr
    assert "stripped cell outputs" not in res_clean.stdout

    # 3. --keep-outputs and -k preserve outputs and execution_count
    for flag in ("--keep-outputs", "-k"):
        _write_nb([{"output_type": "stream", "text": ["kept output\n"]}], 5)
        res_keep = subprocess.run(
            ["bash", str(sync_copy), "--repo-root", str(repo), "--check-only", flag],
            capture_output=True,
            text=True,
            check=False,
        )
        assert res_keep.returncode == 0, res_keep.stderr
        kept = json.loads(nb_path.read_text(encoding="utf-8"))
        assert kept["cells"][0]["outputs"] == [{"output_type": "stream", "text": ["kept output\n"]}]
        assert kept["cells"][0]["execution_count"] == 5


def test_pipeline_archive_s3d_slice_packs_floorplan_sem(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ws = tmp_path / "ws"
    paths = pipeline.build_workspace_paths(str(ws), "test_run", "photo-gt")
    pipeline.ensure_workspace_dirs(paths, "photo-gt")

    monkeypatch.setattr(pipeline, "gcs_exists", lambda _u: False)
    with pytest.raises(FileNotFoundError, match="scenes_0_1.csv"):
        pipeline.archive_s3d_slice(paths, "gs://bucket", 0, 1, prep_profile="photo-gt", overwrite=True)

    data_dir = Path(paths["s3d_data"])
    for rel in (
        "scenes_0_1.csv",
        "floorplan/scene_00000.png",
        "floorplan_sem/scene_00000_sem_idx.png",
        "floorplan_sem/scene_00000_palette.npy",
        "floorplan_sem/scene_00000_labels.json",
        "pi3/scene_00000/r0_0.png",
        "pi3_sem/scene_00000/r0_0_sem_idx.png",
    ):
        p = data_dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")

    captured_tar = tmp_path / "captured.tar"
    cp_dsts: list[str] = []

    def _fake_gcs_cp(src: str, dst: str, *, check: bool = True) -> bool:
        del check
        cp_dsts.append(dst)
        captured_tar.write_bytes(Path(src).read_bytes())
        return True

    monkeypatch.setattr(pipeline, "gcs_cp", _fake_gcs_cp)
    pipeline.archive_s3d_slice(paths, "gs://bucket", 0, 1, prep_profile="photo-gt", overwrite=True)

    assert cp_dsts == ["gs://bucket/s3d_data/photo-gt/scenes_0_1.tar"]
    with tarfile.open(captured_tar, "r") as tf:
        members = set(tf.getnames())
    assert members == {
        "scenes_0_1.csv",
        "floorplan/scene_00000.png",
        "floorplan_sem/scene_00000_sem_idx.png",
        "floorplan_sem/scene_00000_palette.npy",
        "floorplan_sem/scene_00000_labels.json",
        "pi3/scene_00000",
        "pi3/scene_00000/r0_0.png",
        "pi3_sem/scene_00000",
        "pi3_sem/scene_00000/r0_0_sem_idx.png",
    }


def test_workspace_init_and_env_config_zero_setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ws = tmp_path / "fresh_ws"
    paths = pipeline.build_workspace_paths(str(ws), "exp_vm", "photo-gt")
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("GCS_BACKUP_URI", raising=False)
    monkeypatch.delenv("S3D_BASE_URL", raising=False)
    monkeypatch.setattr(pipeline, "ensure_gcp_auth", lambda _u: None)
    monkeypatch.setattr(pipeline, "gcs_exists", lambda _u: False)

    pipeline.ensure_workspace_dirs(paths, "photo-gt")
    for key in ("workspace", "raw_zips", "s3d_raw", "s3d_data", "runs_dir", "run_out", "hf_cache"):
        assert Path(paths[key]).is_dir()
    assert paths["s3d_data"] == str(ws / "s3d_data" / "photo-gt")
    assert os.environ["HF_HOME"] == str(ws / "hf_cache")

    args = pipeline.parse_args(
        ["--gcs-uri", "scenealigner/", "--s3d-base-url", "https://example.org/s3d/README.txt"]
    )
    cfg = pipeline.resolve_config(args, paths)
    assert cfg["gcs_backup_uri"] == "gs://scenealigner"
    assert cfg["s3d_base_url"] == "https://example.org/s3d"

    monkeypatch.delenv("GCS_BACKUP_URI", raising=False)
    monkeypatch.delenv("S3D_BASE_URL", raising=False)
    default_cfg = pipeline.resolve_config(pipeline.parse_args([]), paths)
    assert default_cfg["gcs_backup_uri"] == "gs://scenealigner"

    monkeypatch.delenv("GCS_BACKUP_URI", raising=False)
    no_gcs_cfg = pipeline.resolve_config(pipeline.parse_args(["--no-gcs"]), paths)
    assert no_gcs_cfg["gcs_backup_uri"] == ""

    pipeline._upsert_env_key(paths["env_file"], "GCS_BACKUP_URI", "")
    monkeypatch.delenv("GCS_BACKUP_URI", raising=False)
    empty_env_cfg = pipeline.resolve_config(pipeline.parse_args([]), paths)
    assert empty_env_cfg["gcs_backup_uri"] == ""

    pipeline._upsert_env_key(paths["env_file"], "GCS_BACKUP_URI", "gs://scenealigner")
    pipeline._upsert_env_key(paths["env_file"], "GCS_BACKUP_URI", "gs://updated")
    env_text = Path(paths["env_file"]).read_text(encoding="utf-8")
    assert env_text.count("GCS_BACKUP_URI=") == 1
    assert "GCS_BACKUP_URI=gs://updated" in env_text
    assert stat.S_IMODE(os.stat(paths["env_file"]).st_mode) == 0o600


def test_normalize_s3d_url_handles_paste_glitches() -> None:
    expected = "https://example.org/s3d"
    samples = [
        '"https://example.org/s3d"',
        'https://example.org/s3d"',
        "`https://example.org/s3d`",
        'S3D_BASE_URL = "https://example.org/s3d"',
        'export S3D_BASE_URL="https://example.org/s3d"',
        "\x1b[200~https://example.org/s3d\x1b[201~",
        "https://example.org/s3d/README.txt",
        "https://example.org/s3d/Structured3D_annotation_3d.zip",
        "https://example.org/s3dhttps://example.org/s3d",
        "https://example.org/s3d https://example.org/s3d",
    ]
    for raw in samples:
        assert pipeline._normalize_s3d_url(raw) == expected, f"Failed to normalize {raw!r}"


def test_verify_s3d_base_url_and_interactive_prompt_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws = tmp_path / "ws_verify"
    paths = pipeline.build_workspace_paths(str(ws), "exp_verify")
    pipeline.ensure_workspace_dirs(paths)
    monkeypatch.delenv("S3D_BASE_URL", raising=False)
    monkeypatch.delenv("_SCENEALIGNER_BOOTSTRAPPED", raising=False)
    monkeypatch.setattr(pipeline.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(pipeline.sys.stdout, "isatty", lambda: True)

    bad_secret_url = "https://secret-bad.example.org/typo"
    good_secret_url = "https://secret-good.example.org/s3d"
    probe_log: list[tuple[str, str, dict[str, str]]] = []

    class _FakeResp:
        def __init__(self, status: int) -> None:
            self.status = status

        def __enter__(self) -> _FakeResp:
            return self

        def __exit__(self, *_args: Any) -> bool:
            return False

    def _fake_urlopen(req: urllib.request.Request, timeout: float = 8.0) -> _FakeResp:
        del timeout
        method = req.get_method()
        headers = {k.lower(): v for k, v in req.header_items()}
        probe_log.append((method, req.full_url, headers))
        if req.full_url.startswith(bad_secret_url):
            raise urllib.error.HTTPError(req.full_url, 404, "Not Found", Message(), None)
        if req.full_url.startswith(good_secret_url):
            if method == "HEAD":
                raise urllib.error.HTTPError(req.full_url, 405, "Method Not Allowed", Message(), None)
            if method == "GET" and headers.get("range") == "bytes=0-0":
                return _FakeResp(206)
        raise AssertionError(f"Unexpected probe: {method} {req.full_url}")

    prompts = iter([f'"{bad_secret_url}"', f"{good_secret_url}/README.txt"])

    def _fake_getpass(_prompt: str) -> str:
        env_path = Path(paths["env_file"])
        assert not env_path.exists()
        return next(prompts)

    monkeypatch.setattr(pipeline.urllib.request, "urlopen", _fake_urlopen)
    monkeypatch.setattr(pipeline.getpass, "getpass", _fake_getpass)

    cfg = pipeline.resolve_config(pipeline.parse_args(["--no-gcs", "--stages", "prep"]), paths)
    assert cfg["s3d_base_url"] == good_secret_url
    env_text = Path(paths["env_file"]).read_text(encoding="utf-8")
    assert f"S3D_BASE_URL={good_secret_url}" in env_text
    assert bad_secret_url not in env_text
    assert (
        "GET",
        f"{good_secret_url}/Structured3D_annotation_3d.zip",
        {"user-agent": "aria2/1.37.0", "range": "bytes=0-0"},
    ) in probe_log

    stale_secret = "https://stale-leaked-host.example.org/broken"
    pipeline._upsert_env_key(paths["env_file"], "S3D_BASE_URL", stale_secret)
    monkeypatch.delenv("S3D_BASE_URL", raising=False)
    monkeypatch.setattr(pipeline.sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(pipeline.sys.stdout, "isatty", lambda: False)
    monkeypatch.setattr(
        pipeline.urllib.request,
        "urlopen",
        lambda req, timeout=8.0: (_ for _ in ()).throw(
            urllib.error.HTTPError(req.full_url, 404, "Not Found", Message(), None)
        ),
    )

    with pytest.raises(RuntimeError) as exc_info:
        pipeline.resolve_config(pipeline.parse_args(["--no-gcs", "--stages", "prep"]), paths)
    msg = str(exc_info.value)
    assert "<S3D_BASE_URL>" in msg
    assert stale_secret not in msg


def test_run_prep_stage_skips_annotation_zip_when_chunks_covered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws = tmp_path / "ws_covered"
    paths = pipeline.build_workspace_paths(str(ws), "exp_covered")
    pipeline.ensure_workspace_dirs(paths)
    (Path(paths["s3d_data"]) / "scenes_0_200.csv").write_text("scene_id\n", encoding="utf-8")

    monkeypatch.setattr(pipeline, "restore_s3d_tars", lambda *_a, **_k: None)
    monkeypatch.setattr(pipeline, "ensure_floorplan_sem", lambda *_a, **_k: None)
    monkeypatch.setattr(pipeline, "fetch_zip", lambda *_a, **_k: pytest.fail("fetch_zip should not be called"))

    log_writer = pipeline._CleanLogWriter(paths["log_path"], echo_to_stdout=False, progress_interval_sec=0.0)
    try:
        args = pipeline.parse_args(["--no-gcs", "--stages", "prep", "--chunks", "0"])
        pipeline.run_prep_stage(args, paths, {"gcs_backup_uri": "", "s3d_base_url": ""}, log_writer)
    finally:
        log_writer.close()

    assert "Chunk 0 (0..200) already prepared; skipping." in Path(paths["log_path"]).read_text(encoding="utf-8")


def test_preflight_fails_fast_when_eval_data_missing(tmp_path: Path) -> None:
    paths = pipeline.build_workspace_paths(str(tmp_path / "ws"), "r1")
    pipeline.ensure_workspace_dirs(paths)
    args = pipeline.parse_args(["--stages", "eval", "--no-gcs"])
    cfg = {"gcs_backup_uri": "", "s3d_base_url": ""}

    with pytest.raises(RuntimeError, match="Structured3D_perspective_full_16.zip"):
        pipeline.preflight_data_sources(args, cfg, paths)

    _populate_test_scenes(Path(paths["s3d_raw"]), with_views=True)
    pipeline.preflight_data_sources(args, cfg, paths)

    bad_train = pipeline.parse_args(["--stages", "train", "--sem-mode", "none", "--no-gcs"])
    with pytest.raises(ValueError, match="--no-freeze-lora"):
        pipeline.preflight_data_sources(bad_train, cfg, paths)


def test_ensure_bootstrapped_invokes_setup_and_reexecs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_repo = tmp_path / "repo"
    (fake_repo / "runners").mkdir(parents=True)
    (fake_repo / "requirements.txt").write_text("numpy\n", encoding="utf-8")
    venv_bin = fake_repo / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    fake_venv_py = venv_bin / "python"
    fake_venv_py.write_text("#!/bin/sh\n", encoding="utf-8")
    fake_venv_py.chmod(0o755)

    monkeypatch.setattr(run_vm, "REPO_DIR", str(fake_repo))
    monkeypatch.setattr(pipeline, "_find_7z", lambda required=False: "/usr/bin/7z")
    monkeypatch.setattr(run_vm.shutil, "which", lambda name: "/usr/bin/aria2c" if name == "aria2c" else None)

    setup_calls: list[list[str]] = []
    exec_calls: list[tuple[str, list[str]]] = []

    def _fake_run(cmd: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        setup_calls.append(cmd)
        stamp = fake_repo / ".venv" / ".scenealigner_ready"
        stamp.write_text(pipeline._requirements_hash(str(fake_repo)) + "\n", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0)

    def _fake_execv(path: str, argv: list[str]) -> None:
        exec_calls.append((path, argv))
        raise SystemExit(99)

    monkeypatch.setattr(run_vm.subprocess, "run", _fake_run)
    monkeypatch.setattr(run_vm.os, "execv", _fake_execv)

    args = pipeline.parse_args(["--stages", "prep", "--no-gcs"])
    ws = str(tmp_path / "my_ws")
    with pytest.raises(SystemExit, match="99"):
        run_vm.ensure_bootstrapped(args, ["--stages", "prep", "--no-gcs"], ws, "run_123")

    assert setup_calls == [["bash", str(fake_repo / "runners" / "setup_vm.sh")]]
    assert len(exec_calls) == 1
    assert exec_calls[0][0] == str(fake_venv_py)
    assert "--workspace" in exec_calls[0][1] and ws in exec_calls[0][1]
    assert "--run-id" in exec_calls[0][1] and "run_123" in exec_calls[0][1]
    assert os.environ.get("_SCENEALIGNER_BOOTSTRAPPED") == "1"

    setup_calls.clear()
    with pytest.raises(SystemExit, match="99"):
        run_vm.ensure_bootstrapped(args, ["--stages", "prep", "--no-gcs"], ws, "run_123")
    assert setup_calls == []


def test_clean_log_writer_redacts_secrets_and_collapses_cr(tmp_path: Path) -> None:
    log_path = tmp_path / "pipeline.log"
    secret = "https://secret.example.com/token123"
    writer = pipeline._CleanLogWriter(str(log_path), redact=secret, echo_to_stdout=False, progress_interval_sec=0.0)
    try:
        writer.log_line(f"Downloading from {secret}/file.zip")
        writer.log_line("Connecting to secret.example.com:443...")
        writer.write_chunk("\x1b[32mstep 1/3\x1b[0m\r\x1b[32mstep 2/3\r")
        writer.write_chunk("step 3/3 done\r\nfinal line")
    finally:
        writer.close()

    content = log_path.read_text(encoding="utf-8")
    assert secret not in content
    assert "secret.example.com" not in content
    assert "<REDACTED>/file.zip" in content
    assert "Connecting to <REDACTED>:443..." in content
    assert "\x1b[" not in content
    assert "step 1/3" not in content
    assert "step 2/3" not in content
    assert "step 3/3 done\n" in content
    assert "final line\n" in content


def test_maybe_detach_spawns_background_session_and_writes_pid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = pipeline.build_workspace_paths(str(tmp_path / "ws"), "detach_run")
    pipeline.ensure_workspace_dirs(paths)
    args = pipeline.parse_args(["--detach", "--stages", "train", "--no-gcs"])

    spawned: list[dict[str, Any]] = []

    class _FakeProc:
        pid = 42424

    def _fake_popen(cmd: list[str], **kwargs: Any) -> _FakeProc:
        spawned.append({"cmd": cmd, "kwargs": kwargs})
        return _FakeProc()

    monkeypatch.setattr(run_vm.subprocess, "Popen", _fake_popen)
    monkeypatch.delenv("_SCENEALIGNER_DETACHED", raising=False)

    assert (
        run_vm.maybe_detach(
            args,
            ["--detach", "--stages", "train", "--no-gcs"],
            paths,
            "",
            "detach_run",
        )
        is True
    )
    assert len(spawned) == 1
    assert "--detach" not in spawned[0]["cmd"]
    assert "--run-id" in spawned[0]["cmd"] and "detach_run" in spawned[0]["cmd"]
    assert spawned[0]["kwargs"]["start_new_session"] is True
    assert spawned[0]["kwargs"]["env"]["_SCENEALIGNER_DETACHED"] == "1"
    assert Path(paths["pid_path"]).read_text(encoding="utf-8").strip() == "42424"

    monkeypatch.setenv("_SCENEALIGNER_DETACHED", "1")
    assert (
        run_vm.maybe_detach(
            args,
            ["--detach", "--stages", "train", "--no-gcs"],
            paths,
            "",
            "detach_run",
        )
        is False
    )


def test_external_cwd_invocation_and_install_colab_deps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_repo = tmp_path / "repo"
    fake_repo.mkdir()
    (fake_repo / "requirements.txt").write_text(
        "numpy==2.3.5\npeft==0.18.1\npillow>=10\n",
        encoding="utf-8",
    )
    ext_cwd = tmp_path / "outside"
    ext_cwd.mkdir()
    (ext_cwd / "rel_ckpt").mkdir()
    monkeypatch.chdir(ext_cwd)

    monkeypatch.setattr(run_vm, "REPO_DIR", str(fake_repo))
    monkeypatch.setattr(pipeline, "REPO_DIR", str(fake_repo))

    live_calls: list[tuple[list[str], str | None]] = []

    def _fake_run_live(cmd: list[str], _lw: Any = None, **kwargs: Any) -> None:
        live_calls.append((cmd, kwargs.get("cwd")))

    monkeypatch.setattr(pipeline, "run_live", _fake_run_live)
    rc = run_vm.main(
        ["--stages", "smoke", "--skip-setup", "--no-gcs", "--eval-checkpoint-dir", "./rel_ckpt"]
    )
    assert rc == 0
    assert Path(os.getcwd()).resolve() == fake_repo.resolve()
    assert (fake_repo / "outputs" / "latest_results.json").exists()
    summary = json.loads((fake_repo / "outputs" / "latest_results.json").read_text(encoding="utf-8"))
    assert Path(summary["workspace"]).resolve() == (fake_repo / "outputs").resolve()
    assert summary["config"]["photo_sem_source"] == "gt"
    assert any(Path(c[2]).name == "demo.py" or "pytest" in " ".join(c) for c, _ in live_calls)

    pip_calls: list[tuple[list[str], str | None]] = []

    def _fake_subproc_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        pip_calls.append((cmd, kwargs.get("cwd")))
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(pipeline.subprocess, "run", _fake_subproc_run)
    installed = pipeline.install_colab_deps()
    assert installed == ["peft==0.18.1", "pytest"]
    assert pip_calls and pip_calls[0][1] == str(fake_repo)


def test_slug_and_profile_formatting() -> None:
    assert pipeline.compute_prep_profile("gt", False, False) == "photo-gt"
    assert pipeline.compute_prep_profile("gt", True, False) == "photo-gt_rooms"
    assert pipeline.compute_prep_profile("gt", False, True) == "photo-gt_struct"
    assert pipeline.compute_prep_profile("none", False, False) == "photo-none"
    assert pipeline.compute_prep_profile("none", True, False) == "photo-none_rooms"

    assert pipeline.format_chunks_slug([0, 1, 2]) == "c0-2"
    assert pipeline.format_chunks_slug([0, 2, 3]) == "c0.2-3"
    assert pipeline.format_chunks_slug([0, 8, 9]) == "c0.9"
    assert pipeline.format_chunks_slug([], all_train_chunks=True) == "c-all"

    args = pipeline.parse_args(["--chunks", "0", "1", "2", "--epochs", "10", "--tag", "v2 clean"])
    assert pipeline.build_run_slug(args) == "s3d_photo-gt_consensus_c0-2_ep10_v2-clean"


def test_gcs_zero_download_config_env_and_zip_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gcs = _install_fake_gcs(monkeypatch)
    ws = tmp_path / "ws"
    paths = pipeline.build_workspace_paths(str(ws), "run_env", "photo-gt")
    pipeline.ensure_workspace_dirs(paths, "photo-gt")

    # 1. GCS .env sync & zero-download detection
    gcs["gs://scenealigner/config/.env"] = b"S3D_BASE_URL=https://secret.example.org/s3d\n"
    buf_tar = tmp_path / "empty_slice.tar"
    with tarfile.open(buf_tar, "w") as tf:
        dummy = tmp_path / "scenes_0_50.csv"
        dummy.write_text("scene_id\n", encoding="utf-8")
        tf.add(dummy, arcname="scenes_0_50.csv")
    slice_bytes = buf_tar.read_bytes()
    for s0 in (0, 50, 100, 150):
        gcs[f"gs://scenealigner/s3d_data/photo-gt/scenes_{s0}_{s0 + 50}.tar"] = slice_bytes
    gcs[f"gs://scenealigner/s3d_eval/{pipeline.TEST_SCENES_TAR}"] = slice_bytes

    monkeypatch.setattr(
        pipeline.urllib.request,
        "urlopen",
        lambda *_a, **_k: pytest.fail("urlopen should not probe when GCS covers everything"),
    )
    args = pipeline.parse_args(["--stages", "prep", "train", "eval", "--chunks", "0"])
    cfg = pipeline.resolve_config(args, paths, verify_gcp_auth=True)
    assert cfg["s3d_base_url"] == "https://secret.example.org/s3d"
    assert pipeline.needs_external_s3d_download(args, paths, cfg["gcs_backup_uri"], "photo-gt") is False

    # 2. Stale broken GCS/local .env is healed from Colab Secrets when verified
    gcs["gs://scenealigner/config/.env"] = b"S3D_BASE_URL=https://expired.example.org/old\n"
    Path(paths["env_file"]).unlink()
    gcs.pop(f"gs://scenealigner/s3d_eval/{pipeline.TEST_SCENES_TAR}", None)
    monkeypatch.delenv("S3D_BASE_URL", raising=False)
    monkeypatch.setattr(
        pipeline,
        "_colab_userdata_get",
        lambda key: "https://fresh.example.org/new" if key == "S3D_BASE_URL" else "",
    )

    class _OkResp:
        status = 200

        def __enter__(self) -> _OkResp:
            return self

        def __exit__(self, *_a: Any) -> bool:
            return False

    def _verify_fresh(req: urllib.request.Request, timeout: float = 8.0) -> _OkResp:
        del timeout
        if req.full_url.startswith("https://fresh.example.org/new"):
            return _OkResp()
        raise urllib.error.HTTPError(req.full_url, 403, "Forbidden", Message(), None)

    monkeypatch.setattr(pipeline.urllib.request, "urlopen", _verify_fresh)
    healed_cfg = pipeline.resolve_config(args, paths, verify_gcp_auth=True)
    assert healed_cfg["s3d_base_url"] == "https://fresh.example.org/new"
    assert b"S3D_BASE_URL=https://fresh.example.org/new" in gcs["gs://scenealigner/config/.env"]

    # 3. fetch_zip persists annotation zip to GCS, restores it, and falls back to aria2c on GCS cp error
    lw = pipeline._CleanLogWriter(paths["log_path"], echo_to_stdout=False, progress_interval_sec=0.0)
    try:
        anno_zip = Path(paths["raw_zips"]) / "Structured3D_annotation_3d.zip"
        anno_zip.write_bytes(b"zipdata")
        pipeline.fetch_zip("Structured3D_annotation_3d.zip", paths, healed_cfg, lw)
        assert gcs.get("gs://scenealigner/raw_zips/Structured3D_annotation_3d.zip") == b"zipdata"
        anno_zip.unlink()
        monkeypatch.setattr(pipeline, "run_live", lambda *_a, **_k: pytest.fail("aria2c should not run"))
        got = pipeline.fetch_zip("Structured3D_annotation_3d.zip", paths, healed_cfg, lw)
        assert Path(got).read_bytes() == b"zipdata"

        anno_zip.unlink()

        def _failing_download_cp(src: str, dst: str, *, check: bool = True) -> bool:
            if src.startswith("gs://"):
                raise subprocess.CalledProcessError(1, ["gcloud"])
            gcs[dst.rstrip("/")] = Path(src).read_bytes()
            return True

        monkeypatch.setattr(pipeline, "gcs_cp", _failing_download_cp)
        aria_calls: list[list[str]] = []

        def _fake_aria(cmd: list[str], *_a: Any, **_k: Any) -> None:
            aria_calls.append(cmd)
            anno_zip.write_bytes(b"freshzip")

        monkeypatch.setattr(pipeline, "run_live", _fake_aria)
        got_fallback = pipeline.fetch_zip("Structured3D_annotation_3d.zip", paths, healed_cfg, lw)
        assert Path(got_fallback).read_bytes() == b"freshzip"
        assert aria_calls and aria_calls[0][0] == "aria2c"
    finally:
        lw.close()


def test_gcs_crash_safe_legacy_slice_migration_and_floorplan_sem_healing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gcs = _install_fake_gcs(monkeypatch)
    ws = tmp_path / "ws"
    paths = pipeline.build_workspace_paths(str(ws), "run_mig", "photo-gt")
    pipeline.ensure_workspace_dirs(paths, "photo-gt")
    cfg = {"gcs_backup_uri": "gs://scenealigner", "s3d_base_url": ""}
    args = pipeline.parse_args(["--stages", "prep", "train", "--chunks", "0"])

    legacy_tar = tmp_path / "legacy_0_50.tar"
    legacy_csv = tmp_path / "scenes_0_50_leg.csv"
    legacy_csv.write_text(
        "scene_id,floorplan,floorplan_sem\nscene_00000,floorplan/scene_00000.png,floorplan_sem/scene_00000\n",
        encoding="utf-8",
    )
    with tarfile.open(legacy_tar, "w") as tf:
        tf.add(legacy_csv, arcname="scenes_0_50.csv")
    gcs["gs://scenealigner/s3d_data/scenes_0_50.tar"] = legacy_tar.read_bytes()

    lw = pipeline._CleanLogWriter(paths["log_path"], echo_to_stdout=False, progress_interval_sec=0.0)
    try:
        # 1. restore_s3d_tars keeps legacy flat GCS tar intact until floorplan_sem is healed (crash-safe)
        pipeline.restore_s3d_tars(paths, "gs://scenealigner", lw, prep_profile="photo-gt")
        assert (Path(paths["s3d_data"]) / "scenes_0_50.csv").exists()
        assert "gs://scenealigner/s3d_data/scenes_0_50.tar" in gcs
        assert "gs://scenealigner/s3d_data/photo-gt/scenes_0_50.tar" not in gcs

        # 2. Local flat migration into s3d_data/photo-gt/
        ws_mig = tmp_path / "ws_mig"
        mig_paths = pipeline.build_workspace_paths(str(ws_mig), "mig", "photo-gt")
        (ws_mig / "s3d_data" / "floorplan").mkdir(parents=True)
        (ws_mig / "s3d_data" / "scenes_0_50.csv").write_text("scene_id\n", encoding="utf-8")
        (ws_mig / "s3d_data" / "floorplan" / "scene_00000.png").write_bytes(b"p")
        pipeline.ensure_workspace_dirs(mig_paths, "photo-gt")
        assert (Path(mig_paths["s3d_data"]) / "scenes_0_50.csv").exists()
        assert (Path(mig_paths["s3d_data"]) / "floorplan" / "scene_00000.png").exists()

        # 3. ensure_floorplan_sem heals missing files, archives photo-gt tar, and retires legacy tar
        (Path(paths["s3d_raw"]) / "scene_00000").mkdir(parents=True, exist_ok=True)
        (Path(paths["s3d_raw"]) / "scene_00000" / "annotation_3d.json").write_text("{}", encoding="utf-8")

        def _fake_heal_live(cmd: list[str], *_a: Any, **_k: Any) -> None:
            if len(cmd) >= 3 and cmd[1] == "-c":
                sem_dir = Path(paths["s3d_data"]) / "floorplan_sem"
                sem_dir.mkdir(parents=True, exist_ok=True)
                (sem_dir / "scene_00000_sem_idx.png").write_bytes(b"idx")
                (sem_dir / "scene_00000_palette.npy").write_bytes(b"pal")

        monkeypatch.setattr(pipeline, "run_live", _fake_heal_live)
        pipeline.ensure_floorplan_sem(args, paths, cfg, lw, prep_profile="photo-gt")
        assert "gs://scenealigner/s3d_data/scenes_0_50.tar" not in gcs
        healed_check = tmp_path / "healed_check.tar"
        healed_check.write_bytes(gcs["gs://scenealigner/s3d_data/photo-gt/scenes_0_50.tar"])
        with tarfile.open(healed_check, "r") as tf:
            names = set(tf.getnames())
        assert "floorplan_sem/scene_00000_sem_idx.png" in names
        assert "floorplan_sem/scene_00000_palette.npy" in names

        # 4. run_train_stage restores missing chunk 1 slice tar and passes --scene_ranges
        for s0, s1 in ((50, 100), (100, 150), (150, 200)):
            (Path(paths["s3d_data"]) / f"scenes_{s0}_{s1}.csv").write_text(
                f"scene_id\nscene_{s0:05d}\n", encoding="utf-8"
            )
        for s0, s1 in ((250, 300), (300, 350), (350, 400)):
            (Path(paths["s3d_data"]) / f"scenes_{s0}_{s1}.csv").write_text(
                f"scene_id\nscene_{s0:05d}\n", encoding="utf-8"
            )
        c1_csv = tmp_path / "scenes_200_250.csv"
        c1_csv.write_text("scene_id\nscene_00200\n", encoding="utf-8")
        c1_tar = tmp_path / "scenes_200_250.tar"
        with tarfile.open(c1_tar, "w") as tf:
            tf.add(c1_csv, arcname="scenes_200_250.csv")
        gcs["gs://scenealigner/s3d_data/photo-gt/scenes_200_250.tar"] = c1_tar.read_bytes()
        train_cmds: list[list[str]] = []

        def _fake_train_live(cmd: list[str], *_a: Any, **_k: Any) -> None:
            train_cmds.append(cmd)
            out_dir = Path(cmd[cmd.index("--output") + 1])
            (out_dir / "checkpoint.pt").write_bytes(b"ckpt")

        monkeypatch.setattr(pipeline, "run_live", _fake_train_live)
        train_args = pipeline.parse_args(
            ["--stages", "train", "--chunks", "0", "1", "--slice-scenes", "50", "--eval-every-epochs", "0"]
        )
        pipeline.run_train_stage(train_args, paths, cfg, lw)
        assert (Path(paths["s3d_data"]) / "scenes_200_250.csv").exists()
        assert len(train_cmds) == 1
        idx = train_cmds[0].index("--scene_ranges")
        assert train_cmds[0][idx + 1 : idx + 3] == ["0:200", "200:400"]
    finally:
        lw.close()


def test_gcs_corrupt_archives_partial_eval_and_train_guards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gcs = _install_fake_gcs(monkeypatch)
    ws = tmp_path / "ws_corrupt"
    paths = pipeline.build_workspace_paths(str(ws), "run_corrupt", "photo-gt")
    pipeline.ensure_workspace_dirs(paths, "photo-gt")
    cfg = {"gcs_backup_uri": "gs://scenealigner", "s3d_base_url": ""}

    # 1. Train-without-prep preflight rejects missing chunk slices
    train_missing = pipeline.parse_args(["--stages", "train", "--chunks", "0", "2"])
    with pytest.raises(RuntimeError, match="Train stage requires prepared slices"):
        pipeline.preflight_data_sources(train_missing, cfg, paths)

    # 2. Partial eval scenes (sentinels 03250/03400 present, scene_03450 views missing) still flagged
    _populate_test_scenes(Path(paths["s3d_raw"]), with_views=True)
    shutil.rmtree(Path(paths["s3d_raw"]) / "scene_03450" / "2D_rendering")
    eval_args = pipeline.parse_args(["--stages", "eval"])
    assert pipeline.needs_external_s3d_download(eval_args, paths, "gs://scenealigner", "photo-gt") is True
    with pytest.raises(RuntimeError, match="Structured3D_perspective_full_17.zip"):
        pipeline.preflight_data_sources(eval_args, cfg, paths)
    (Path(paths["s3d_raw"]) / "scene_03450" / "2D_rendering").mkdir()

    # 3. 0-byte recon .npz rejected by _is_recon_shard_complete
    recon_dir = Path(paths["s3d_recon_cache"])
    v0 = Path(paths["s3d_raw"]) / "scene_03250" / "2D_rendering" / "0" / "perspective" / "full" / "0"
    v0.mkdir(parents=True, exist_ok=True)
    for req in ("camera_pose.txt", "depth.png", "rgb_rawlight.png"):
        (v0 / req).write_text("ok", encoding="utf-8")
    shard_sdir = recon_dir / "scene_03250"
    shard_sdir.mkdir(parents=True)
    npz_path = shard_sdir / "2D_rendering__0__perspective__full__0.npz"
    npz_path.write_bytes(b"")
    assert not pipeline._is_recon_shard_complete(str(recon_dir), 3250, 3300, s3d_raw_dir=paths["s3d_raw"])
    npz_path.write_bytes(b"v0")
    assert pipeline._is_recon_shard_complete(str(recon_dir), 3250, 3300, s3d_raw_dir=paths["s3d_raw"])

    lw = pipeline._CleanLogWriter(paths["log_path"], echo_to_stdout=False, progress_interval_sec=0.0)
    try:
        # 4. Corrupt slice tar removed from GCS; run_train_stage raises instead of silently proceeding
        gcs["gs://scenealigner/s3d_data/photo-gt/scenes_0_200.tar"] = b"not-a-tar"
        train_c0 = pipeline.parse_args(["--stages", "train", "--chunks", "0"])
        with pytest.raises(RuntimeError, match="missing locally after GCS restore"):
            pipeline.run_train_stage(train_c0, paths, cfg, lw)
        assert "gs://scenealigner/s3d_data/photo-gt/scenes_0_200.tar" not in gcs

        # 5. Corrupt test_scenes_3250_3500.tar triggers zip fallback and overwrites broken GCS tar
        gcs[f"gs://scenealigner/s3d_eval/{pipeline.TEST_SCENES_TAR}"] = b"corrupt-test-tar"
        shutil.rmtree(paths["s3d_raw"])
        Path(paths["s3d_raw"]).mkdir()

        def _fake_fetch(filename: str, p: dict[str, str], *_a: Any) -> str:
            zp = Path(p["raw_zips"]) / filename
            zp.parent.mkdir(parents=True, exist_ok=True)
            zp.write_bytes(b"zip")
            return str(zp)

        def _fake_extract(zip_path: str, _pat: list[str], exp: str, w: str, *, delete_zip: bool, log_writer: Any) -> None:
            del log_writer
            _populate_test_scenes(Path(w) / "Structured3D", with_views=True)
            Path(exp).parent.mkdir(parents=True, exist_ok=True)
            Path(exp).mkdir(exist_ok=True) if "annotation" not in exp else Path(exp).write_text("{}", encoding="utf-8")
            if delete_zip and os.path.exists(zip_path):
                os.remove(zip_path)

        orig_extract_7z = pipeline.extract_7z
        monkeypatch.setattr(pipeline, "fetch_zip", _fake_fetch)
        monkeypatch.setattr(pipeline, "extract_7z", _fake_extract)
        pipeline.ensure_eval_test_scenes(eval_args, paths, cfg, lw)
        assert pipeline.are_test_scenes_ready(paths["s3d_raw"])
        assert gcs[f"gs://scenealigner/s3d_eval/{pipeline.TEST_SCENES_TAR}"] != b"corrupt-test-tar"

        # 6. Corrupt recon shard in GCS is warned & removed without crashing eval
        shutil.rmtree(recon_dir)
        gcs[f"gs://scenealigner/{pipeline.RECON_CACHE_PREFIX}/recon_3250_3300.tar"] = b"bad-recon"
        pipeline.restore_eval_recon_cache(paths, "gs://scenealigner", lw)
        assert f"gs://scenealigner/{pipeline.RECON_CACHE_PREFIX}/recon_3250_3300.tar" not in gcs

        # 7. < 22 GiB free disk guard disables --recon_cache in run_eval_stage
        DiskUsage = namedtuple("usage", ["total", "used", "free"])
        monkeypatch.setattr(pipeline.shutil, "disk_usage", lambda _p: DiskUsage(100, 90, 10 * 1024**3))
        eval_cmds: list[list[str]] = []
        monkeypatch.setattr(pipeline, "run_live", lambda cmd, *_a, **_k: eval_cmds.append(cmd))
        eval_none = pipeline.parse_args(["--stages", "eval", "--sem-mode", "none", "--no-freeze-lora"])
        pipeline.run_eval_stage(eval_none, paths, cfg, lw)
        assert len(eval_cmds) == 1 and "--recon_cache" not in eval_cmds[0]

        # 8. extract_7z deletes corrupt zip before re-raising
        bad_zip = Path(paths["raw_zips"]) / "bad.zip"
        bad_zip.write_bytes(b"corrupt")
        with monkeypatch.context() as mp:
            mp.setattr(pipeline, "extract_7z", orig_extract_7z)
            mp.setattr(pipeline, "_find_7z", lambda required=True: "/usr/bin/7z")
            mp.setattr(
                pipeline.subprocess,
                "run",
                lambda cmd, **_k: (_ for _ in ()).throw(subprocess.CalledProcessError(2, cmd)),
            )
            with pytest.raises(subprocess.CalledProcessError):
                pipeline.extract_7z(str(bad_zip), [], str(tmp_path / "missing"), str(ws), delete_zip=False, log_writer=lw)
        assert not bad_zip.exists()

        # 9. write_final_results skips corrupt eval JSON without crashing
        (Path(paths["run_out"]) / "eval_s3d_none.json").write_text("{truncated", encoding="utf-8")
        summary = pipeline.write_final_results(eval_none, paths, cfg, ["none"], lw)
        assert summary["eval_metrics"] == {}
    finally:
        lw.close()


def test_gcs_multi_machine_manifest_and_run_restore_lifecycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gcs = _install_fake_gcs(monkeypatch)
    ws = tmp_path / "ws_multi"
    gcs_uri = "gs://scenealigner"

    # 1. Stale completed local ACTIVE_RUN is cleared so newer remote ACTIVE_RUN wins
    stale_args = pipeline.parse_args(["--stages", "train", "--chunks", "0", "--epochs", "5"])
    stale_id = "2026-11-10_00-00-00Z_stale"
    (ws / "runs" / stale_id).mkdir(parents=True)
    (ws / "runs" / "ACTIVE_RUN").write_text(f"{stale_id}\n", encoding="utf-8")
    (ws / "runs" / stale_id / "run_manifest.json").write_text(
        json.dumps({"run_id": stale_id, "status": "in_progress", "updated_at_utc": "2026-11-10T01:00:00Z"}),
        encoding="utf-8",
    )
    gcs[f"{gcs_uri}/runs/{stale_id}/run_manifest.json"] = json.dumps(
        {"run_id": stale_id, "status": "completed", "updated_at_utc": "2026-11-10T02:00:00Z"}
    ).encode("utf-8")

    target_args = pipeline.parse_args(["--stages", "eval", "--chunks", "0", "--sem-mode", "consensus", "--no-freeze-lora"])
    remote_id = "2026-11-15_01-00-00Z_remote"
    gcs[f"{gcs_uri}/runs/ACTIVE_RUN"] = f"{remote_id}\n".encode("utf-8")
    gcs[f"{gcs_uri}/runs/{remote_id}/run_manifest.json"] = json.dumps(
        {
            "run_id": remote_id,
            "status": "in_progress",
            "updated_at_utc": "2026-11-15T01:05:00Z",
            "config_fingerprint": pipeline.compute_run_fingerprint(target_args),
        }
    ).encode("utf-8")
    gcs[f"{gcs_uri}/runs/{remote_id}/semantic_adapter.pt"] = b"sem-adapter"
    gcs[f"{gcs_uri}/runs/{remote_id}/eval_progress_none_uniform_consensus.jsonl"] = b'{"scene": 3250}\n'

    picked_id, picked_resumed = pipeline.resolve_run_id(target_args, str(ws), gcs_uri)
    assert (picked_id, picked_resumed) == (remote_id, True)
    assert not (ws / "runs" / "ACTIVE_RUN").exists()

    # Explicit --run-id reflects remote in_progress status accurately
    explicit_args = pipeline.parse_args(["--stages", "eval", "--run-id", remote_id])
    assert pipeline.resolve_run_id(explicit_args, str(ws), gcs_uri) == (remote_id, True)
    brand_new_args = pipeline.parse_args(["--stages", "eval", "--run-id", "brand_new_run"])
    assert pipeline.resolve_run_id(brand_new_args, str(ws), gcs_uri) == ("brand_new_run", False)

    # 2. run_eval_stage restores run artifacts before _resolve_eval_modes without adapter_config.json or clobbering pipeline.log
    paths = pipeline.build_workspace_paths(str(ws), remote_id, "photo-gt")
    pipeline.ensure_workspace_dirs(paths, "photo-gt")
    _populate_test_scenes(Path(paths["s3d_raw"]), with_views=True)
    gcs[f"{gcs_uri}/runs/{remote_id}/pipeline.log"] = b"old remote log\n"
    Path(paths["log_path"]).write_text("active local header\n", encoding="utf-8")

    eval_cmds: list[list[str]] = []
    monkeypatch.setattr(pipeline, "run_live", lambda cmd, *_a, **_k: eval_cmds.append(cmd))
    DiskUsage = namedtuple("usage", ["total", "used", "free"])
    monkeypatch.setattr(pipeline.shutil, "disk_usage", lambda _p: DiskUsage(100, 50, 50 * 1024**3))
    lw = pipeline._CleanLogWriter(paths["log_path"], echo_to_stdout=False, progress_interval_sec=0.0)
    try:
        modes = pipeline.run_eval_stage(
            pipeline.parse_args(["--stages", "eval", "--sem-mode", "none", "--no-freeze-lora"]),
            paths,
            {"gcs_backup_uri": gcs_uri, "s3d_base_url": ""},
            lw,
        )
    finally:
        lw.close()
    assert modes == ["none", "uniform", "consensus"]
    assert (Path(paths["run_out"]) / "semantic_adapter.pt").read_bytes() == b"sem-adapter"
    assert (Path(paths["run_out"]) / "eval_progress_none_uniform_consensus.jsonl").exists()
    assert Path(paths["log_path"]).read_text(encoding="utf-8").startswith("active local header\n")

    # 3. init_session upfront restores persisted artifacts on a fresh Colab/VM workspace
    fresh_ws = tmp_path / "ws_fresh_colab"
    monkeypatch.setattr(pipeline, "are_test_scenes_ready", lambda _d: True)
    session = pipeline.init_session(
        stages=("eval",),
        chunks=(0,),
        freeze_lora=False,
        workspace=str(fresh_ws),
        gcs_uri=gcs_uri,
        verify_gcp_auth=False,
    )
    assert Path(session.paths["run_out"]).name == remote_id
    assert session.resumed is True
    assert (Path(session.paths["run_out"]) / "semantic_adapter.pt").read_bytes() == b"sem-adapter"


def test_timestamped_run_id_active_run_and_completion_lifecycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gcs = _install_fake_gcs(monkeypatch)
    ws = tmp_path / "ws_runs"
    gcs_uri = "gs://scenealigner"

    # Run 1 mints timestamped ID & tracks ACTIVE_RUN
    args1 = pipeline.parse_args(["--stages", "prep", "train", "eval", "--chunks", "0", "--epochs", "10"])
    run1, resumed1 = pipeline.resolve_run_id(args1, str(ws), gcs_uri)
    assert resumed1 is False
    assert run1.endswith("_s3d_photo-gt_consensus_c0_ep10")
    paths1 = pipeline.build_workspace_paths(str(ws), run1, "photo-gt")
    pipeline.ensure_workspace_dirs(paths1, "photo-gt")
    cfg1 = {"gcs_backup_uri": gcs_uri, "s3d_base_url": ""}
    pipeline.init_run_manifest(args1, paths1, cfg1, resumed=resumed1)
    assert (ws / "runs" / "ACTIVE_RUN").read_text(encoding="utf-8").strip() == run1
    assert f"{gcs_uri}/runs/ACTIVE_RUN" in gcs

    # Run 2 with same config (even after local runs/ wiped) auto-resumes run1 from GCS
    shutil.rmtree(ws / "runs")
    run2, resumed2 = pipeline.resolve_run_id(args1, str(ws), gcs_uri)
    assert (run2, resumed2) == (run1, True)

    # Run 3 with changed chunks mints a fresh ID
    args3 = pipeline.parse_args(["--stages", "prep", "train", "eval", "--chunks", "0", "1", "--epochs", "10"])
    run3, resumed3 = pipeline.resolve_run_id(args3, str(ws), gcs_uri)
    assert resumed3 is False
    assert run3 != run1
    assert run3.endswith("_s3d_photo-gt_consensus_c0-1_ep10")

    # Run 4 with --new-run ignores matching ACTIVE_RUN
    args4 = pipeline.parse_args(["--stages", "prep", "train", "eval", "--chunks", "0", "--epochs", "10", "--new-run"])
    run4, resumed4 = pipeline.resolve_run_id(args4, str(ws), gcs_uri)
    assert resumed4 is False
    assert run4.endswith("_s3d_photo-gt_consensus_c0_ep10")

    # Completion clears ACTIVE_RUN, updates INDEX.jsonl & LATEST_SUMMARY.txt
    paths4 = pipeline.build_workspace_paths(str(ws), run4, "photo-gt")
    pipeline.ensure_workspace_dirs(paths4, "photo-gt")
    pipeline.init_run_manifest(args4, paths4, cfg1, resumed=False)
    (Path(paths4["run_out"]) / "eval_s3d_consensus.json").write_text(
        json.dumps({"Pos@0.1m": 5.0, "Pos@0.5m": 45.0, "Pos@1.0m": 60.0, "Ang&Pos@(30,1m)": 58.0}),
        encoding="utf-8",
    )
    lw = pipeline._CleanLogWriter(paths4["log_path"], echo_to_stdout=False, progress_interval_sec=0.0)
    try:
        pipeline.write_final_results(args4, paths4, cfg1, ["consensus"], lw, resumed=False)
    finally:
        lw.close()

    assert not (ws / "runs" / "ACTIVE_RUN").exists()
    assert f"{gcs_uri}/runs/ACTIVE_RUN" not in gcs
    assert f"{gcs_uri}/runs/LATEST_SUMMARY.txt" in gcs
    assert (ws / "runs" / "INDEX.jsonl").exists()
    assert f"{gcs_uri}/runs/INDEX.jsonl" in gcs
    manifest4 = json.loads(Path(paths4["manifest_json"]).read_text(encoding="utf-8"))
    assert manifest4["status"] == "completed"

    # After completion, same config starts a brand-new run (resumed == False)
    _, resumed5 = pipeline.resolve_run_id(args1, str(ws), gcs_uri)
    assert resumed5 is False


def test_run_vm_end_to_end_pipeline_saves_persistent_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws = tmp_path / "fresh_outputs"
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.setattr(pipeline, "_verify_s3d_base_url", lambda _u: None)

    commands: list[list[str]] = []
    eval_invocations: list[tuple[Path, bool, list[str]]] = []

    def _fake_fetch(filename: str, paths: dict[str, str], _cfg: dict[str, str], _lw: Any) -> str:
        zp = Path(paths["raw_zips"]) / filename
        zp.parent.mkdir(parents=True, exist_ok=True)
        zp.write_bytes(b"zip")
        return str(zp)

    def _fake_extract(
        zip_path: str,
        _patterns: list[str],
        expected_path: str,
        workspace: str,
        *,
        delete_zip: bool,
        log_writer: Any,
    ) -> None:
        del log_writer
        exp = Path(expected_path)
        s3d_raw = Path(workspace) / "Structured3D"
        if exp.name == "annotation_3d.json":
            for s in range(0, 200):
                (s3d_raw / f"scene_{s:05d}").mkdir(parents=True, exist_ok=True)
                (s3d_raw / f"scene_{s:05d}" / "annotation_3d.json").write_text("{}", encoding="utf-8")
            _populate_test_scenes(s3d_raw, with_views=False)
        else:
            start = int(exp.name.split("_")[1])
            for s in range(start, min(start + 200, 3500)):
                (s3d_raw / f"scene_{s:05d}" / "2D_rendering").mkdir(parents=True, exist_ok=True)
        if delete_zip and os.path.exists(zip_path):
            os.remove(zip_path)

    def _fake_run_live(cmd: list[str], _lw: Any = None, **_kwargs: Any) -> None:
        commands.append(cmd)
        script_name = Path(cmd[1]).name
        if script_name == "prepare_s3d.py":
            out_dir = Path(cmd[cmd.index("--output") + 1])
            s0 = int(cmd[cmd.index("--scenes") + 1])
            s1 = int(cmd[cmd.index("--scenes") + 2])
            (out_dir / f"scenes_{s0}_{s1}.csv").write_text("scene_id\n", encoding="utf-8")
        elif script_name == "train.py":
            out_dir = Path(cmd[cmd.index("--output") + 1])
            total_epochs = int(cmd[cmd.index("--epochs") + 1])
            target_ep = (
                int(cmd[cmd.index("--stop_after_epoch") + 1])
                if "--stop_after_epoch" in cmd
                else total_epochs
            )
            step = 250 * target_ep
            (out_dir / "checkpoint.pt").write_bytes(b"ckpt")
            (out_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
            (out_dir / "semantic_adapter.pt").write_bytes(b"sem")
            (out_dir / "train_state.json").write_text(
                json.dumps(
                    {
                        "step": step,
                        "steps_per_epoch": 250,
                        "completed_epochs": target_ep,
                        "total_epochs": total_epochs,
                    }
                ),
                encoding="utf-8",
            )
            (out_dir / "log.csv").write_text(f"step,loss,feat,regr,geo,topo,sem\n{step},1.25,0.4,0.2,0.1,0.1,0.3\n")
        elif script_name == "evaluate_s3d.py":
            out_dir = Path(cmd[cmd.index("--output_dir") + 1])
            modes_idx = cmd.index("--sem_modes") + 1
            req_modes: list[str] = []
            for tok in cmd[modes_idx:]:
                if tok.startswith("--"):
                    break
                req_modes.append(tok)
            eval_modes: list[str] = []
            for idx, mode in enumerate(req_modes):
                out_file = out_dir / f"eval_s3d_{mode}.json"
                if "--resume" in cmd and out_file.exists():
                    continue
                eval_modes.append(mode)
                metrics = {
                    "Pos@0.1m": 4.0 + idx,
                    "Pos@0.5m": 40.0 + idx,
                    "Pos@1.0m": 55.0 + idx,
                    "Ang&Pos@(30,1m)": 52.0 + idx,
                }
                out_file.write_text(json.dumps(metrics), encoding="utf-8")
            eval_invocations.append((out_dir, "--max_scenes" in cmd, eval_modes))

    monkeypatch.setattr(pipeline, "fetch_zip", _fake_fetch)
    monkeypatch.setattr(pipeline, "extract_7z", _fake_extract)
    monkeypatch.setattr(pipeline, "ensure_floorplan_sem", lambda *_a, **_k: None)
    monkeypatch.setattr(pipeline, "run_live", _fake_run_live)

    rc = run_vm.main(
        [
            "--skip-setup",
            "--no-gcs",
            "--s3d-base-url",
            "https://s3d.example.org/README.txt",
            "--workspace",
            str(ws),
            "--run-id",
            "vm_test_run",
            "--chunks",
            "0",
            "--slice-scenes",
            "100",
            "--epochs",
            "2",
            "--stages",
            "prep",
            "train",
            "eval",
        ]
    )
    assert rc == 0

    prep_cmds = [c for c in commands if Path(c[1]).name == "prepare_s3d.py"]
    train_cmds = [c for c in commands if Path(c[1]).name == "train.py"]
    eval_cmds = [c for c in commands if Path(c[1]).name == "evaluate_s3d.py"]
    assert len(prep_cmds) == 2
    assert len(train_cmds) == 2 and all("--resume" in c for c in train_cmds)
    assert train_cmds[0][train_cmds[0].index("--stop_after_epoch") + 1] == "1"
    assert train_cmds[1][train_cmds[1].index("--stop_after_epoch") + 1] == "2"
    assert "--scene_ranges" in train_cmds[0] and "0:200" in train_cmds[0]
    assert len(eval_cmds) == 3 and all("--resume" in c for c in eval_cmds)
    assert eval_invocations[0] == (ws / "runs" / "vm_test_run" / "epochs" / "epoch_01", True, ["none", "uniform", "consensus"])
    assert eval_invocations[1] == (ws / "runs" / "vm_test_run" / "epochs" / "epoch_02", True, ["uniform", "consensus"])
    assert eval_invocations[2] == (ws / "runs" / "vm_test_run", False, ["none", "uniform", "consensus"])

    assert not (ws / "Structured3D" / "scene_00000" / "2D_rendering").exists()

    run_dir = ws / "runs" / "vm_test_run"
    summary_json = run_dir / "results_summary.json"
    summary_txt = run_dir / "results_summary.txt"
    latest_json = ws / "latest_results.json"
    latest_txt = ws / "latest_results.txt"
    log_path = run_dir / "pipeline.log"
    manifest_json = run_dir / "run_manifest.json"
    index_jsonl = ws / "runs" / "INDEX.jsonl"

    for artifact in (summary_json, summary_txt, latest_json, latest_txt, log_path, manifest_json, index_jsonl):
        assert artifact.exists(), f"Missing expected result artifact: {artifact}"

    assert json.loads(manifest_json.read_text(encoding="utf-8"))["status"] == "completed"
    assert summary_json.read_text(encoding="utf-8") == latest_json.read_text(encoding="utf-8")
    assert summary_txt.read_text(encoding="utf-8") == latest_txt.read_text(encoding="utf-8")

    data = json.loads(summary_json.read_text(encoding="utf-8"))
    assert data["prepared_scene_ranges"] == [[0, 100], [100, 200]]
    assert data["training"]["last_logged_step"] == 500
    assert data["eval_metrics"]["consensus"]["Pos@0.5m"] == 42.0
    assert len(data["epoch_evals"]) == 2
    assert [r["epoch"] for r in data["epoch_evals"]] == [1, 2]
    assert data["epoch_evals"][0]["max_scenes"] == 50
    assert (run_dir / "epochs" / "epoch_02" / "eval_s3d_none.json").read_text(encoding="utf-8") == (
        run_dir / "epochs" / "epoch_01" / "eval_s3d_none.json"
    ).read_text(encoding="utf-8")
    assert "42.00" in summary_txt.read_text(encoding="utf-8")
    assert "Between-Epoch Subset Evals (50 scenes):" in summary_txt.read_text(encoding="utf-8")


def test_floorplan_sem_healing_triggers_url_probe_preflight_and_photo_none_guards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gcs = _install_fake_gcs(monkeypatch)
    ws = tmp_path / "ws_heal"
    paths = pipeline.build_workspace_paths(str(ws), "heal_check", "photo-gt")
    pipeline.ensure_workspace_dirs(paths, "photo-gt")

    # 1. Local CSV missing floorplan_sem files (Request #6 VM state)
    (Path(paths["s3d_data"]) / "scenes_0_200.csv").write_text(
        "scene_id,floorplan,floorplan_sem\nscene_00000,floorplan/scene_00000.png,floorplan_sem/scene_00000\n",
        encoding="utf-8",
    )

    prep_args = pipeline.parse_args(["--stages", "prep", "--chunks", "0"])
    train_args = pipeline.parse_args(["--stages", "train", "--chunks", "0"])
    smoke_args = pipeline.parse_args(["--stages", "smoke"])

    assert pipeline.needs_external_s3d_download(prep_args, paths, "gs://scenealigner", "photo-gt") is True
    assert pipeline.needs_external_s3d_download(train_args, paths, "gs://scenealigner", "photo-gt") is True

    broken_secret = "https://stale-vm-broken.example.org/s3d"
    pipeline._upsert_env_key(paths["env_file"], "S3D_BASE_URL", broken_secret)
    monkeypatch.delenv("S3D_BASE_URL", raising=False)
    monkeypatch.delenv("GCS_BACKUP_URI", raising=False)
    monkeypatch.delenv("_SCENEALIGNER_BOOTSTRAPPED", raising=False)
    monkeypatch.setattr(pipeline.sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(pipeline.sys.stdout, "isatty", lambda: False)
    monkeypatch.setattr(
        pipeline.urllib.request,
        "urlopen",
        lambda req, timeout=8.0: (_ for _ in ()).throw(
            urllib.error.HTTPError(req.full_url, 404, "Not Found", Message(), None)
        ),
    )

    with pytest.raises(RuntimeError) as exc_info:
        pipeline.resolve_config(train_args, paths, verify_gcp_auth=True)
    err_msg = str(exc_info.value)
    assert "<S3D_BASE_URL>" in err_msg
    assert broken_secret not in err_msg
    assert "gs://scenealigner/config/.env" not in gcs

    # Smoke stage does not probe upfront in resolve_config, and must never upload an unverified broken .env
    monkeypatch.delenv("S3D_BASE_URL", raising=False)
    smoke_cfg = pipeline.resolve_config(smoke_args, paths, verify_gcp_auth=True)
    assert smoke_cfg["s3d_base_url"] == broken_secret
    assert "gs://scenealigner/config/.env" not in gcs

    empty_url_cfg = {"gcs_backup_uri": "gs://scenealigner", "s3d_base_url": ""}
    with pytest.raises(RuntimeError, match="Structured3D_annotation_3d.zip"):
        pipeline.preflight_data_sources(prep_args, empty_url_cfg, paths)
    with pytest.raises(RuntimeError, match="Structured3D_annotation_3d.zip"):
        pipeline.preflight_data_sources(train_args, empty_url_cfg, paths)

    # 2. Unmigrated legacy flat GCS tars on an otherwise empty local workspace
    empty_ws = tmp_path / "ws_legacy_gcs"
    empty_paths = pipeline.build_workspace_paths(str(empty_ws), "legacy_gcs", "photo-gt")
    pipeline.ensure_workspace_dirs(empty_paths, "photo-gt")
    for s0 in (0, 50, 100, 150):
        gcs[f"gs://scenealigner/s3d_data/scenes_{s0}_{s0 + 50}.tar"] = b"legacy-tar"

    assert pipeline.needs_external_s3d_download(prep_args, empty_paths, "gs://scenealigner", "photo-gt") is True
    gcs["gs://scenealigner/raw_zips/Structured3D_annotation_3d.zip"] = b"anno-zip"
    assert pipeline.needs_external_s3d_download(prep_args, empty_paths, "gs://scenealigner", "photo-gt") is False

    # 3. --photo-sem-source none guards
    bad_none_train = pipeline.parse_args(["--stages", "train", "--photo-sem-source", "none", "--sem-mode", "consensus"])
    with pytest.raises(ValueError, match="photo-sem-source none"):
        pipeline.preflight_data_sources(bad_none_train, empty_url_cfg, empty_paths)

    eval_none_args = pipeline.parse_args(["--photo-sem-source", "none"])
    assert pipeline._resolve_eval_modes(eval_none_args, str(tmp_path)) == ["none"]


def test_evaluate_s3d_max_scenes_and_resume_subset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import evaluate_s3d
    import numpy as np
    import torch
    from PIL import Image

    seen_scenes: list[int] = []
    s3d_root = tmp_path / "Structured3D"
    out_dir = tmp_path / "out_subset"
    out_dir.mkdir(parents=True)

    monkeypatch.setattr(evaluate_s3d.s3d, "TEST_SCENES", [3250, 3251, 3252])
    monkeypatch.setattr(evaluate_s3d, "load_models", lambda **_k: (object(), object()))
    monkeypatch.setattr(evaluate_s3d, "load_model", lambda *_a, **_k: torch.nn.Linear(2, 2))
    monkeypatch.setattr(
        evaluate_s3d.s3d,
        "load_floorplan",
        lambda d: (seen_scenes.append(int(Path(d).name.split("_")[1])), (Image.new("RGB", (16, 16)), ((0.0, 0.0), 1.0)))[1],
    )
    monkeypatch.setattr(
        evaluate_s3d.s3d,
        "list_views",
        lambda d: [str(Path(d) / "2D_rendering/0/perspective/full/0")],
    )
    monkeypatch.setattr(
        evaluate_s3d,
        "reconstruct",
        lambda *_a, **_k: {
            "points_cam0": np.zeros((1, 4, 4, 3), dtype=np.float32),
            "conf": np.ones((1, 4, 4), dtype=np.float32),
            "extrinsics": np.eye(4, dtype=np.float32)[None, ...],
            "est_gravity_cam": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        },
    )
    monkeypatch.setattr(evaluate_s3d, "align_to_floorplan", lambda *_a, **_k: {"proj_params": object(), "M": np.eye(2, 3)})
    monkeypatch.setattr(evaluate_s3d, "pose_errors", lambda *_a, **_k: (0.05, 5.0))

    # Out-of-subset records in progress JSONL should be ignored when --max_scenes 1 is passed
    (out_dir / "eval_progress_none.jsonl").write_text(
        json.dumps({"scene": 3251, "modes": {"none": {"t_errs": [99.0], "r_errs": [99.0], "failed": 0}}}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_s3d.py",
            "--s3d_root",
            str(s3d_root),
            "--checkpoint_dir",
            str(out_dir),
            "--sem_modes",
            "none",
            "--output_dir",
            str(out_dir),
            "--max_scenes",
            "1",
            "--resume",
        ],
    )
    evaluate_s3d.main()
    assert seen_scenes == [3250]
    metrics = json.loads((out_dir / "eval_s3d_none.json").read_text(encoding="utf-8"))
    assert metrics["Pos@0.1m"] == 100.0

    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluate_s3d.py", "--s3d_root", str(s3d_root), "--max_scenes", "0"],
    )
    with pytest.raises(ValueError, match="--max_scenes"):
        evaluate_s3d.main()


def test_between_epoch_eval_resume_finished_epoch_and_gcs_nested_restore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 1. Inter-eval flags must NOT change run fingerprint or slug so active runs resume cleanly
    a0 = pipeline.parse_args(["--stages", "train", "--eval-every-epochs", "0", "--inter-eval-scenes", "0"])
    a1 = pipeline.parse_args(["--stages", "train", "--eval-every-epochs", "1", "--inter-eval-scenes", "50"])
    assert pipeline.compute_run_fingerprint(a0) == pipeline.compute_run_fingerprint(a1)
    assert pipeline.build_run_slug(a0) == pipeline.build_run_slug(a1)

    # 2. restore_run_from_gcs preserves nested epochs/... paths without touching top-level eval_s3d_*.json
    gcs = _install_fake_gcs(monkeypatch)
    ws = tmp_path / "ws_restore"
    run_id = "resume_nested"
    paths = pipeline.build_workspace_paths(str(ws), run_id, "photo-gt")
    pipeline.ensure_workspace_dirs(paths, "photo-gt")
    remote_root = f"gs://scenealigner/runs/{run_id}"
    gcs[f"{remote_root}/checkpoint.pt"] = b"ckpt-ep1"
    gcs[f"{remote_root}/epochs/epoch_01/eval_s3d_none.json"] = json.dumps(
        {"Pos@0.1m": 1.0, "Pos@0.5m": 10.0, "Pos@1.0m": 20.0, "Ang&Pos@(30,1m)": 15.0}
    ).encode("utf-8")
    assert pipeline.restore_run_from_gcs(paths, "gs://scenealigner", allow_log_overwrite=False) is True
    assert (Path(paths["run_out"]) / "epochs" / "epoch_01" / "eval_s3d_none.json").exists()
    assert not (Path(paths["run_out"]) / "eval_s3d_none.json").exists()

    # 3. Resuming from an already-finished Epoch 1 checkpoint without train_state.json evaluates Epoch 1 first
    ws2 = tmp_path / "ws_resume_ep1"
    run_id2 = "resume_ep1"
    p2 = pipeline.build_workspace_paths(str(ws2), run_id2, "photo-gt")
    pipeline.ensure_workspace_dirs(p2, "photo-gt")
    (Path(p2["s3d_data"]) / "scenes_0_200.csv").write_text("scene_id\n", encoding="utf-8")
    (Path(p2["s3d_data"]) / "floorplan_sem").mkdir(parents=True, exist_ok=True)
    _populate_test_scenes(Path(p2["s3d_raw"]), with_views=True)
    # Seed legacy finished Epoch 1 checkpoint (no train_state.json, no epochs/epoch_01/)
    run_out2 = Path(p2["run_out"])
    run_out2.mkdir(parents=True, exist_ok=True)
    (run_out2 / "checkpoint.pt").write_bytes(b"ckpt-ep1")
    (run_out2 / "adapter_config.json").write_text("{}", encoding="utf-8")
    (run_out2 / "semantic_adapter.pt").write_bytes(b"sem1")

    events: list[tuple[str, int, int]] = []
    monkeypatch.setattr(pipeline, "restore_s3d_tars", lambda *_a, **_k: None)
    monkeypatch.setattr(pipeline, "ensure_floorplan_sem", lambda *_a, **_k: None)

    def _fake_live(cmd: list[str], _lw: Any = None, **_kwargs: Any) -> None:
        script = Path(cmd[1]).name
        if script == "train.py":
            stop_ep = int(cmd[cmd.index("--stop_after_epoch") + 1])
            out_dir = Path(cmd[cmd.index("--output") + 1])
            cur_ep = 1
            if (out_dir / "train_state.json").exists():
                cur_ep = stop_ep
            step = 100 * cur_ep
            events.append(("train", stop_ep, cur_ep))
            (out_dir / "train_state.json").write_text(
                json.dumps({"step": step, "steps_per_epoch": 100, "completed_epochs": cur_ep, "total_epochs": 2}),
                encoding="utf-8",
            )
            (out_dir / "semantic_adapter.pt").write_bytes(f"sem{cur_ep}".encode("utf-8"))
        elif script == "evaluate_s3d.py":
            out_dir = Path(cmd[cmd.index("--output_dir") + 1])
            ep_num = int(out_dir.name.split("_")[1])
            events.append(("eval", ep_num, int(cmd[cmd.index("--max_scenes") + 1])))
            for m in ("none", "uniform", "consensus"):
                f = out_dir / f"eval_s3d_{m}.json"
                if not f.exists():
                    f.write_text(
                        json.dumps({"Pos@0.1m": 5.0 * ep_num, "Pos@0.5m": 40.0 + ep_num, "Pos@1.0m": 60.0, "Ang&Pos@(30,1m)": 55.0}),
                        encoding="utf-8",
                    )

    monkeypatch.setattr(pipeline, "run_live", _fake_live)
    args = pipeline.parse_args(["--stages", "train", "--chunks", "0", "--epochs", "2", "--workspace", str(ws2), "--no-gcs"])
    cfg = {"gcs_backup_uri": "", "s3d_base_url": ""}
    lw = pipeline._CleanLogWriter(p2["log_path"], echo_to_stdout=False)
    try:
        pipeline.run_train_stage(args, p2, cfg, lw)
    finally:
        lw.close()

    assert events == [("train", 1, 1), ("eval", 1, 50), ("train", 2, 2), ("eval", 2, 50)]
    eval_rows = pipeline._read_epoch_evals(p2["run_out"])
    assert [r["epoch"] for r in eval_rows] == [1, 2]
    # Re-running run_train_stage when all epochs + inter-evals are complete is a no-op
    events.clear()
    lw2 = pipeline._CleanLogWriter(p2["log_path"], echo_to_stdout=False)
    try:
        pipeline.run_train_stage(args, p2, cfg, lw2)
    finally:
        lw2.close()
    assert events == []


def test_interactive_stdout_and_run_live_colab_stream_without_duplicate_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: list[str] = []

    class OutStream:
        __module__ = "ipykernel.iostream"

        def isatty(self) -> bool:
            return False

        def write(self, data: str) -> int:
            captured.append(data)
            return len(data)

        def flush(self) -> None:
            pass

    monkeypatch.setattr(pipeline.sys, "stdout", OutStream())
    assert pipeline._is_interactive_stdout() is True

    args = pipeline.parse_args(["--stages", "train", "--workspace", str(tmp_path), "--no-gcs"])
    paths = pipeline.build_workspace_paths(str(tmp_path), "colab_stream", "photo-gt")
    session = pipeline.PipelineSession(args, paths, {"gcs_backup_uri": "", "s3d_base_url": ""}, resumed=False)
    lw = session._open_log()
    assert lw.echo_to_stdout is False

    code = (
        "import sys; "
        "sys.stdout.write('\\rstep 1/2\\x1b[2K\\x1b[0Gstep 2/2\\r'); "
        "sys.stdout.flush()"
    )
    try:
        pipeline.run_live([sys.executable, "-c", code], lw)
    finally:
        lw.close()

    joined = "".join(captured)
    assert "\r" in joined
    assert "step 2/2" in joined
    assert "\x1b" not in joined
    assert joined.endswith("\n")
    assert "[progress]" not in joined
    log_text = Path(paths["log_path"]).read_text(encoding="utf-8")
    assert "step 2/2" in log_text

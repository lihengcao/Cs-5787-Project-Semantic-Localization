#!/usr/bin/env python3
"""Persistent-VM CLI entrypoint for SceneAligner (Structured3D)."""
from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
from typing import Any

SCRIPT_PATH = os.path.abspath(__file__)
RUNNERS_DIR = os.path.dirname(SCRIPT_PATH)
REPO_DIR = os.path.dirname(RUNNERS_DIR)
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from runners import pipeline


def _resolve_eval_checkpoint_dir(raw_val: str | None, orig_cwd: str) -> str | None:
    if not raw_val:
        return None
    expanded = os.path.expanduser(raw_val)
    if os.path.isabs(expanded):
        return os.path.abspath(expanded)
    joined = os.path.abspath(os.path.join(orig_cwd, expanded))
    if raw_val.startswith((".", "~")) or os.path.exists(joined):
        return joined
    return raw_val


def _rewrite_cli_args(
    orig_argv: list[str],
    *,
    resolved_workspace: str,
    run_id: str,
    resolved_eval_ckpt: str | None = None,
    strip_detach: bool = False,
) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(orig_argv):
        tok = orig_argv[i]
        if tok in ("--workspace", "--run-id", "--run-name"):
            i += 2
            continue
        if tok.startswith(("--workspace=", "--run-id=", "--run-name=")):
            i += 1
            continue
        if tok == "--new-run":
            i += 1
            continue
        if strip_detach and tok == "--detach":
            i += 1
            continue
        if resolved_eval_ckpt is not None:
            if tok == "--eval-checkpoint-dir":
                i += 2
                continue
            if tok.startswith("--eval-checkpoint-dir="):
                i += 1
                continue
        out.append(tok)
        i += 1

    out.extend(["--workspace", resolved_workspace, "--run-id", run_id])
    if resolved_eval_ckpt is not None:
        out.extend(["--eval-checkpoint-dir", resolved_eval_ckpt])
    return out


def ensure_bootstrapped(
    args: argparse.Namespace,
    orig_argv: list[str],
    resolved_workspace: str,
    run_id: str,
    *,
    resolved_eval_ckpt: str | None = None,
    resumed: bool = False,
) -> None:
    if args.skip_setup:
        return

    venv_py = os.path.join(REPO_DIR, ".venv", "bin", "python")
    marker = os.path.join(REPO_DIR, ".venv", ".scenealigner_ready")
    in_venv = os.path.exists(venv_py) and os.path.realpath(sys.executable) == os.path.realpath(venv_py)
    if in_venv and os.environ.get("_SCENEALIGNER_BOOTSTRAPPED") == "1":
        return

    expected_hash = pipeline._requirements_hash(REPO_DIR)
    ready = False
    if (
        os.path.exists(venv_py)
        and os.path.isfile(marker)
        and bool(pipeline._find_7z(required=False))
        and bool(shutil.which("aria2c"))
    ):
        try:
            with open(marker, encoding="utf-8") as f:
                ready = f.read().strip() == expected_hash
        except OSError:
            ready = False

    if not ready:
        setup_sh = os.path.join(REPO_DIR, "runners", "setup_vm.sh")
        sys.stdout.write("[run_vm] Running runners/setup_vm.sh to prepare .venv and system tools...\n")
        sys.stdout.flush()
        subprocess.run(["bash", setup_sh], cwd=REPO_DIR, check=True)

    if os.path.exists(venv_py) and (not in_venv or not ready):
        os.environ["_SCENEALIGNER_BOOTSTRAPPED"] = "1"
        os.environ["_SCENEALIGNER_RUN_RESUMED"] = "1" if resumed else "0"
        forwarded = _rewrite_cli_args(
            orig_argv,
            resolved_workspace=resolved_workspace,
            run_id=run_id,
            resolved_eval_ckpt=resolved_eval_ckpt,
        )
        os.execv(venv_py, [venv_py, SCRIPT_PATH, *forwarded])


def maybe_detach(
    args: argparse.Namespace,
    orig_argv: list[str],
    paths: dict[str, str],
    gcs_uri: str,
    run_id: str,
    *,
    resolved_eval_ckpt: str | None = None,
    resumed: bool = False,
) -> bool:
    if not args.detach or os.environ.get("_SCENEALIGNER_DETACHED") == "1":
        return False

    del gcs_uri  # config already resolved into os.environ / .env
    log_path = paths["log_path"]
    pid_path = paths["pid_path"]
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    child_env = os.environ.copy()
    child_env["_SCENEALIGNER_DETACHED"] = "1"
    child_env["_SCENEALIGNER_BOOTSTRAPPED"] = "1"
    child_env["_SCENEALIGNER_RUN_RESUMED"] = "1" if resumed else "0"

    child_argv = _rewrite_cli_args(
        orig_argv,
        resolved_workspace=paths["workspace"],
        run_id=run_id,
        resolved_eval_ckpt=resolved_eval_ckpt,
        strip_detach=True,
    )
    child_cmd = [sys.executable, SCRIPT_PATH, *child_argv]

    with open(log_path, "a", encoding="utf-8") as log_fp:
        proc = subprocess.Popen(
            child_cmd,
            cwd=REPO_DIR,
            env=child_env,
            stdin=subprocess.DEVNULL,
            stdout=log_fp,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    pipeline._atomic_write_text(pid_path, f"{proc.pid}\n")
    sys.stdout.write(
        f"[run_vm] Started detached background process PID {proc.pid} for run {run_id}.\n"
        f"[run_vm] Log: tail -f {log_path}\n"
    )
    sys.stdout.flush()
    return True


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    orig_cwd = os.getcwd()
    args = pipeline.parse_args(raw_argv)

    resolved_ws = (
        os.path.abspath(os.path.expanduser(args.workspace))
        if args.workspace
        else os.path.join(REPO_DIR, "outputs")
    )
    args.eval_checkpoint_dir = _resolve_eval_checkpoint_dir(args.eval_checkpoint_dir, orig_cwd)

    os.chdir(REPO_DIR)

    prep_profile = pipeline.compute_prep_profile(
        args.photo_sem_source, args.include_rooms, args.structural_photo_semantics
    )
    probe_paths = pipeline.build_workspace_paths(resolved_ws, "_init", prep_profile)
    os.makedirs(probe_paths["workspace"], exist_ok=True)
    os.makedirs(probe_paths["s3d_data_root"], exist_ok=True)
    pipeline.migrate_local_legacy_s3d_data(probe_paths, prep_profile)

    cfg = pipeline.resolve_config(
        args,
        probe_paths,
        verify_gcp_auth=(os.environ.get("_SCENEALIGNER_BOOTSTRAPPED") != "1"),
    )
    run_id, resumed = pipeline.resolve_run_id(args, resolved_ws, cfg["gcs_backup_uri"])
    paths = pipeline.build_workspace_paths(resolved_ws, run_id, prep_profile)
    pipeline.ensure_workspace_dirs(paths, prep_profile)
    if (resumed or bool(getattr(args, "run_id", None))) and cfg["gcs_backup_uri"]:
        pipeline.restore_run_from_gcs(paths, cfg["gcs_backup_uri"], allow_log_overwrite=True)
    pipeline.preflight_data_sources(args, cfg, paths)
    manifest = pipeline.init_run_manifest(args, paths, cfg, resumed=resumed)
    is_resumed = bool(manifest.get("resumed")) or (os.environ.get("_SCENEALIGNER_RUN_RESUMED") == "1")

    ensure_bootstrapped(
        args,
        raw_argv,
        paths["workspace"],
        run_id,
        resolved_eval_ckpt=args.eval_checkpoint_dir,
        resumed=is_resumed,
    )
    if maybe_detach(
        args,
        raw_argv,
        paths,
        cfg["gcs_backup_uri"],
        run_id,
        resolved_eval_ckpt=args.eval_checkpoint_dir,
        resumed=is_resumed,
    ):
        return 0

    log_writer = pipeline._CleanLogWriter(
        paths["log_path"],
        redact=cfg["s3d_base_url"],
        echo_to_stdout=not pipeline._is_interactive_stdout(),
    )
    prev_sigterm = signal.getsignal(signal.SIGTERM)

    def _handle_sigterm(signum: int, frame: Any) -> None:
        del frame
        raise KeyboardInterrupt(f"Received signal {signum}")

    signal.signal(signal.SIGTERM, _handle_sigterm)
    eval_modes: list[str] | None = None
    try:
        pipeline.log_system_banner(paths, log_writer, runner_name="runners/run_vm.py")
        action = "Resuming" if is_resumed else "Starting"
        log_writer.log_line(
            f"{action} run {run_id} (profile={prep_profile}, photo_sem={args.photo_sem_source}, floorplan_sem=s3d_annos)"
        )
        for stage in args.stages:
            if stage == "smoke":
                pipeline.run_smoke_stage(paths, log_writer)
            elif stage == "prep":
                pipeline.run_prep_stage(args, paths, cfg, log_writer)
            elif stage == "train":
                pipeline.run_train_stage(args, paths, cfg, log_writer)
            elif stage == "eval":
                eval_modes = pipeline.run_eval_stage(args, paths, cfg, log_writer)
            elif stage == "backup":
                pipeline.run_backup_stage(args, paths, cfg, log_writer)
        pipeline.write_final_results(args, paths, cfg, eval_modes, log_writer, resumed=is_resumed)
        return 0
    except KeyboardInterrupt:
        log_writer.log_line(f"Interrupted during run {run_id}; progress preserved for auto-resume.")
        return 130
    finally:
        signal.signal(signal.SIGTERM, prev_sigterm)
        log_writer.close()


if __name__ == "__main__":
    raise SystemExit(main())

"""Evaluation on the Structured3D test scenes: single-view camera localization (Table 3).

    python evaluate_s3d.py --s3d_root <Structured3D>
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import zipfile
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from scenealigner import structured3d as s3d
from scenealigner.evaluation import compute_pose_error_analytic, get_camera_center_on_density_px
from scenealigner.inference import (S3D_MODEL_REPO, AlignmentFailureError, PoseAlignment, SemanticAdapter,
                                   SemanticMapInput, SemMode, align_to_floorplan, load_model, load_semantic_adapter,
                                   reconstruct_projection_geometry)
from scenealigner.preprocess import ReconstructionDict, load_models, reconstruct
from scenealigner.semantics import DEFAULT_CLIP_MODEL, ClipTextEncoder


def pose_errors(
    recon: ReconstructionDict,
    alignment: PoseAlignment,
    plan_params: s3d.PlanParams,
    view_dir: str,
) -> tuple[float, float]:
    """Positional (m) and angular (degrees) errors of the camera of a view aligned to its floorplan."""
    eye, forward = s3d.load_camera(view_dir)[:2]
    center_px = get_camera_center_on_density_px(recon["extrinsics"], alignment["proj_params"])[0]
    t_err, r_err = compute_pose_error_analytic(alignment["M"], recon["extrinsics"][0], alignment["proj_params"], center_px,
                                               s3d.world_to_plan(eye, plan_params), forward[:2] * (1, -1))
    return float(t_err * s3d.MM_PER_PIXEL / plan_params[1] / 1000), r_err


def compute_metrics(t_errs: list[float], r_errs: list[float]) -> dict[str, float]:
    """Table 3: recalls (%) over the views."""
    t_errs_arr, r_errs_arr = np.array(t_errs), np.array(r_errs)
    metrics = {f"Pos@{thr}m": np.mean(t_errs_arr < thr) for thr in (0.1, 0.5, 1.0)}
    metrics["Ang&Pos@(30,1m)"] = np.mean((t_errs_arr < 1.0) & (r_errs_arr < 30.0))
    return {k: float(v) * 100 for k, v in metrics.items()}


_METRICS_KEYS = ("Pos@0.1m", "Pos@0.5m", "Pos@1.0m", "Ang&Pos@(30,1m)")


def _is_valid_metrics_json(path: str) -> bool:
    """Return True if path holds a finite numeric eval metrics dict with all expected keys."""
    if not os.path.isfile(path):
        return False
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError, ValueError):
        return False
    if not isinstance(data, dict):
        return False
    return all(
        isinstance(data.get(k), (int, float))
        and not isinstance(data.get(k), bool)
        and bool(np.isfinite(float(data[k])))
        for k in _METRICS_KEYS
    )


def _sync_eval_dir(local_dir: str, gcs_uri: str) -> None:
    """Best-effort rsync of eval progress and output JSONs to GCS."""
    target = gcs_uri.rstrip("/")
    res = subprocess.run(
        ["gcloud", "storage", "rsync", "--recursive", "--quiet", local_dir, target],
        capture_output=True,
        text=True,
        check=False,
    )
    if res.returncode != 0:
        err = res.stderr.strip() or f"exit code {res.returncode}"
        getattr(tqdm, "write", print)(f"[warn] GCS eval sync failed ({err})")


def _load_eval_progress(
    progress_path: str, modes: Sequence[SemMode], *, allowed_scenes: set[int] | None = None
) -> tuple[set[int], dict[SemMode, dict[str, Any]]]:
    """Load per-scene eval accumulators from JSONL, truncating any incomplete trailing line."""
    done_scenes: set[int] = set()
    results_by_mode: dict[SemMode, dict[str, Any]] = {
        m: {"t_errs": [], "r_errs": [], "failed": 0} for m in modes
    }
    if not os.path.exists(progress_path):
        return done_scenes, results_by_mode
    valid_offset = 0
    with open(progress_path, "rb") as f:
        while raw := f.readline():
            if not raw.endswith(b"\n"):
                break
            try:
                rec = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                break
            if not isinstance(rec, dict) or "scene" not in rec or "modes" not in rec:
                break
            rec_modes = rec["modes"]
            if not isinstance(rec_modes, dict) or any(m not in rec_modes for m in modes):
                break
            scene_id = int(rec["scene"])
            if allowed_scenes is None or scene_id in allowed_scenes:
                done_scenes.add(scene_id)
                for m in modes:
                    m_rec = rec_modes[m]
                    results_by_mode[m]["t_errs"].extend(m_rec["t_errs"])
                    results_by_mode[m]["r_errs"].extend(m_rec["r_errs"])
                    results_by_mode[m]["failed"] += int(m_rec["failed"])
            valid_offset = f.tell()
    if valid_offset < os.path.getsize(progress_path):
        with open(progress_path, "r+b") as f:
            f.truncate(valid_offset)
    return done_scenes, results_by_mode


_REQUIRED_RECON_KEYS = ("points_cam0", "conf", "extrinsics", "est_gravity_cam")


def _view_recon_cache_path(recon_cache_dir: str, scene: int, scene_dir: str, view_dir: str) -> str:
    """Build a safe per-view cache file path inside recon_cache_dir/scene_XXXXX/."""
    rel = os.path.normpath(os.path.relpath(view_dir, scene_dir))
    if rel == "." or rel.startswith("..") or os.path.isabs(rel):
        raise ValueError(f"view_dir {view_dir!r} must reside inside scene_dir {scene_dir!r}")
    safe_name = rel.replace(os.sep, "__") + ".npz"
    return os.path.join(recon_cache_dir, f"scene_{scene:05d}", safe_name)


def _load_cached_recon(cache_path: str) -> ReconstructionDict | None:
    """Load lossless float32 reconstruction from .npz cache, or remove and return None if invalid."""
    if not os.path.exists(cache_path):
        return None
    try:
        with np.load(cache_path, allow_pickle=False) as data:
            arrays: dict[str, Any] = {}
            for key in _REQUIRED_RECON_KEYS:
                if key not in data.files:
                    raise KeyError(f"Missing {key!r} in cached recon {cache_path}")
                arr = data[key]
                if arr.dtype != np.float32:
                    raise ValueError(f"Expected float32 for {key!r} in {cache_path}, got {arr.dtype}")
                arrays[key] = arr
            image_sizes = data["image_sizes"] if "image_sizes" in data.files else np.empty((0, 2), dtype=np.int64)
        return {
            "points_cam0": arrays["points_cam0"],
            "conf": arrays["conf"],
            "extrinsics": arrays["extrinsics"],
            "est_gravity_cam": arrays["est_gravity_cam"],
            "preprocessed_images": np.empty((0, 0, 0, 3), dtype=np.uint8),
            "image_sizes": image_sizes,
        }
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile) as err:
        getattr(tqdm, "write", print)(f"[warn] Ignoring invalid recon cache {cache_path} ({err})")
        try:
            os.remove(cache_path)
        except OSError:
            pass
        return None


def _save_cached_recon(cache_path: str, recon: ReconstructionDict) -> None:
    """Atomically persist lossless float32 reconstruction arrays to cache_path."""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    tmp_cache = f"{cache_path}.tmp.{os.getpid()}"
    try:
        with open(tmp_cache, "wb") as f:
            np.savez(
                f,
                points_cam0=np.asarray(recon["points_cam0"], dtype=np.float32),
                conf=np.asarray(recon["conf"], dtype=np.float32),
                extrinsics=np.asarray(recon["extrinsics"], dtype=np.float32),
                est_gravity_cam=np.asarray(recon["est_gravity_cam"], dtype=np.float32),
                image_sizes=np.asarray(recon.get("image_sizes", np.empty((0, 2), dtype=np.int64))),
            )
        os.replace(tmp_cache, cache_path)
    except BaseException:
        if os.path.exists(tmp_cache):
            try:
                os.remove(tmp_cache)
            except OSError:
                pass
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--s3d_root", required=True, help="Structured3D dataset: the directory of the scene_* folders")
    parser.add_argument("--checkpoint_dir", default=S3D_MODEL_REPO, help="LoRA layers")
    parser.add_argument("--dinov3_weights", default=None, help="Meta's DINOv3 ViT-B/16 weights")
    parser.add_argument("--output", default="outputs/s3d_results.json")
    parser.add_argument("--output_dir", default=None, help="directory for eval_s3d_<mode>.json outputs and resume state")
    parser.add_argument("--sem_mode", choices=["none", "uniform", "consensus"], default="none")
    parser.add_argument(
        "--sem_modes",
        nargs="+",
        choices=["none", "uniform", "consensus"],
        default=None,
        help="evaluate multiple semantic modes in a single pass sharing Pi3 reconstructions",
    )
    parser.add_argument("--photo_sem_source", choices=["gt", "none"], default="gt")
    parser.add_argument(
        "--semantic_adapter",
        default=None,
        help="path or directory for semantic_adapter.pt (defaults to --checkpoint_dir when --sem_mode != none)",
    )
    parser.add_argument(
        "--lambda_sem",
        type=float,
        default=None,
        help="semantic weight at inference (defaults to checkpoint metadata or 0.5)",
    )
    parser.add_argument("--include_rooms", action="store_true", help="include room-type semantics on the floorplan")
    parser.add_argument(
        "--structural_photo_semantics",
        action="store_true",
        help="restrict photo GT semantics to structural classes (wall/door/window)",
    )
    parser.add_argument("--clip_model", default=DEFAULT_CLIP_MODEL, help="OpenCLIP model identifier")
    parser.add_argument(
        "--recon_cache",
        default=None,
        help="directory of lossless float32 cached test-view reconstructions (saved on miss)",
    )
    parser.add_argument("--resume", action="store_true", help="resume from per-scene JSONL progress and skip finished modes")
    parser.add_argument("--sync_gcs_uri", default=None, help="optional gs://... directory URI to rsync progress and outputs")
    parser.add_argument("--sync_every_scenes", type=int, default=25, help="sync eval directory to GCS every N scenes")
    parser.add_argument("--max_scenes", type=int, default=None, help="evaluate only the first N test scenes")
    args = parser.parse_args()

    if args.lambda_sem is not None and args.lambda_sem < 0.0:
        raise ValueError(f"--lambda_sem must be non-negative, got {args.lambda_sem}")
    if args.sync_every_scenes <= 0:
        raise ValueError(f"--sync_every_scenes must be positive, got {args.sync_every_scenes}")
    if args.max_scenes is not None and args.max_scenes <= 0:
        raise ValueError(f"--max_scenes must be positive, got {args.max_scenes}")

    modes: list[SemMode] = list(dict.fromkeys(args.sem_modes or [args.sem_mode]))
    if args.output_dir:
        output_paths: dict[SemMode, str] = {m: os.path.join(args.output_dir, f"eval_s3d_{m}.json") for m in modes}
    elif len(modes) == 1:
        output_paths = {modes[0]: args.output}
    else:
        stem, ext = os.path.splitext(args.output)
        output_paths = {m: f"{stem}_{m}{ext}" for m in modes}

    pending_modes: list[SemMode] = []
    for m in modes:
        out_path = output_paths[m]
        if args.resume and os.path.exists(out_path):
            if _is_valid_metrics_json(out_path):
                continue
            print(f"[warn] Removing invalid/corrupt eval output {out_path}; re-evaluating mode {m!r}.")
            os.remove(out_path)
        pending_modes.append(m)
    if not pending_modes:
        print("All requested eval modes already complete; skipping.")
        return

    progress_dir = args.output_dir or (os.path.dirname(args.output) or ".")
    os.makedirs(progress_dir, exist_ok=True)
    progress_path = os.path.join(progress_dir, f"eval_progress_{'_'.join(pending_modes)}.jsonl")

    np.random.seed(42)
    recon_models: tuple[Any, Any] | None = load_models(recon_model="pi3-original") if args.recon_cache is None else None
    model = load_model(args.checkpoint_dir, args.dinov3_weights)

    any_semantics = args.photo_sem_source != "none" and any(m != "none" for m in pending_modes)
    semantic_adapter: SemanticAdapter | None = None
    text_encoder: ClipTextEncoder | None = None
    eff_lambda = 0.5
    if any_semantics:
        fallback_dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            device = next(model.parameters()).device if hasattr(model, "parameters") else fallback_dev
        except StopIteration:
            device = fallback_dev
        adapter_src = args.semantic_adapter or args.checkpoint_dir
        semantic_adapter, saved_lambda = load_semantic_adapter(adapter_src, device=device)
        semantic_adapter.eval()
        if args.lambda_sem is not None:
            eff_lambda = float(args.lambda_sem)
        elif saved_lambda is not None and saved_lambda > 0.0:
            eff_lambda = saved_lambda
        else:
            eff_lambda = 0.5
        if eff_lambda < 0.0:
            raise ValueError(f"lambda_sem must be non-negative, got {eff_lambda}")
        text_encoder = ClipTextEncoder(args.clip_model)

    eval_scenes = list(s3d.TEST_SCENES)
    if args.max_scenes is not None:
        eval_scenes = eval_scenes[: args.max_scenes]

    if args.resume:
        done_scenes, results_by_mode = _load_eval_progress(
            progress_path,
            pending_modes,
            allowed_scenes={int(s) for s in eval_scenes} if args.max_scenes is not None else None,
        )
    else:
        done_scenes, results_by_mode = set(), {m: {"t_errs": [], "r_errs": [], "failed": 0} for m in pending_modes}

    allowed = s3d.STRUCTURAL_PHOTO_LABELS if args.structural_photo_semantics else None
    progress_mode = "a" if args.resume else "w"
    scenes_since_sync = 0
    desc = f"eval s3d ({len(eval_scenes)} scenes)" if args.max_scenes is not None else "eval s3d"
    pbar = tqdm(eval_scenes, desc=desc)
    set_postfix = getattr(pbar, "set_postfix_str", None)
    close_pbar = getattr(pbar, "close", None)
    try:
        with open(progress_path, progress_mode, encoding="utf-8") as progress_f:
            for scene in pbar:
                if int(scene) in done_scenes:
                    continue
                scene_dir = os.path.join(args.s3d_root, f"scene_{scene:05d}")
                plan_img, plan_params = s3d.load_floorplan(scene_dir)
                plan_sem: SemanticMapInput | None = None
                if any_semantics:
                    assert text_encoder is not None
                    plan_sem_idx, plan_palette, _ = s3d.load_semantic_floorplan(
                        scene_dir,
                        text_encoder=text_encoder,
                        include_rooms=args.include_rooms,
                        plan_params=plan_params,
                    )
                    plan_sem = (plan_sem_idx, plan_palette, None)

                scene_stats: dict[SemMode, dict[str, Any]] = {
                    m: {"t_errs": [], "r_errs": [], "failed": 0} for m in pending_modes
                }
                views = s3d.list_views(scene_dir)
                for vi, view_dir in enumerate(views):
                    if set_postfix is not None:
                        set_postfix(f"scene {scene} view {vi + 1}/{len(views)}")
                    recon: ReconstructionDict | None = None
                    cache_path: str | None = None
                    if args.recon_cache:
                        cache_path = _view_recon_cache_path(args.recon_cache, int(scene), scene_dir, view_dir)
                        recon = _load_cached_recon(cache_path)
                    if recon is None:
                        if recon_models is None:
                            recon_models = load_models(recon_model="pi3-original")
                        recon = reconstruct(
                            [os.path.join(view_dir, "rgb_rawlight.png")], recon_models[0], recon_models[1]
                        )
                        if cache_path is not None:
                            _save_cached_recon(cache_path, recon)
                    photo_sem: SemanticMapInput | None = None
                    if any_semantics:
                        assert text_encoder is not None
                        p_idx, p_palette, _ = s3d.load_view_gt_semantics(
                            view_dir, text_encoder=text_encoder, allowed_labels=allowed
                        )
                        if p_idx.ndim == 2 and recon["conf"].ndim == 3:
                            p_idx = p_idx[None, ...]
                        photo_sem = (p_idx, p_palette, recon["conf"])

                    for sem_mode in pending_modes:
                        use_sem_for_mode = sem_mode != "none" and args.photo_sem_source != "none"
                        alignment: PoseAlignment
                        try:
                            if use_sem_for_mode:
                                assert semantic_adapter is not None and photo_sem is not None
                                alignment = align_to_floorplan(
                                    recon,
                                    plan_img,
                                    model,
                                    sem_mode=sem_mode,
                                    semantic_adapter=semantic_adapter,
                                    photo_sem=photo_sem,
                                    plan_sem=plan_sem,
                                    lambda_sem=eff_lambda,
                                )
                            else:
                                alignment = align_to_floorplan(recon, plan_img, model)
                        except AlignmentFailureError:  # no similarity transform: the identity is used
                            proj_params = reconstruct_projection_geometry(
                                recon["points_cam0"], recon["conf"], recon["est_gravity_cam"]
                            )[0]
                            alignment = {"proj_params": proj_params, "M": np.eye(2, 3)}
                            scene_stats[sem_mode]["failed"] += 1
                        t_err, r_err = pose_errors(recon, alignment, plan_params, view_dir)
                        scene_stats[sem_mode]["t_errs"].append(t_err)
                        scene_stats[sem_mode]["r_errs"].append(r_err)

                for m in pending_modes:
                    results_by_mode[m]["t_errs"].extend(scene_stats[m]["t_errs"])
                    results_by_mode[m]["r_errs"].extend(scene_stats[m]["r_errs"])
                    results_by_mode[m]["failed"] += scene_stats[m]["failed"]
                done_scenes.add(int(scene))
                progress_f.write(json.dumps({"scene": int(scene), "modes": scene_stats}) + "\n")
                progress_f.flush()
                scenes_since_sync += 1
                if args.sync_gcs_uri and scenes_since_sync >= args.sync_every_scenes:
                    _sync_eval_dir(progress_dir, args.sync_gcs_uri)
                    scenes_since_sync = 0
    finally:
        if close_pbar is not None:
            close_pbar()

    for sem_mode in pending_modes:
        stats = results_by_mode[sem_mode]
        metrics = compute_metrics(stats["t_errs"], stats["r_errs"])
        print(f"[{sem_mode}] {len(stats['t_errs'])} views, {stats['failed']} without a similarity transform")
        print("  ".join(f"{k:>16}" for k in metrics))
        print("  ".join(f"{v:16.2f}" for v in metrics.values()))
        out_path = output_paths[sem_mode]
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        tmp_path = f"{out_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        os.replace(tmp_path, out_path)

    if os.path.exists(progress_path):
        os.remove(progress_path)
    if args.sync_gcs_uri:
        _sync_eval_dir(progress_dir, args.sync_gcs_uri)


if __name__ == "__main__":
    main()

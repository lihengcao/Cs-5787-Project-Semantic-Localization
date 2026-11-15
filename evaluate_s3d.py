"""Evaluation on the Structured3D test scenes: single-view camera localization (Table 3).

    python evaluate_s3d.py --s3d_root <Structured3D>
"""
from __future__ import annotations

import argparse
import json
import os
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--s3d_root", required=True, help="Structured3D dataset: the directory of the scene_* folders")
    parser.add_argument("--checkpoint_dir", default=S3D_MODEL_REPO, help="LoRA layers")
    parser.add_argument("--dinov3_weights", default=None, help="Meta's DINOv3 ViT-B/16 weights")
    parser.add_argument("--output", default="outputs/s3d_results.json")
    parser.add_argument("--sem_mode", choices=["none", "uniform", "consensus"], default="none")
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
    args = parser.parse_args()

    if args.lambda_sem is not None and args.lambda_sem < 0.0:
        raise ValueError(f"--lambda_sem must be non-negative, got {args.lambda_sem}")

    np.random.seed(42)
    recon_model, geocalib = load_models(recon_model="pi3-original")
    model = load_model(args.checkpoint_dir, args.dinov3_weights)

    sem_mode: SemMode = args.sem_mode
    use_semantics = sem_mode != "none" and args.photo_sem_source != "none"
    semantic_adapter: SemanticAdapter | None = None
    text_encoder: ClipTextEncoder | None = None
    eff_lambda = 0.5
    if use_semantics:
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

    t_errs, r_errs, failed = [], [], 0
    for scene in tqdm(s3d.TEST_SCENES):
        scene_dir = os.path.join(args.s3d_root, f"scene_{scene:05d}")
        plan_img, plan_params = s3d.load_floorplan(scene_dir)
        plan_sem: SemanticMapInput | None = None
        if use_semantics:
            assert text_encoder is not None
            plan_sem_idx, plan_palette, _ = s3d.load_semantic_floorplan(
                scene_dir,
                text_encoder=text_encoder,
                include_rooms=args.include_rooms,
                plan_params=plan_params,
            )
            plan_sem = (plan_sem_idx, plan_palette, None)

        for view_dir in s3d.list_views(scene_dir):
            recon = reconstruct([os.path.join(view_dir, "rgb_rawlight.png")], recon_model, geocalib)
            try:
                if use_semantics:
                    assert text_encoder is not None and semantic_adapter is not None
                    allowed = s3d.STRUCTURAL_PHOTO_LABELS if args.structural_photo_semantics else None
                    p_idx, p_palette, _ = s3d.load_view_gt_semantics(
                        view_dir, text_encoder=text_encoder, allowed_labels=allowed
                    )
                    if p_idx.ndim == 2 and recon["conf"].ndim == 3:
                        p_idx = p_idx[None, ...]
                    photo_sem: SemanticMapInput = (p_idx, p_palette, recon["conf"])
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
                proj_params = reconstruct_projection_geometry(recon["points_cam0"], recon["conf"], recon["est_gravity_cam"])[0]
                alignment: PoseAlignment = {"proj_params": proj_params, "M": np.eye(2, 3)}
                failed += 1
            t_err, r_err = pose_errors(recon, alignment, plan_params, view_dir)
            t_errs.append(t_err)
            r_errs.append(r_err)

    metrics = compute_metrics(t_errs, r_errs)
    print(f"{len(t_errs)} views, {failed} without a similarity transform")
    print("  ".join(f"{k:>16}" for k in metrics))
    print("  ".join(f"{v:16.2f}" for v in metrics.values()))
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()

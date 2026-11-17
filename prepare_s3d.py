"""Training data from Structured3D: density maps of single views and their correspondences with the floorplan.

Every view gives two samples: one from its pi3 reconstruction, one from its ground-truth depth map and camera.

    python prepare_s3d.py --s3d_root <Structured3D> --output <data>
"""
from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from PIL import Image
from tqdm import tqdm

from scenealigner import structured3d as s3d
from scenealigner.evaluation import process_gt_correspondences
from scenealigner.inference import reconstruct_projection_geometry
from scenealigner.preprocess import load_models, reconstruct
from scenealigner.semantics import (
    DEFAULT_CLIP_MODEL,
    ClipTextEncoder,
    aggregate_topdown_semantics,
    project_posed_semantics,
    save_semantic_map,
)
from scenealigner.training import render_training_map

if TYPE_CHECKING:
    from scenealigner.evaluation import CorrespondencesDict
    from scenealigner.preprocess import ReconstructionDict

MIN_CORRESPONDENCES = 10


def depth_sample(
    world_points: NDArray[np.float64],
    plan_xy: NDArray[np.floating[Any]],
    plan_gray: NDArray[np.uint8],
    size: int = 512,
    padding: float = 0.1,
) -> tuple[Image.Image, NDArray[np.float32], NDArray[np.float32]]:
    """Density map of the ground-truth points of a view, seen from above in the world frame (z up), and its
    correspondences: the pixels of the points that lie on walls (dark pixels) of the floorplan."""
    valid = ~np.isnan(world_points).any(axis=-1)
    points, plan_xy = world_points[valid], plan_xy[valid]

    # Height (lowest 20% and highest 5% removed) and x/y percentile filters
    keep = (points[:, 2] >= np.percentile(points[:, 2], 20.0)) & (points[:, 2] <= np.percentile(points[:, 2], 95.0))
    points, plan_xy = points[keep], plan_xy[keep]
    x, y = points[:, 0], points[:, 1]
    keep = (x >= np.percentile(x, 2.5)) & (x <= np.percentile(x, 97.5)) & (y >= np.percentile(y, 2.5)) & (y <= np.percentile(y, 97.5))
    x, y, plan_xy = x[keep], y[keep], plan_xy[keep]

    # Top-down projection, centered in the map with a margin
    width, height = x.max() - x.min(), y.max() - y.min()
    scale = (size * (1 - padding * 2)) / max(width, height)
    u = ((x - x.min()) * scale + (size - width * scale) / 2).astype(np.int32)
    v = size - 1 - ((y - y.min()) * scale + (size - height * scale) / 2).astype(np.int32)
    density_map = np.power(np.bincount(v * size + u, minlength=size * size).reshape(size, size).astype(np.float32), 0.5)
    density_img = Image.fromarray((255 - density_map / density_map.max() * 255.0).astype(np.uint8))

    inside = ((plan_xy >= 0) & (plan_xy < size)).all(axis=1)
    density_px, plan_px = np.stack([u, v], axis=1)[inside].astype(np.float32), plan_xy[inside].astype(np.float32)
    plan_idx = np.clip(plan_px.astype(int), 0, size - 1)
    on_wall = plan_gray[plan_idx[:, 1], plan_idx[:, 0]] < 50
    return density_img, density_px[on_wall], plan_px[on_wall]


def pi3_sample(
    recon: ReconstructionDict, plan_xy: NDArray[np.floating[Any]], plan_gray: NDArray[np.uint8]
) -> tuple[Image.Image, NDArray[np.float32], NDArray[np.float32]]:
    """Density map of the pi3 reconstruction of a view and its correspondences: the pixels of the reconstructed
    points whose ground-truth positions (plan_xy) lie on walls of the floorplan."""
    inside = ~np.isnan(plan_xy).any(axis=-1) & ((plan_xy >= 0) & (plan_xy < plan_gray.shape[0])).all(axis=-1)
    plan_idx = plan_xy[inside].astype(int)
    on_wall = inside.copy()
    on_wall[inside] = plan_gray[plan_idx[:, 1], plan_idx[:, 0]] < 50
    v, u = np.nonzero(on_wall)
    corr_data: CorrespondencesDict = {"tensor_coords": np.stack([u, v], axis=1).astype(np.float32), "plan_coords": plan_xy[on_wall].astype(np.float32)}

    proj_params, points, mask = reconstruct_projection_geometry(recon["points_cam0"], recon["conf"], recon["est_gravity_cam"])
    density_px, plan_px = process_gt_correspondences(corr_data, 0, recon["points_cam0"], proj_params, mask)
    return render_training_map(points, proj_params), density_px, plan_px


def project_depth_semantics(
    world_points: NDArray[np.floating[Any]],
    aligned_sem_idx: NDArray[np.integer[Any]],
    view_palette: NDArray[np.floating[Any]],
    view_labels: Sequence[str] | None = None,
    size: int = 512,
    padding: float = 0.1,
    tau: float = 5.0,
) -> tuple[NDArray[np.uint8], NDArray[np.float32], NDArray[np.float16], list[str]]:
    """Project view semantics along Structured3D depth points using the exact depth_sample grid & filters."""
    aligned_arr = np.asarray(aligned_sem_idx)
    if aligned_arr.shape != world_points.shape[:2]:
        raise ValueError(
            f"aligned_sem_idx shape {aligned_arr.shape} must match world_points spatial shape {world_points.shape[:2]}"
        )
    empty_args = (
        np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.int32),
        view_palette,
        size,
    )
    valid = ~np.isnan(world_points).any(axis=-1)
    points = world_points[valid]
    sem_pts = aligned_arr[valid]
    if len(points) == 0:
        return aggregate_topdown_semantics(*empty_args, tau=tau, labels=view_labels)

    keep_z = (points[:, 2] >= np.percentile(points[:, 2], 20.0)) & (points[:, 2] <= np.percentile(points[:, 2], 95.0))
    points, sem_pts = points[keep_z], sem_pts[keep_z]
    if len(points) == 0:
        return aggregate_topdown_semantics(*empty_args, tau=tau, labels=view_labels)

    x, y = points[:, 0], points[:, 1]
    keep_xy = (
        (x >= np.percentile(x, 2.5)) & (x <= np.percentile(x, 97.5))
        & (y >= np.percentile(y, 2.5)) & (y <= np.percentile(y, 97.5))
    )
    x, y, sem_pts = x[keep_xy], y[keep_xy], sem_pts[keep_xy]
    if len(x) == 0:
        return aggregate_topdown_semantics(*empty_args, tau=tau, labels=view_labels)

    width, height = x.max() - x.min(), y.max() - y.min()
    span = max(width, height)
    if span <= 0:
        return aggregate_topdown_semantics(*empty_args, tau=tau, labels=view_labels)
    scale = (size * (1 - padding * 2)) / span
    u = ((x - x.min()) * scale + (size - width * scale) / 2).astype(np.int32)
    v = size - 1 - ((y - y.min()) * scale + (size - height * scale) / 2).astype(np.int32)
    return aggregate_topdown_semantics(v, u, sem_pts, view_palette, size, tau=tau, labels=view_labels)


def project_pi3_semantics(
    recon: ReconstructionDict,
    aligned_sem_idx: NDArray[np.integer[Any]],
    view_palette: NDArray[np.floating[Any]],
    view_labels: Sequence[str] | None = None,
    tau: float = 5.0,
) -> tuple[NDArray[np.uint8], NDArray[np.float32], NDArray[np.float16], list[str]]:
    """Project view semantics along pi3 points matching render_training_map rounding."""
    proj_params, points, mask = reconstruct_projection_geometry(recon["points_cam0"], recon["conf"], recon["est_gravity_cam"])
    return project_posed_semantics(
        points,
        mask,
        aligned_sem_idx,
        view_palette,
        proj_params,
        tau=tau,
        training_rounding=True,
        labels=view_labels,
    )


def _load_part_progress(part_path: str) -> tuple[set[str], list[dict[str, Any]]]:
    """Load completed scene IDs and rows from a .jsonl.part file, truncating any partial tail."""
    done_scenes: set[str] = set()
    pairs: list[dict[str, Any]] = []
    if not os.path.exists(part_path):
        return done_scenes, pairs
    valid_offset = 0
    with open(part_path, "rb") as f:
        while raw := f.readline():
            if not raw.endswith(b"\n"):
                break
            try:
                rec = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                break
            if not isinstance(rec, dict) or "scene_id" not in rec or "rows" not in rec:
                break
            done_scenes.add(str(rec["scene_id"]))
            pairs.extend(rec["rows"])
            valid_offset = f.tell()
    if valid_offset < os.path.getsize(part_path):
        with open(part_path, "r+b") as f:
            f.truncate(valid_offset)
    return done_scenes, pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--s3d_root", required=True, help="Structured3D dataset: the directory of the scene_* folders")
    parser.add_argument("--output", required=True)
    parser.add_argument("--scenes", type=int, nargs=2, default=[s3d.TRAIN_SCENES.start, s3d.TRAIN_SCENES.stop], help="first train scene and the one after the last")
    parser.add_argument("--no_semantics", action="store_true", help="skip all semantic map generation (floorplan + photo)")
    parser.add_argument("--include_rooms", action="store_true", help="include room-type labels in floorplan semantics")
    parser.add_argument("--clip_model", default=DEFAULT_CLIP_MODEL, help="HuggingFace CLIP model ID for text palette encoding")
    parser.add_argument("--photo_sem_source", choices=["gt", "none"], default="gt", help="2D photo semantic source")
    parser.add_argument("--structural_photo_semantics", action="store_true", help="restrict photo GT semantics to wall/door/window")
    parser.add_argument("--resume", action="store_true", help="skip completed slice CSVs and resume from per-scene .jsonl.part")
    args = parser.parse_args()

    out_csv = os.path.join(args.output, f"scenes_{args.scenes[0]}_{args.scenes[1]}.csv")
    part_path = f"{out_csv}.jsonl.part"
    if args.resume and os.path.exists(out_csv):
        print(f"Skipping completed slice {out_csv}")
        return

    export_floorplan_sem = not args.no_semantics
    export_photo_sem = (not args.no_semantics) and (args.photo_sem_source != "none")
    allowed_photo_labels = s3d.STRUCTURAL_PHOTO_LABELS if args.structural_photo_semantics else None

    recon_model, geocalib = load_models(recon_model="pi3-original")
    text_encoder = None if args.no_semantics else ClipTextEncoder(model_name=args.clip_model)
    os.makedirs(os.path.join(args.output, "floorplan"), exist_ok=True)
    if export_floorplan_sem:
        os.makedirs(os.path.join(args.output, "floorplan_sem"), exist_ok=True)

    done_scenes, pairs = _load_part_progress(part_path) if args.resume else (set(), [])
    part_mode = "a" if args.resume else "w"
    pbar = tqdm(range(*args.scenes), desc=f"s3d {args.scenes[0]}..{args.scenes[1]}")
    set_postfix = getattr(pbar, "set_postfix_str", None)
    close_pbar = getattr(pbar, "close", None)
    try:
        with open(part_path, part_mode, encoding="utf-8") as part_f:
            for scene in pbar:
                scene_id = f"scene_{scene:05d}"
                if scene_id in done_scenes:
                    continue
                scene_dir = os.path.join(args.s3d_root, scene_id)
                # Scenes 1600 to 1799 have no perspective views
                if scene not in s3d.TRAIN_SCENES or scene in s3d.OFF_CANVAS_SCENES or not os.path.isdir(os.path.join(scene_dir, "2D_rendering")):
                    continue
                plan_img, plan_params = s3d.load_floorplan(scene_dir)
                plan_path = os.path.join("floorplan", f"{scene_id}.png")
                plan_img.save(os.path.join(args.output, plan_path))
                plan_gray = np.array(plan_img.convert("L"))

                sem_prefix = None
                if export_floorplan_sem:
                    sem_idx, palette, labels = s3d.load_semantic_floorplan(
                        scene_dir, text_encoder=text_encoder, include_rooms=args.include_rooms, plan_params=plan_params
                    )
                    sem_prefix = os.path.join("floorplan_sem", scene_id)
                    save_semantic_map(os.path.join(args.output, sem_prefix), sem_idx, palette, labels=labels)

                views = s3d.list_views(scene_dir)
                scene_rows: list[dict[str, Any]] = []
                for vi, view_dir in enumerate(views):
                    if set_postfix is not None:
                        set_postfix(f"{scene_id} view {vi + 1}/{len(views)}")
                    world_points = s3d.load_world_points(view_dir)
                    plan_xy = s3d.world_to_plan(world_points, plan_params)
                    recon = reconstruct([os.path.join(view_dir, "rgb_rawlight.png")], recon_model, geocalib)
                    samples = {"depth": depth_sample(world_points, plan_xy, plan_gray), "pi3": pi3_sample(recon, plan_xy, plan_gray)}

                    view_sem_args: tuple[NDArray[np.uint8], NDArray[np.float16], list[str]] | None = None
                    if export_photo_sem and any(len(sample[1]) >= MIN_CORRESPONDENCES for sample in samples.values()):
                        view_sem_args = s3d.load_view_gt_semantics(
                            view_dir, text_encoder=text_encoder, allowed_labels=allowed_photo_labels
                        )

                    room_id, position_id = view_dir.split(os.sep)[-4], view_dir.split(os.sep)[-1]
                    for source, sample in samples.items():
                        if len(sample[1]) < MIN_CORRESPONDENCES:
                            continue
                        density_img, density_px, plan_px = sample
                        name = os.path.join(source, scene_id, f"{room_id}_{position_id}")
                        os.makedirs(os.path.join(args.output, source, scene_id), exist_ok=True)
                        density_img.save(os.path.join(args.output, f"{name}.png"))
                        np.save(os.path.join(args.output, f"{name}.npy"), np.stack([plan_px, density_px]))
                        row = {"scene_id": scene_id, "density": f"{name}.png", "floorplan": plan_path, "correspondences": f"{name}.npy"}
                        if sem_prefix is not None:
                            row["floorplan_sem"] = sem_prefix
                        if view_sem_args is not None:
                            v_idx, v_pal, v_lbls = view_sem_args
                            if source == "depth":
                                d_idx, d_rel, d_pal, d_lbls = project_depth_semantics(world_points, v_idx, v_pal, v_lbls)
                            else:
                                d_idx, d_rel, d_pal, d_lbls = project_pi3_semantics(recon, v_idx, v_pal, v_lbls)
                            sem_name = os.path.join(f"{source}_sem", scene_id, f"{room_id}_{position_id}")
                            save_semantic_map(os.path.join(args.output, sem_name), d_idx, d_pal, sem_rel=d_rel, labels=d_lbls)
                            row["density_sem"] = sem_name
                        scene_rows.append(row)
                pairs.extend(scene_rows)
                done_scenes.add(scene_id)
                part_f.write(json.dumps({"scene_id": scene_id, "rows": scene_rows}) + "\n")
                part_f.flush()
    finally:
        if close_pbar is not None:
            close_pbar()
    columns = (
        ["scene_id", "density", "floorplan", "correspondences"]
        + (["floorplan_sem"] if export_floorplan_sem else [])
        + (["density_sem"] if export_photo_sem else [])
    )
    tmp_csv = f"{out_csv}.tmp"
    pd.DataFrame(pairs, columns=columns).to_csv(tmp_csv, index=False)
    os.replace(tmp_csv, out_csv)
    if os.path.exists(part_path):
        os.remove(part_path)


if __name__ == "__main__":
    main()

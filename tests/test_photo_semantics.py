"""Tests for Workstream 2a: GT photo semantic loading, grid alignment, and 3D top-down consensus projection."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import pytest

import prepare_s3d
from scenealigner import structured3d as s3d
from scenealigner.inference import ProjParams, get_density_coordinates, render_density_map
from scenealigner.preprocess import get_resized_size
from scenealigner.semantics import (
    aggregate_topdown_semantics,
    align_mask_to_grid,
    load_semantic_map,
    pool_semantic_patches,
    project_posed_semantics,
    save_semantic_map,
)
from scenealigner.training import render_training_map


class _OrthoTextEncoder:
    """Deterministic unit-vector encoder producing distinct orthogonal vectors for unit tests."""

    def __init__(self, dim: int = 16) -> None:
        self.dim = dim

    def encode_palette(self, labels: list[str] | tuple[str, ...]) -> np.ndarray:
        palette = np.zeros((len(labels) + 1, self.dim), dtype=np.float32)
        for idx, _ in enumerate(labels, start=1):
            palette[idx, (idx - 1) % self.dim] = 1.0
        return palette.astype(np.float16)


def test_align_mask_to_grid_landscape_portrait_and_crop() -> None:
    # Landscape 1280x720 in pad mode -> (518, 294) centered vertically with 112 px padding top/bottom
    land_mask = np.zeros((720, 1280), dtype=np.uint8)
    land_mask[100:600, 200:1000] = 3
    land_conf = np.full((720, 1280), 0.75, dtype=np.float32)

    aligned_mask, aligned_conf = align_mask_to_grid(land_mask, target_size=518, mode="pad", conf=land_conf)
    new_w, new_h = get_resized_size(1280, 720, 518, "pad")
    assert (new_w, new_h) == (518, 294)
    assert aligned_mask.shape == (518, 518) and aligned_mask.dtype == np.uint8
    assert aligned_conf.shape == (518, 518) and aligned_conf.dtype == np.float32
    assert set(np.unique(aligned_mask).tolist()).issubset({0, 3})
    assert np.all(aligned_mask[:112, :] == 0)
    assert np.all(aligned_mask[112 + 294:, :] == 0)
    assert np.all(aligned_conf[:112, :] == 0.0)
    assert np.all(aligned_conf[112 + 294:, :] == 0.0)
    assert int(aligned_mask[259, 259]) == 3
    assert aligned_conf[259, 259] == pytest.approx(0.75, abs=1e-4)

    # Portrait 720x1280 in pad mode -> (294, 518) centered horizontally with 112 px padding left/right
    port_mask = np.full((1280, 720), 7, dtype=np.uint8)
    port_pad = align_mask_to_grid(port_mask, target_size=518, mode="pad")
    assert port_pad.shape == (518, 518)
    assert np.all(port_pad[:, :112] == 0)
    assert np.all(port_pad[:, 112 + 294:] == 0)
    assert np.all(port_pad[:, 112:112 + 294] == 7)

    # Portrait 720x1280 in crop mode -> resized to (518, 924), middle [203:721] kept (top/bottom bands dropped)
    crop_src = np.full((1280, 720), 5, dtype=np.uint8)
    crop_src[:200, :] = 1  # lies strictly above crop window (200 * 924 / 1280 = 144 < 203)
    crop_src[1100:, :] = 2  # lies strictly below crop window (1100 * 924 / 1280 = 794 > 721)
    port_crop = align_mask_to_grid(crop_src, target_size=518, mode="crop")
    assert port_crop.shape == (518, 518)
    assert np.all(port_crop == 5)

    # Batched (S, H, W) input preserves batch dimension and validates conf shape
    batched_mask = np.stack([land_mask, land_mask], axis=0)
    batched_conf = np.stack([land_conf, land_conf], axis=0)
    b_mask, b_conf = align_mask_to_grid(batched_mask, target_size=518, mode="pad", conf=batched_conf)
    assert b_mask.shape == (2, 518, 518)
    assert b_conf.shape == (2, 518, 518)
    with pytest.raises(ValueError):
        align_mask_to_grid(land_mask, target_size=518, conf=np.ones((10, 10), dtype=np.float32))


def test_aggregate_topdown_semantics_consensus_and_compaction(tmp_path: Path) -> None:
    # Row 0 unused zero vector; 1..4 unit basis concepts + unused concept 4 ("sofa") + negative opposite 5
    dim = 16
    palette = np.zeros((6, dim), dtype=np.float16)
    palette[1, 0] = 1.0  # wall
    palette[2, 1] = 1.0  # chair
    palette[3, 2] = 1.0  # door
    palette[4, 3] = 1.0  # sofa (not observed in points -> must be compacted away)
    palette[5, 0] = -1.0  # opposite vector to test zero-norm cancellation
    labels = ["wall", "chair", "door", "sofa", "anti_wall"]

    # Build synthetic points in 4 test cells on a 64x64 grid:
    # (10, 10): 1 unanimous wall point -> rel = 1 - exp(-1/5)
    # (20, 20): 10 unanimous wall points -> rel = 1 - exp(-10/5)
    # (30, 30): conflict: 6 chair points + 4 wall points -> mu = (0.4, 0.6), winner chair, attenuated reliability
    # (40, 40): exact opposite 1 wall + 1 anti_wall -> mu = 0 -> dropped to 0
    rows = np.array([10] + [20] * 10 + [30] * 10 + [40, 40], dtype=np.int32)
    cols = np.array([10] + [20] * 10 + [30] * 10 + [40, 40], dtype=np.int32)
    sem_pts = np.array([1] + [1] * 10 + [2] * 6 + [1] * 4 + [1, 5], dtype=np.int32)

    sem_idx, sem_rel, compact_pal, compact_labels = aggregate_topdown_semantics(
        rows, cols, sem_pts, palette, density_size=64, tau=5.0, labels=labels
    )

    assert sem_idx.shape == (64, 64) and sem_idx.dtype == np.uint8
    assert sem_rel.shape == (64, 64) and sem_rel.dtype == np.float32
    assert set(compact_labels) == {"wall", "chair"}
    assert "sofa" not in compact_labels and "door" not in compact_labels and "anti_wall" not in compact_labels
    assert compact_pal.shape == (len(compact_labels) + 1, dim)
    assert np.allclose(compact_pal[0], 0.0)

    wall_id = compact_labels.index("wall") + 1
    chair_id = compact_labels.index("chair") + 1
    assert sem_idx[10, 10] == wall_id
    assert sem_idx[20, 20] == wall_id
    assert sem_idx[30, 30] == chair_id
    assert sem_idx[40, 40] == 0 and sem_rel[40, 40] == 0.0

    expected_single = 1.0 - math.exp(-1.0 / 5.0)
    expected_multi = 1.0 - math.exp(-10.0 / 5.0)
    # Conflict cell: mu = (0.4, 0.6, 0..), ||mu|| = sqrt(0.52), hat_e = (0.4, 0.6) / sqrt(0.52), cos with chair = 0.6 / sqrt(0.52)
    # r = ||mu|| * (1 - exp(-10/5)) * cos = 0.6 * (1 - exp(-2))
    expected_conflict = 0.6 * (1.0 - math.exp(-10.0 / 5.0))
    assert sem_rel[10, 10] == pytest.approx(expected_single, abs=1e-4)
    assert sem_rel[20, 20] == pytest.approx(expected_multi, abs=1e-4)
    assert sem_rel[30, 30] == pytest.approx(expected_conflict, abs=1e-4)
    assert sem_rel[20, 20] > sem_rel[30, 30] > sem_rel[10, 10]

    # Disk round-trip + patch pooling check
    prefix = str(tmp_path / "topdown_sem")
    save_semantic_map(prefix, sem_idx, compact_pal, sem_rel=sem_rel, labels=compact_labels)
    l_idx, l_pal, l_rel = load_semantic_map(prefix)
    patch_feat, patch_rel = pool_semantic_patches(l_idx, l_pal, l_rel, patch_size=16)
    assert patch_feat.shape == (dim, 4, 4)
    assert patch_rel.shape == (1, 4, 4)

    # Empty valid points return empty maps cleanly
    e_idx, e_rel, e_pal, e_lbls = aggregate_topdown_semantics(
        np.array([], dtype=np.int32),
        np.array([], dtype=np.int32),
        np.array([], dtype=np.int32),
        palette,
        density_size=64,
        labels=labels,
    )
    assert np.all(e_idx == 0) and np.all(e_rel == 0.0)
    assert e_pal.shape == (1, dim) and e_lbls == []


def test_project_posed_semantics_multiview_and_rounding_parity() -> None:
    size = 64
    proj_params: ProjParams = {
        "R_rect": np.eye(3, dtype=np.float64),
        "min_x": 0.0,
        "min_z": 0.0,
        "scale": 1.0,
        "cx": 0.0,
        "cz": 0.0,
        "density_size": size,
    }
    palette = np.zeros((3, 16), dtype=np.float16)
    palette[1, 0] = 1.0  # wall
    palette[2, 1] = 1.0  # door
    labels = ["wall", "door"]

    # Fractional z=10.3 yields v_f=10.3:
    # training_rounding=True -> size - 1 - int(10.3) = 53
    # training_rounding=False -> int(float32(63 - 10.3)) = int(52.7) = 52
    pts = np.array([[25.4, 0.0, 10.3]], dtype=np.float64)
    mask = np.ones((1, 1, 1), dtype=bool)
    sem_grid = np.array([[[1]]], dtype=np.uint8)

    train_idx, _, _, _ = project_posed_semantics(
        pts, mask, sem_grid, palette, proj_params, training_rounding=True, labels=labels
    )
    inf_idx, _, _, _ = project_posed_semantics(
        pts, mask, sem_grid, palette, proj_params, training_rounding=False, labels=labels
    )

    train_den = np.array(render_training_map(pts, proj_params).convert("L"))
    inf_den = np.array(render_density_map(pts, proj_params).convert("L"))
    # Baseline maps must match their exact respective rows 53 and 52
    assert train_den[53, 25] < 255 and train_den[52, 25] == 255
    assert inf_den[52, 25] < 255 and inf_den[53, 25] == 255
    assert train_idx[53, 25] > 0 and train_idx[52, 25] == 0
    assert inf_idx[52, 25] > 0 and inf_idx[53, 25] == 0

    # Multi-view (S=2): both views observing identical point (20, 30) boosts reliability over S=1
    mv_mask = np.ones((2, 2, 2), dtype=bool)
    mv_pts = np.tile(np.array([[30.2, 0.0, 20.2]], dtype=np.float64), (8, 1))
    mv_sem_agree = np.ones((2, 2, 2), dtype=np.uint8)  # 8 agreeing wall points at cell
    sv_idx, sv_rel, _, _ = project_posed_semantics(
        mv_pts[:4], mv_mask[:1], mv_sem_agree[:1], palette, proj_params, tau=5.0, labels=labels
    )
    mv_idx, mv_rel, _, _ = project_posed_semantics(
        mv_pts, mv_mask, mv_sem_agree, palette, proj_params, tau=5.0, labels=labels
    )
    coords = get_density_coordinates(mv_pts[:1], proj_params).astype(np.int32)[0]
    col, row = int(coords[0]), int(coords[1])
    assert sv_idx[row, col] > 0 and mv_idx[row, col] > 0
    assert mv_rel[row, col] > sv_rel[row, col]

    # Multi-view disagreement across cameras lowers reliability compared to agreement
    mv_sem_disagree = np.stack([np.ones((2, 2), dtype=np.uint8), np.full((2, 2), 2, dtype=np.uint8)], axis=0)
    _, dis_rel, _, _ = project_posed_semantics(
        mv_pts, mv_mask, mv_sem_disagree, palette, proj_params, tau=5.0, labels=labels
    )
    assert dis_rel[row, col] < mv_rel[row, col]


def test_load_view_gt_semantics_nyu40_and_palette_png(tmp_path: Path) -> None:
    view_dir = tmp_path / "view_0"
    view_dir.mkdir(parents=True)

    # Synthetic 1280x720 NYUv2 class map covering wall(1), floor(2), chair(5), door(8),
    # ceiling(22), otherstructure(38), otherfurniture(39), otherprop(40)
    raw = np.zeros((720, 1280), dtype=np.uint8)
    raw[100:250, 100:300] = 1   # wall
    raw[250:400, 100:300] = 2   # floor (default excluded)
    raw[100:250, 400:600] = 5   # chair
    raw[100:250, 700:900] = 8   # door
    raw[0:80, 100:300] = 22     # ceiling (default excluded)
    raw[400:550, 100:300] = 38  # otherstructure (ignored)
    raw[400:550, 400:600] = 39  # otherfurniture (ignored)
    raw[400:550, 700:900] = 40  # otherprop (ignored)

    # Save as palette-mode 'P' PNG with arbitrary non-identity RGB colormap to confirm raw indices aren't RGB-expanded
    pal_img = Image.fromarray(raw, mode="P")
    colormap = [((i * 73) % 255, (i * 139) % 255, (i * 197) % 255) for i in range(256)]
    pal_img.putpalette([c for rgb in colormap for c in rgb])
    pal_img.save(view_dir / "semantic.png")

    encoder: Any = _OrthoTextEncoder(dim=16)
    sem_idx, palette, labels = s3d.load_view_gt_semantics(str(view_dir), text_encoder=encoder)
    assert sem_idx.shape == (518, 518) and sem_idx.dtype == np.uint8
    assert labels == ["wall", "chair", "door"]
    assert palette.shape == (4, 16)
    # 112-px vertical padding must remain 0
    assert np.all(sem_idx[:112, :] == 0) and np.all(sem_idx[112 + 294:, :] == 0)
    assert set(np.unique(sem_idx).tolist()) == {0, 1, 2, 3}

    # Structural-only ablation drops 'chair' and keeps only 'wall', 'door'
    st_idx, st_pal, st_labels = s3d.load_view_gt_semantics(
        str(view_dir), text_encoder=encoder, allowed_labels=s3d.STRUCTURAL_PHOTO_LABELS
    )
    assert st_labels == ["wall", "door"]
    assert st_pal.shape == (3, 16)
    assert set(np.unique(st_idx).tolist()) == {0, 1, 2}

    # Verify grayscale 'L' PNG loads identically
    Image.fromarray(raw, mode="L").save(view_dir / "semantic.png")
    l_idx, l_pal, l_labels = s3d.load_view_gt_semantics(str(view_dir), text_encoder=encoder)
    assert l_labels == labels
    assert np.array_equal(l_idx, sem_idx)
    assert np.array_equal(l_pal, palette)


def test_prepare_s3d_depth_and_pi3_semantic_projection_align_with_density() -> None:
    size = 518
    rng = np.random.default_rng(42)

    # Synthetic 518x518 world_points with valid band matching S3D padded view [112:406, :]
    world_points = np.full((size, size, 3), np.nan, dtype=np.float64)
    yy, xx = np.meshgrid(np.arange(112, 406), np.arange(size), indexing="ij")
    world_points[112:406, :, 0] = (xx - 259.0) * 10.0 + rng.normal(0.0, 2.0, size=xx.shape)
    world_points[112:406, :, 1] = 2000.0 + rng.normal(0.0, 2.0, size=xx.shape)
    world_points[112:406, :, 2] = 500.0 + (yy - 112.0) * 6.0  # height spans ceiling..floor

    plan_xy = np.stack([np.full((size, size), 100.0), np.full((size, size), 100.0)], axis=-1)
    plan_gray = np.zeros((512, 512), dtype=np.uint8)

    view_sem_idx = np.zeros((size, size), dtype=np.uint8)
    view_sem_idx[112:260, :] = 1  # wall
    view_sem_idx[260:406, :] = 2  # door
    palette = np.zeros((3, 16), dtype=np.float16)
    palette[1, 0] = 1.0
    palette[2, 1] = 1.0
    labels = ["wall", "door"]

    # Default size=512 must match between depth_sample and project_depth_semantics and be divisible by patch_size=16
    den_img, _, _ = prepare_s3d.depth_sample(world_points, plan_xy, plan_gray)
    d_idx, d_rel, d_pal, d_lbls = prepare_s3d.project_depth_semantics(
        world_points, view_sem_idx, palette, labels
    )
    den_arr = np.array(den_img.convert("L"))
    assert d_idx.shape == den_arr.shape == (512, 512) and d_rel.shape == (512, 512)
    assert np.any(d_idx > 0)
    # With every valid world point assigned a semantic class, semantic support matches density support 1:1
    assert np.array_equal(d_idx > 0, den_arr < 255)
    assert set(d_lbls).issubset({"wall", "door"}) and d_pal.shape[0] == len(d_lbls) + 1
    d_patch_feat, d_patch_rel = pool_semantic_patches(d_idx, d_pal, d_rel, patch_size=16)
    assert d_patch_feat.shape == (16, 32, 32) and d_patch_rel.shape == (1, 32, 32)

    # Synthetic pi3 reconstruction (S=1, H=518, W=518)
    cam_pts = np.zeros((1, size, size, 3), dtype=np.float32)
    cam_pts[0, :, :, 0] = np.broadcast_to(np.linspace(-2.0, 2.0, size, dtype=np.float32)[None, :], (size, size))
    cam_pts[0, :, :, 1] = np.broadcast_to(np.linspace(-1.0, 1.0, size, dtype=np.float32)[:, None], (size, size))
    cam_pts[0, :, :, 2] = 3.0 + np.broadcast_to(np.linspace(0.0, 1.0, size, dtype=np.float32)[None, :], (size, size))
    conf = np.ones((1, size, size), dtype=np.float32)
    conf[0, 112:406, 80:440] = 5.0
    recon: Any = {
        "points_cam0": cam_pts,
        "conf": conf,
        "est_gravity_cam": np.array([0.0, 1.0, 0.0], dtype=np.float32),
    }

    pi3_img, _, _ = prepare_s3d.pi3_sample(recon, plan_xy[:size, :size], plan_gray)
    p_idx, p_rel, p_pal, p_lbls = prepare_s3d.project_pi3_semantics(recon, view_sem_idx, palette, labels)
    pi3_arr = np.array(pi3_img.convert("L"))
    assert p_idx.shape == pi3_arr.shape == (512, 512) and p_rel.shape == (512, 512)
    assert np.any(p_idx > 0)
    assert np.array_equal(p_idx > 0, pi3_arr < 255)
    assert p_pal.shape[0] == len(p_lbls) + 1
    p_patch_feat, _ = pool_semantic_patches(p_idx, p_pal, p_rel, patch_size=16)
    assert p_patch_feat.shape == (16, 32, 32)


def test_align_mask_to_grid_int64_and_dtype_preservation() -> None:
    high_val = np.int64(2**35 + 7)
    mask64 = np.zeros((720, 1280), dtype=np.int64)
    mask64[200:500, 300:900] = high_val

    out64 = align_mask_to_grid(mask64, target_size=518, mode="pad")
    assert out64.dtype == np.int64 and out64.shape == (518, 518)
    assert int(out64[259, 259]) == int(high_val)
    assert np.all(out64[:112, :] == 0)

    mask_u64 = mask64.astype(np.uint64)
    out_u64 = align_mask_to_grid(mask_u64, target_size=518, mode="crop")
    assert out_u64.dtype == np.uint64 and out_u64.shape == (518, 518)
    assert int(out_u64[259, 259]) == int(high_val)


def test_aggregate_topdown_semantics_cap255_conf_and_min_snap_cos() -> None:
    import warnings

    # 300 distinct concepts (> 255 limit for uint8): verify deterministic top-255 compaction & normalized palette
    rng = np.random.default_rng(7)
    raw_vecs = rng.normal(size=(300, 16)).astype(np.float32)
    palette_big = np.vstack([np.zeros((1, 16), dtype=np.float32), raw_vecs * 3.0])  # deliberately unnormalized
    rows = np.arange(300, dtype=np.int32) // 64
    cols = np.arange(300, dtype=np.int32) % 64
    sem_pts = np.arange(1, 301, dtype=np.int32)
    # Give concepts 256..300 two cells each while 1..45 have one cell, or replicate last 255 twice
    rows = np.concatenate([rows, np.arange(45, 300, dtype=np.int32) // 64 + 10])
    cols = np.concatenate([cols, np.arange(45, 300, dtype=np.int32) % 64])
    sem_pts = np.concatenate([sem_pts, np.arange(46, 301, dtype=np.int32)])
    labels_big = [f"c{i}" for i in range(1, 301)]

    idx_big, _, pal_big, lbls_big = aggregate_topdown_semantics(
        rows, cols, sem_pts, palette_big, density_size=64, labels=labels_big
    )
    assert idx_big.dtype == np.uint8 and len(lbls_big) == 255 and pal_big.shape == (256, 16)
    assert lbls_big == [f"c{i}" for i in range(46, 301)]
    # Compacted palette rows must be L2-normalized despite unnormalized input
    pal_norms = np.linalg.norm(pal_big[1:].astype(np.float32), axis=1)
    assert np.allclose(pal_norms, 1.0, atol=1e-2)

    # Confidence weighting, +inf/NaN filtering without warnings, and min_snap_cos pruning
    pal = np.zeros((3, 16), dtype=np.float16)
    pal[1, 0] = 1.0  # e1
    pal[2, 1] = 1.0  # e2
    r = np.array([5, 5, 5, 6, 7], dtype=np.int32)
    c = np.array([5, 5, 5, 6, 7], dtype=np.int32)
    s = np.array([1, 2, 1, 1, 2], dtype=np.int32)
    w = np.array([3.0, 1.0, math.inf, math.nan, 1.0], dtype=np.float32)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        out_i, _, _, _ = aggregate_topdown_semantics(r, c, s, pal, density_size=16, point_conf=w, min_snap_cos=0.9)
    assert not any(issubclass(item.category, RuntimeWarning) for item in caught)
    assert out_i[5, 5] == 1  # +inf dropped; 3*e1 + 1*e2 -> cos = 3/sqrt(10) ≈ 0.9487 > 0.9
    assert out_i[6, 6] == 0  # NaN weight dropped
    assert out_i[7, 7] == 2  # cos = 1.0 > 0.9

    # Test strict threshold pruning (> 0.96 drops 3:1 conflict whose cos = 3 / sqrt(10) ≈ 0.9487)
    out_strict, _, _, _ = aggregate_topdown_semantics(
        np.array([5, 5], dtype=np.int32),
        np.array([5, 5], dtype=np.int32),
        np.array([1, 2], dtype=np.int32),
        pal,
        density_size=16,
        point_conf=np.array([3.0, 1.0], dtype=np.float32),
        min_snap_cos=0.96,
    )
    assert out_strict[5, 5] == 0


def test_load_view_gt_semantics_crop_mode_drops_margin_classes(tmp_path: Path) -> None:
    view_dir = tmp_path / "portrait_view"
    view_dir.mkdir(parents=True)
    raw = np.full((1280, 720), 5, dtype=np.uint8)  # chair in cropped interior
    raw[:150, :] = 1  # wall only in top cropped-out margin (150 * 924 / 1280 = 108 < 203)
    raw[1130:, :] = 8  # door only in bottom cropped-out margin (1130 * 924 / 1280 = 815 > 721)
    Image.fromarray(raw, mode="L").save(view_dir / "semantic.png")

    encoder: Any = _OrthoTextEncoder(dim=16)
    sem_idx, pal, labels = s3d.load_view_gt_semantics(str(view_dir), text_encoder=encoder, mode="crop")
    assert sem_idx.shape == (518, 518)
    assert labels == ["chair"]
    assert pal.shape == (2, 16)
    assert np.all(sem_idx == 1)


def test_prepare_s3d_main_cli_semantic_modes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import pandas as pd

    scene_dir = tmp_path / "s3d" / "scene_00000"
    view_dir = scene_dir / "2D_rendering" / "0" / "perspective" / "full" / "00"
    view_dir.mkdir(parents=True)
    (view_dir / "rgb_rawlight.png").touch()
    (view_dir / "camera_pose.txt").touch()
    (view_dir / "depth.png").touch()

    raw_sem = np.zeros((720, 1280), dtype=np.uint8)
    raw_sem[200:500, 200:600] = 1  # wall
    raw_sem[200:500, 650:950] = 5  # chair
    Image.fromarray(raw_sem, mode="L").save(view_dir / "semantic.png")

    size = 518
    world_points = np.full((size, size, 3), np.nan, dtype=np.float64)
    yy, xx = np.meshgrid(np.arange(112, 406), np.arange(size), indexing="ij")
    world_points[112:406, :, 0] = (xx - 259.0) * 10.0
    world_points[112:406, :, 1] = 2000.0
    world_points[112:406, :, 2] = 500.0 + (yy - 112.0) * 6.0

    cam_pts = np.zeros((1, size, size, 3), dtype=np.float32)
    cam_pts[0, :, :, 0] = np.broadcast_to(np.linspace(-2.0, 2.0, size, dtype=np.float32)[None, :], (size, size))
    cam_pts[0, :, :, 1] = np.broadcast_to(np.linspace(-1.0, 1.0, size, dtype=np.float32)[:, None], (size, size))
    cam_pts[0, :, :, 2] = 3.0 + np.broadcast_to(np.linspace(0.0, 1.0, size, dtype=np.float32)[None, :], (size, size))
    conf = np.ones((1, size, size), dtype=np.float32)
    conf[0, 112:406, 80:440] = 5.0
    recon_stub: Any = {
        "points_cam0": cam_pts,
        "conf": conf,
        "est_gravity_cam": np.array([0.0, 1.0, 0.0], dtype=np.float32),
    }

    plan_img = Image.new("RGB", (512, 512), (0, 0, 0))
    plan_params = ((0.0, 0.0), 1.0, (0, 0))

    stub_encoder = _OrthoTextEncoder(dim=16)
    monkeypatch.setattr(prepare_s3d, "load_models", lambda recon_model="pi3-original": (None, None))
    monkeypatch.setattr(prepare_s3d, "reconstruct", lambda paths, m, g: recon_stub)
    monkeypatch.setattr(prepare_s3d, "ClipTextEncoder", lambda model_name=None: stub_encoder)
    monkeypatch.setattr(s3d, "load_floorplan", lambda d: (plan_img, plan_params))
    monkeypatch.setattr(
        s3d,
        "load_semantic_floorplan",
        lambda d, text_encoder=None, include_rooms=False, plan_params=None: (
            np.ones((512, 512), dtype=np.uint8),
            (text_encoder or stub_encoder).encode_palette(["wall"]),
            ["wall"],
        ),
    )
    monkeypatch.setattr(s3d, "list_views", lambda d: [str(view_dir)])
    monkeypatch.setattr(s3d, "load_world_points", lambda vd: world_points)
    monkeypatch.setattr(
        s3d, "world_to_plan", lambda wp, pp: np.full(wp.shape[:2] + (2,), 100.0, dtype=np.float64)
    )

    def _run_cli(extra_args: list[str], out_name: str) -> pd.DataFrame:
        out_dir = tmp_path / out_name
        monkeypatch.setattr(
            "sys.argv",
            ["prepare_s3d.py", "--s3d_root", str(tmp_path / "s3d"), "--output", str(out_dir), "--scenes", "0", "1"]
            + extra_args,
        )
        prepare_s3d.main()
        return pd.read_csv(out_dir / "scenes_0_1.csv")

    df_gt = _run_cli([], "out_gt")
    assert df_gt.columns.tolist() == ["scene_id", "density", "floorplan", "correspondences", "floorplan_sem", "density_sem"]
    assert len(df_gt) == 2
    for sem_prefix in df_gt["density_sem"].tolist():
        sem_idx, pal, _ = load_semantic_map(str(tmp_path / "out_gt" / str(sem_prefix)))
        assert sem_idx.shape == (512, 512) and pal.shape[1] == 16

    df_struct = _run_cli(["--structural_photo_semantics"], "out_struct")
    import json

    for sem_prefix in df_struct["density_sem"].tolist():
        lbls = json.loads((tmp_path / "out_struct" / f"{sem_prefix}_labels.json").read_text())
        assert set(lbls).issubset({"wall", "door", "window"})

    df_photo_none = _run_cli(["--photo_sem_source", "none"], "out_photo_none")
    assert df_photo_none.columns.tolist() == ["scene_id", "density", "floorplan", "correspondences", "floorplan_sem"]

    df_no_sem = _run_cli(["--no_semantics"], "out_no_sem")
    assert df_no_sem.columns.tolist() == ["scene_id", "density", "floorplan", "correspondences"]

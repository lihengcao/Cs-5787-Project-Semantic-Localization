"""Tests for Workstream 3: factorized patch pooling, SemanticAdapter, joint augmentation, losses, and inference."""
from __future__ import annotations

import csv
import json
import random
import warnings
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
from PIL import Image
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import evaluate_s3d
import train
from scenealigner import structured3d as s3d
from scenealigner.inference import (
    AlignmentFailureError,
    SemanticAdapter,
    align_to_floorplan,
    compute_fused_sim_matrix,
    estimate_correspondences,
    load_semantic_adapter,
    reconstruct_projection_geometry,
    render_density_map,
    sample_descriptors,
    save_semantic_adapter,
)
from scenealigner.semantics import pool_semantic_patches, project_posed_semantics, save_semantic_map
from scenealigner.training import (
    DensityFloorplanDataset,
    TrainingSampleDict,
    collate,
    feature_matching_loss,
    semantic_contrastive_loss,
)


def _reference_pool_dense(
    sem_idx: np.ndarray, palette: np.ndarray, sem_rel: np.ndarray, patch_size: int = 16
) -> tuple[np.ndarray, np.ndarray]:
    """Slow brute-force dense reference implementation ignoring row-0 palette entries."""
    bsz, h, w = sem_idx.shape
    hp, wp = h // patch_size, w // patch_size
    dim = palette.shape[-1]
    out_feat = np.zeros((bsz, dim, hp, wp), dtype=np.float32)
    out_rel = np.zeros((bsz, 1, hp, wp), dtype=np.float32)
    for b in range(bsz):
        pal_b = palette if palette.ndim == 2 else palette[b]
        for py in range(hp):
            for px in range(wp):
                sub_i = sem_idx[b, py * patch_size : (py + 1) * patch_size, px * patch_size : (px + 1) * patch_size]
                sub_r = sem_rel[b, py * patch_size : (py + 1) * patch_size, px * patch_size : (px + 1) * patch_size]
                act = sub_i > 0
                n_act = float(act.sum())
                if n_act == 0:
                    continue
                vecs = pal_b[sub_i[act]].astype(np.float32)
                w_vec = (vecs * sub_r[act, None].astype(np.float32)).sum(axis=0)
                norm = float(np.linalg.norm(w_vec))
                out_rel[b, 0, py, px] = min(1.0, norm / max(1.0, n_act))
                if norm > 1e-6:
                    out_feat[b, :, py, px] = w_vec / norm
    return out_feat, out_rel


def test_pool_semantic_patches_factorized_memory_and_parity() -> None:
    rng = np.random.default_rng(123)
    bsz, h, w, num_pal, dim = 2, 64, 64, 8, 32
    sem_idx = rng.integers(0, num_pal, size=(bsz, h, w), dtype=np.uint8)
    sem_rel = rng.uniform(0.0, 1.0, size=(bsz, h, w)).astype(np.float32)
    palette = rng.normal(size=(bsz, num_pal, dim)).astype(np.float32)
    palette /= np.maximum(np.linalg.norm(palette, axis=-1, keepdims=True), 1e-6)
    palette[:, 0, :] = 9.0  # non-zero row-0 must STILL be ignored for unlabeled sem_idx==0

    ref_feat, ref_rel = _reference_pool_dense(sem_idx, palette, sem_rel, patch_size=16)
    np_feat, np_rel = pool_semantic_patches(sem_idx, palette, sem_rel, patch_size=16)
    assert np.allclose(np_feat, ref_feat, atol=1e-5)
    assert np.allclose(np_rel, ref_rel, atol=1e-5)

    # Non-contiguous PyTorch tensor parity (transpose + flip -> non-contiguous)
    t_idx = torch.from_numpy(sem_idx).transpose(1, 2)
    t_rel = torch.from_numpy(sem_rel).transpose(1, 2)
    t_pal = torch.from_numpy(palette).half()
    t_feat, t_rel_out = pool_semantic_patches(t_idx, t_pal, t_rel, patch_size=16)
    ref_t_feat, ref_t_rel = _reference_pool_dense(
        np.transpose(sem_idx, (0, 2, 1)), palette.astype(np.float16).astype(np.float32), np.transpose(sem_rel, (0, 2, 1)), 16
    )
    assert torch.allclose(t_feat, torch.from_numpy(ref_t_feat), atol=5e-3)
    assert torch.allclose(t_rel_out, torch.from_numpy(ref_t_rel), atol=5e-3)

    # High-resolution 1280x1280 test (would allocate ~13 GB naively for B=4, D=512; fast when factorized)
    big_idx = rng.integers(0, 6, size=(1280, 1280), dtype=np.uint8)
    big_pal = np.zeros((6, 512), dtype=np.float16)
    big_pal[1:] = torch.nn.functional.normalize(torch.randn(5, 512), dim=-1).numpy().astype(np.float16)
    b_feat, b_rel = pool_semantic_patches(big_idx, big_pal, patch_size=16)
    assert b_feat.shape == (512, 80, 80) and b_rel.shape == (1, 80, 80)
    t_big_feat, t_big_rel = pool_semantic_patches(torch.from_numpy(big_idx), torch.from_numpy(big_pal), patch_size=16)
    assert torch.allclose(t_big_feat, torch.from_numpy(b_feat), atol=1e-2)
    assert torch.allclose(t_big_rel, torch.from_numpy(b_rel), atol=1e-2)


def test_semantic_adapter_preserves_zero_patches_and_roundtrips(tmp_path: Path) -> None:
    adapter = SemanticAdapter(in_dim=16, hidden_dim=32, out_dim=8)
    with torch.no_grad():
        for mod in adapter.modules():
            if isinstance(mod, nn.Linear) and mod.bias is not None:
                mod.bias.fill_(0.5)

    x = torch.zeros(2, 4, 16)
    x[0, 1] = torch.nn.functional.normalize(torch.randn(16), dim=0)
    x[1, 3] = torch.nn.functional.normalize(torch.randn(16), dim=0)
    out = adapter(x)
    assert out.shape == (2, 4, 8)
    # Inactive patches must remain exact 0.0 despite non-zero MLP biases
    assert torch.equal(out[0, 0], torch.zeros(8))
    assert torch.equal(out[1, 0], torch.zeros(8))
    assert out[0, 1].norm(p=2).item() == pytest.approx(1.0, abs=1e-5)
    assert out[1, 3].norm(p=2).item() == pytest.approx(1.0, abs=1e-5)

    # Spatial forward_map
    fmap = torch.zeros(1, 16, 4, 4)
    fmap[0, :, 2, 3] = x[0, 1]
    out_map = adapter.forward_map(fmap)
    assert out_map.shape == (1, 8, 4, 4)
    assert torch.equal(out_map[0, :, 0, 0], torch.zeros(8))
    assert torch.allclose(out_map[0, :, 2, 3], out[0, 1], atol=1e-6)

    # Save & load roundtrip (directory and .pt file path) with weights_only=True
    saved_path = save_semantic_adapter(str(tmp_path), adapter, lambda_sem=0.75)
    assert Path(saved_path).name == "semantic_adapter.pt"
    loaded_dir, l_dir = load_semantic_adapter(str(tmp_path), device="cpu")
    loaded_file, l_file = load_semantic_adapter(saved_path, device="cpu")
    assert l_dir == pytest.approx(0.75)
    assert l_file == pytest.approx(0.75)
    assert torch.allclose(loaded_dir(x), out, atol=1e-6)
    assert torch.allclose(loaded_file(x), out, atol=1e-6)


def test_dataset_and_collate_joint_rot90_crop_and_nearest_resize(tmp_path: Path) -> None:
    import pandas as pd

    # 1. Baseline 4-column CSV works cleanly with load_semantics=False and fails clearly with True
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    Image.new("RGB", (64, 64), (255, 255, 255)).save(base_dir / "d.png")
    Image.new("RGB", (64, 64), (255, 255, 255)).save(base_dir / "p.png")
    pts = np.stack([np.tile(np.array([[10.0, 20.0]], dtype=np.float32), (20, 1))] * 2, axis=0)
    np.save(base_dir / "c.npy", pts)
    pd.DataFrame([{"scene_id": "s0", "density": "d.png", "floorplan": "p.png", "correspondences": "c.npy"}]).to_csv(
        base_dir / "train.csv", index=False
    )
    ds_base = DensityFloorplanDataset(str(base_dir), num_points=8, min_points=1, image_size=64, load_semantics=False)
    sample_base = ds_base[0]
    assert set(sample_base.keys()) == {"density_img", "plan_img", "density_pts", "plan_pts", "valid", "scene_id"}
    with pytest.raises(ValueError, match="floorplan_sem"):
        DensityFloorplanDataset(str(base_dir), load_semantics=True)

    # 2. Semantic 6-column CSV with thin 1-px landmark strokes rotated/cropped/resized through collate
    sem_dir = tmp_path / "sem"
    sem_dir.mkdir()
    Image.new("RGB", (512, 512), (255, 255, 255)).save(sem_dir / "den.png")
    Image.new("RGB", (512, 512), (255, 255, 255)).save(sem_dir / "plan.png")

    # Landmark cluster centered at density (x=140, y=200) and plan (x=360, y=140) as 1-px strokes of concept 1 & 2
    pal = np.zeros((3, 16), dtype=np.float16)
    pal[1, 0] = 1.0  # door
    pal[2, 1] = 1.0  # window
    den_idx = np.zeros((512, 512), dtype=np.uint8)
    den_rel = np.zeros((512, 512), dtype=np.float32)
    # Place 1-px cross of concept 1 ("door") inside 16x16 patch containing (140, 200)
    den_idx[200, 130:146] = 1
    den_rel[200, 130:146] = 1.0

    plan_idx = np.zeros((512, 512), dtype=np.uint8)
    # Place 1-px cross of concept 2 ("window") inside 16x16 patch containing (360, 140)
    plan_idx[140, 355:370] = 2

    save_semantic_map(str(sem_dir / "den_sem"), den_idx, pal, sem_rel=den_rel, labels=["door", "window"])
    save_semantic_map(str(sem_dir / "plan_sem"), plan_idx, pal, labels=["door", "window"])

    # Spread correspondences horizontally along the 1-px landmark stroke in both images
    pts_plan = np.stack([np.linspace(356.0, 368.0, 16, dtype=np.float32), np.full(16, 140.0, dtype=np.float32)], axis=1)
    pts_den = np.stack([np.linspace(132.0, 144.0, 16, dtype=np.float32), np.full(16, 200.0, dtype=np.float32)], axis=1)
    np.save(sem_dir / "corr.npy", np.stack([pts_plan, pts_den]))

    pd.DataFrame(
        [
            {
                "scene_id": "s_sem",
                "density": "den.png",
                "floorplan": "plan.png",
                "correspondences": "corr.npy",
                "floorplan_sem": "plan_sem",
                "density_sem": "den_sem",
            }
        ]
    ).to_csv(sem_dir / "train_sem.csv", index=False)

    ds_sem = DensityFloorplanDataset(str(sem_dir), num_points=8, min_points=1, load_semantics=True)
    for seed in range(12):
        random.seed(seed)
        np.random.seed(seed)
        samples: list[TrainingSampleDict] = [ds_sem[0], ds_sem[0]]
        batch = collate(samples, resolutions=[512, 768])
        assert (
            "density_sem_feat" in batch
            and "density_sem_rel" in batch
            and "plan_sem_feat" in batch
            and "plan_sem_rel" in batch
        )
        d_feat = batch["density_sem_feat"]
        d_rel_map = batch["density_sem_rel"]
        p_feat = batch["plan_sem_feat"]
        p_rel_map = batch["plan_sem_rel"]
        valid = batch["valid"]

        # Sample at normalized points with nearest to inspect owning patch exact value
        d_samp = torch.nn.functional.grid_sample(d_feat, batch["density_pts"].unsqueeze(1), mode="nearest", align_corners=False).squeeze(2).permute(0, 2, 1)
        p_samp = torch.nn.functional.grid_sample(p_feat, batch["plan_pts"].unsqueeze(1), mode="nearest", align_corners=False).squeeze(2).permute(0, 2, 1)
        d_r = torch.nn.functional.grid_sample(d_rel_map, batch["density_pts"].unsqueeze(1), mode="nearest", align_corners=False).squeeze(2).permute(0, 2, 1)
        p_r = torch.nn.functional.grid_sample(p_rel_map, batch["plan_pts"].unsqueeze(1), mode="nearest", align_corners=False).squeeze(2).permute(0, 2, 1)

        # Target unit vector e_0 ("door") on density and e_1 ("window") on plan
        assert torch.allclose(d_samp[valid][:, 0], torch.ones_like(d_samp[valid][:, 0]), atol=1e-4)
        assert torch.allclose(p_samp[valid][:, 1], torch.ones_like(p_samp[valid][:, 1]), atol=1e-4)
        # Active-pixel consensus on a single-concept thin stroke must stay 1.0 even after NEAREST resize to 768
        assert torch.allclose(d_r[valid], torch.ones_like(d_r[valid]), atol=1e-4)
        assert torch.allclose(p_r[valid], torch.ones_like(p_r[valid]), atol=1e-4)


def test_compute_fused_sim_matrix_modes_and_symmetry_tiebreaking() -> None:
    desc_geo_d = torch.zeros(1, 2, 8)
    desc_geo_d[0, :, 0] = 1.0
    map_geo_p = torch.zeros(1, 8, 1, 2)
    map_geo_p[0, 0, :, :] = 1.0  # identical geo similarity = 1.0 everywhere

    desc_sem_d = torch.zeros(1, 2, 4)
    desc_sem_d[0, 0, 0] = 1.0  # query 0 = concept 0 ("door")
    desc_sem_d[0, 1, 0] = 0.0  # query 1 = unlabeled zero

    map_sem_p = torch.zeros(1, 4, 1, 2)
    map_sem_p[0, 0, 0, 0] = 1.0  # patch A = "door"
    map_sem_p[0, 1, 0, 1] = 1.0  # patch B = "window" (orthogonal)

    rel_d = torch.tensor([[[0.8], [0.0]]])  # (1, 2, 1)
    map_rel_p = torch.tensor([[[[0.5, 0.9]]]])  # (1, 1, 1, 2)

    s_none = compute_fused_sim_matrix(desc_geo_d, map_geo_p, sem_mode="none")
    assert s_none[0, 0, 0].item() == pytest.approx(s_none[0, 0, 1].item(), abs=1e-6)

    s_uni = compute_fused_sim_matrix(
        desc_geo_d, map_geo_p, sem_mode="uniform", desc_sem_d=desc_sem_d, map_sem_p=map_sem_p, lambda_sem=0.5
    )
    assert s_uni[0, 0, 0].item() - s_uni[0, 0, 1].item() == pytest.approx(0.5, abs=1e-6)

    s_cons = compute_fused_sim_matrix(
        desc_geo_d,
        map_geo_p,
        sem_mode="consensus",
        desc_sem_d=desc_sem_d,
        map_sem_p=map_sem_p,
        rel_d=rel_d,
        map_rel_p=map_rel_p,
        lambda_sem=0.5,
    )
    assert s_cons[0, 0, 0].item() - s_cons[0, 0, 1].item() == pytest.approx(0.5 * 0.8 * 0.5, abs=1e-6)
    # Query 1 has zero reliability -> exact fallback to s_none
    assert torch.allclose(s_cons[:, 1, :], s_none[:, 1, :], atol=1e-7)

    with pytest.raises(ValueError, match="rel"):
        compute_fused_sim_matrix(
            desc_geo_d, map_geo_p, sem_mode="consensus", desc_sem_d=desc_sem_d, map_sem_p=map_sem_p
        )


def test_feature_matching_and_semantic_contrastive_losses() -> None:
    same_scene = torch.zeros(1, 1, dtype=torch.bool)
    valid = torch.tensor([[True, True, True]])
    feat_a = torch.randn(1, 3, 8, requires_grad=True)
    feat_b = torch.randn(1, 3, 8, requires_grad=True)

    adapter = SemanticAdapter(in_dim=8, hidden_dim=16, out_dim=6)
    raw_a = torch.zeros(1, 3, 8)
    raw_b = torch.zeros(1, 3, 8)
    raw_a[0, 0, 0] = 1.0  # wall
    raw_a[0, 1, 0] = 1.0  # wall
    raw_a[0, 2, 1] = 1.0  # door
    raw_b.copy_(raw_a)
    sem_a = adapter(raw_a)
    sem_b = adapter(raw_b)

    # Zero reliability -> exact zero grad on adapter
    zero_rel = torch.zeros(1, 3, 1)
    loss_zero = feature_matching_loss(
        feat_a, feat_b, same_scene, valid, sem_a=sem_a, sem_b=sem_b, rel_a=zero_rel, rel_b=zero_rel, lambda_sem=0.5
    )
    loss_zero.backward()
    for p in adapter.parameters():
        assert p.grad is None or torch.all(p.grad == 0)

    # Active reliability -> non-zero grad on adapter
    adapter.zero_grad()
    sem_a2, sem_b2 = adapter(raw_a), adapter(raw_b)
    act_rel = torch.ones(1, 3, 1)
    loss_act = feature_matching_loss(
        feat_a.detach().requires_grad_(True),
        feat_b.detach().requires_grad_(True),
        same_scene,
        valid,
        sem_a=sem_a2,
        sem_b=sem_b2,
        rel_a=act_rel,
        rel_b=act_rel,
        lambda_sem=0.5,
    )
    loss_act.backward()
    assert any(p.grad is not None and p.grad.abs().sum().item() > 0 for p in adapter.parameters())

    # Product reliability scaling: w_ij = r_i * r_j
    rel_08 = torch.full((1, 3, 1), 0.8)
    rel_05 = torch.full((1, 3, 1), 0.5)
    with torch.no_grad():
        s_a_eval, s_b_eval = adapter(raw_a), adapter(raw_b)
        loss_prod = feature_matching_loss(
            feat_a.detach(), feat_b.detach(), same_scene, valid, sem_a=s_a_eval, sem_b=s_b_eval, rel_a=rel_08, rel_b=rel_05, lambda_sem=0.5
        )
        loss_scaled = feature_matching_loss(
            feat_a.detach(), feat_b.detach(), same_scene, valid, sem_a=s_a_eval, sem_b=s_b_eval, rel_a=act_rel, rel_b=act_rel, lambda_sem=0.5 * 0.8 * 0.5
        )
        assert loss_prod.item() == pytest.approx(loss_scaled.item(), abs=1e-6)

    # Semantic contrastive loss: masks same-concept wall<->wall while remaining finite with single-concept or zero batches
    adapter.zero_grad()
    loss_sem = semantic_contrastive_loss(adapter(raw_a), adapter(raw_b), raw_a, raw_b, valid, rel_a=act_rel, rel_b=act_rel)
    assert torch.isfinite(loss_sem) and loss_sem.item() > 0.0
    loss_sem.backward()

    # Single concept everywhere (only "wall") -> all off-diagonals masked -> finite 0.0 loss (no NaN!)
    raw_single = torch.zeros(1, 3, 8)
    raw_single[:, :, 0] = 1.0
    loss_single = semantic_contrastive_loss(adapter(raw_single), adapter(raw_single), raw_single, raw_single, valid)
    assert torch.isfinite(loss_single) and loss_single.item() == pytest.approx(0.0, abs=1e-7)
    loss_single.backward()

    # All unlabeled zeros -> finite 0.0
    raw_empty = torch.zeros(1, 3, 8)
    loss_empty = semantic_contrastive_loss(adapter(raw_empty), adapter(raw_empty), raw_empty, raw_empty, valid)
    assert torch.isfinite(loss_empty) and loss_empty.item() == pytest.approx(0.0, abs=1e-7)


class _ToyBackbone(nn.Module):
    """Lightweight differentiable CPU backbone matching DINOv3's forward_features output signature."""

    def __init__(self, embed_dim: int = 16) -> None:
        super().__init__()
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=16, stride=16)

    def forward_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.proj(x)
        return {"x_norm_patchtokens": feat.flatten(2).transpose(1, 2)}


class _SymmetricRoomBackbone(nn.Module):
    """Synthetic backbone giving identical geometric features on 180-deg symmetric left/right halves."""

    def __init__(self) -> None:
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # 180-deg invariant spatial descriptor: w_ij * (1, r_norm, cos(2*theta), sin(2*theta)) around grid center
        pooled = torch.nn.functional.avg_pool2d((x.mean(dim=1, keepdim=True) < 0.0).float(), kernel_size=16, stride=16)
        B, _, H, W = pooled.shape
        cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
        dy = torch.arange(H, device=x.device, dtype=torch.float32) - cy
        dx = torch.arange(W, device=x.device, dtype=torch.float32) - cx
        gy, gx = torch.meshgrid(dy, dx, indexing="ij")
        r_norm = torch.sqrt(gx * gx + gy * gy) / max(cx, cy, 1.0)
        theta = torch.atan2(gy, gx)
        wall = (pooled.flatten(1) > 0.01).float()

        tokens = torch.zeros(B, H * W, 8, device=x.device)
        tokens[..., 0] = 0.2
        tokens[..., 1] = wall * 2.0
        tokens[..., 2] = wall * r_norm.flatten()[None, :] * 2.0
        tokens[..., 3] = wall * torch.cos(2.0 * theta).flatten()[None, :] * 2.0
        tokens[..., 4] = wall * torch.sin(2.0 * theta).flatten()[None, :] * 2.0
        return {"x_norm_patchtokens": tokens}


def test_train_compute_loss_checkpoint_resume_and_inference_180_symmetry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _ToyBackbone(embed_dim=16)
    adapter = SemanticAdapter(in_dim=16, hidden_dim=16, out_dim=8)
    batch: Any = {
        "density_img": torch.randn(2, 3, 64, 64),
        "plan_img": torch.randn(2, 3, 64, 64),
        "density_pts": torch.rand(2, 12, 2) * 2 - 1,
        "plan_pts": torch.rand(2, 12, 2) * 2 - 1,
        "valid": torch.ones(2, 12, dtype=torch.bool),
        "scene_id": ["scene_a", "scene_b"],
        "density_sem_feat": torch.nn.functional.normalize(torch.randn(2, 16, 4, 4), dim=1),
        "density_sem_rel": torch.ones(2, 1, 4, 4),
        "plan_sem_feat": torch.nn.functional.normalize(torch.randn(2, 16, 4, 4), dim=1),
        "plan_sem_rel": torch.ones(2, 1, 4, 4),
    }

    for step in (0, 15, 25):
        model.zero_grad()
        adapter.zero_grad()
        loss, losses = train.compute_loss(
            model, batch, step=step, total_steps=100, sem_mode="consensus", semantic_adapter=adapter, lambda_sem=0.5, alpha_sem=0.2
        )
        assert torch.isfinite(loss)
        loss.backward()
        assert "feat" in losses and "sem" in losses
        if step >= 15:
            assert "regr" in losses
        if step >= 25:
            assert "geo" in losses and "topo" in losses

    # Save checkpoint + semantic_adapter.pt roundtrip
    model_with_save: Any = model
    model_with_save.save_pretrained = lambda out_dir: None  # stub PEFT save_pretrained
    monkeypatch.setattr(train, "get_peft_model_state_dict", lambda m: {"proj.weight": model.proj.weight.detach().clone()})
    opt = torch.optim.AdamW(list(model.parameters()) + list(adapter.parameters()), lr=1e-4)
    train.save_checkpoint(str(tmp_path), model_with_save, opt, step=25, semantic_adapter=adapter, lambda_sem=0.5)
    ckpt = torch.load(tmp_path / "checkpoint.pt", map_location="cpu", weights_only=True)
    assert ckpt["step"] == 25 and "semantic_adapter" in ckpt and (tmp_path / "semantic_adapter.pt").is_file()

    # 180-deg symmetric room inference tie-breaking test
    res = 512
    # Construct identity-projection adapter mapping CLIP dim 0 ("door") and dim 1 ("window") cleanly
    sym_adapter = SemanticAdapter(in_dim=4, hidden_dim=8, out_dim=4)
    with torch.no_grad():
        lin0 = sym_adapter.net[0]
        ln = sym_adapter.net[1]
        lin1 = sym_adapter.net[3]
        assert isinstance(lin0, nn.Linear) and isinstance(ln, nn.LayerNorm) and isinstance(lin1, nn.Linear)
        # Map e_0 -> +axis0, e_1 -> +axis1 linearly without relu killing sign
        lin0.weight.zero_()
        lin0.bias.fill_(2.0)  # stay in positive linear region of GELU
        lin0.weight[0, 0] = 3.0
        lin0.weight[1, 1] = 3.0
        ln.weight.fill_(1.0)
        ln.bias.zero_()
        lin1.weight.zero_()
        lin1.bias.zero_()
        lin1.weight[0, 0] = 1.0
        lin1.weight[1, 1] = 1.0

    # Symmetrically placed North/South wall bars on a 512x512 floorplan (top row ~ 120, bottom row ~ 392)
    arr_plan = np.full((res, res, 3), 255, dtype=np.uint8)
    arr_plan[115:130, 128:384] = 0  # North wall ("door")
    arr_plan[382:397, 128:384] = 0  # South wall ("window")
    plan_img = Image.fromarray(arr_plan)
    den_img = Image.fromarray(arr_plan)  # identical 180-deg symmetric geometry before labeling

    pal = np.zeros((3, 4), dtype=np.float16)
    pal[1, 0] = 1.0  # door
    pal[2, 1] = 1.0  # window
    plan_sem_idx = np.zeros((res, res), dtype=np.uint8)
    plan_sem_idx[115:130, 128:384] = 1
    plan_sem_idx[382:397, 128:384] = 2

    # Same orientation (door=North, window=South)
    den_sem_idx = plan_sem_idx.copy()
    den_rel = (den_sem_idx > 0).astype(np.float32)

    np.random.seed(0)
    src_px, dst_px = estimate_correspondences(
        _SymmetricRoomBackbone(),
        den_img,
        plan_img,
        sem_mode="consensus",
        semantic_adapter=sym_adapter,
        density_sem=(den_sem_idx, pal, den_rel),
        plan_sem=(plan_sem_idx, pal, None),
        lambda_sem=1.0,
    )
    # Semantic consensus ensures North density pixels ("door") map to North plan y < 256 (not 180-deg flipped y > 256)
    north_src = src_px[:, 1] < 256
    south_src = src_px[:, 1] > 256
    assert north_src.any() and south_src.any()
    assert dst_px[north_src, 1].mean() < 200.0
    assert dst_px[south_src, 1].mean() > 312.0

    # End-to-end align_to_floorplan 180-deg symmetry test on a synthetic 3D reconstruction
    # Build (1, 64, 64, 3) grid: rows 0..31 low-conf padding, rows 32..47 North wall (z ~ +2), rows 48..63 South wall (z ~ -2)
    pts_grid = np.zeros((1, 64, 64, 3), dtype=np.float32)
    conf_grid = np.full((1, 64, 64), 0.1, dtype=np.float32)
    photo_idx_same = np.zeros((1, 64, 64), dtype=np.uint8)
    photo_idx_swap = np.zeros((1, 64, 64), dtype=np.uint8)

    xs = np.linspace(-2.0, 2.0, 64, dtype=np.float32)
    ys = np.linspace(0.0, 1.0, 16, dtype=np.float32)
    for r_local, y_val in enumerate(ys):
        z_offset = (r_local / 15.0 - 0.5) * 0.1
        pts_grid[0, 32 + r_local, :, 0] = xs
        pts_grid[0, 32 + r_local, :, 1] = y_val
        pts_grid[0, 32 + r_local, :, 2] = 2.0 + z_offset
        pts_grid[0, 48 + r_local, :, 0] = xs
        pts_grid[0, 48 + r_local, :, 1] = y_val
        pts_grid[0, 48 + r_local, :, 2] = -2.0 + z_offset
    conf_grid[0, 32:, :] = 1.0
    photo_idx_same[0, 32:48, :] = 1  # North = door
    photo_idx_same[0, 48:64, :] = 2  # South = window
    photo_idx_swap[0, 32:48, :] = 2  # Swapped: North = window
    photo_idx_swap[0, 48:64, :] = 1  # Swapped: South = door

    recon_sym = {
        "points_cam0": pts_grid,
        "conf": conf_grid,
        "est_gravity_cam": np.array([0.0, 1.0, 0.0], dtype=np.float32),
    }
    proj_ref, pts_ref, mask_ref = reconstruct_projection_geometry(pts_grid, conf_grid, recon_sym["est_gravity_cam"])
    plan_sym_img = render_density_map(pts_ref, proj_ref)
    plan_s_idx, _, plan_s_pal, _ = project_posed_semantics(
        pts_ref, mask_ref, photo_idx_same, pal, proj_ref, conf_grid=conf_grid, training_rounding=False
    )

    sym_backbone = _SymmetricRoomBackbone()
    np.random.seed(0)
    aln_same = align_to_floorplan(
        recon_sym,
        plan_sym_img,
        sym_backbone,
        sem_mode="consensus",
        semantic_adapter=sym_adapter,
        photo_sem=(photo_idx_same, pal, conf_grid),
        plan_sem=(plan_s_idx, plan_s_pal, None),
        lambda_sem=1.0,
    )
    assert aln_same["M"][0, 0] > 0.8 and aln_same["M"][1, 1] > 0.8

    np.random.seed(0)
    aln_swap = align_to_floorplan(
        recon_sym,
        plan_sym_img,
        sym_backbone,
        sem_mode="consensus",
        semantic_adapter=sym_adapter,
        photo_sem=(photo_idx_swap, pal, conf_grid),
        plan_sem=(plan_s_idx, plan_s_pal, None),
        lambda_sem=1.0,
    )
    assert aln_swap["M"][0, 0] < -0.8 and aln_swap["M"][1, 1] < -0.8


def test_evaluate_s3d_cli_sem_modes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    scene_dir = tmp_path / "s3d" / "scene_03250"
    view_dir = scene_dir / "2D_rendering" / "0" / "perspective" / "full" / "00"
    view_dir.mkdir(parents=True)
    for fn in ("rgb_rawlight.png", "depth.png"):
        (view_dir / fn).touch()
    (view_dir / "camera_pose.txt").write_text(" ".join(["0.0"] * 11))

    recon_stub: Any = {
        "points_cam0": np.zeros((1, 518, 518, 3), dtype=np.float32),
        "conf": np.ones((1, 518, 518), dtype=np.float32),
        "est_gravity_cam": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        "extrinsics": np.eye(4, dtype=np.float32)[None, ...],
    }
    plan_img = Image.new("RGB", (512, 512), (255, 255, 255))
    plan_params = ((0.0, 0.0), 1.0, (0, 0))
    calls: list[dict[str, Any]] = []

    monkeypatch.setattr(evaluate_s3d.s3d, "TEST_SCENES", range(3250, 3251))
    monkeypatch.setattr(evaluate_s3d, "load_models", lambda recon_model="pi3-original": (None, None))
    monkeypatch.setattr(evaluate_s3d, "load_model", lambda ckpt, dino=None: object())
    monkeypatch.setattr(evaluate_s3d, "reconstruct", lambda paths, m, g: recon_stub)
    monkeypatch.setattr(evaluate_s3d, "pose_errors", lambda recon, aln, pp, vd: (0.25, 5.0))
    monkeypatch.setattr(evaluate_s3d.s3d, "load_floorplan", lambda d: (plan_img, plan_params))
    monkeypatch.setattr(evaluate_s3d.s3d, "list_views", lambda d: [str(view_dir)])
    monkeypatch.setattr(
        evaluate_s3d.s3d,
        "load_semantic_floorplan",
        lambda d, text_encoder=None, include_rooms=False, plan_params=None: (
            np.zeros((512, 512), dtype=np.uint8),
            np.zeros((1, 16), dtype=np.float16),
            [],
        ),
    )
    monkeypatch.setattr(
        evaluate_s3d.s3d,
        "load_view_gt_semantics",
        lambda vd, text_encoder=None, allowed_labels=None: (
            np.zeros((518, 518), dtype=np.uint8),
            np.zeros((1, 16), dtype=np.float16),
            [],
        ),
    )
    monkeypatch.setattr(
        evaluate_s3d,
        "ClipTextEncoder",
        lambda model_name=None: object(),
    )
    monkeypatch.setattr(
        evaluate_s3d,
        "load_semantic_adapter",
        lambda p, device="cpu": (SemanticAdapter(in_dim=16, hidden_dim=16, out_dim=8), 0.5),
    )

    def _fake_align(recon: Any, plan: Any, model: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {"proj_params": {"density_size": 512}, "M": np.eye(2, 3)}

    monkeypatch.setattr(evaluate_s3d, "align_to_floorplan", _fake_align)

    # 1. Baseline mode: --sem_mode none must never require semantic adapter or CLIP encoder
    def _fail_adapter(*a: Any, **kw: Any) -> Any:
        raise AssertionError("load_semantic_adapter should not be called in --sem_mode none")

    monkeypatch.setattr(evaluate_s3d, "load_semantic_adapter", _fail_adapter)
    out_none = tmp_path / "res_none.json"
    monkeypatch.setattr("sys.argv", ["evaluate_s3d.py", "--s3d_root", str(tmp_path / "s3d"), "--output", str(out_none)])
    evaluate_s3d.main()
    assert json.loads(out_none.read_text())["Pos@0.5m"] == pytest.approx(100.0)
    assert calls[-1].get("sem_mode", "none") == "none"

    # 2. Consensus + GT photo semantics mode
    monkeypatch.setattr(
        evaluate_s3d,
        "load_semantic_adapter",
        lambda p, device="cpu": (SemanticAdapter(in_dim=16, hidden_dim=16, out_dim=8), 0.5),
    )
    out_cons = tmp_path / "res_cons.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate_s3d.py",
            "--s3d_root",
            str(tmp_path / "s3d"),
            "--checkpoint_dir",
            str(tmp_path / "ckpt"),
            "--sem_mode",
            "consensus",
            "--photo_sem_source",
            "gt",
            "--output",
            str(out_cons),
        ],
    )
    evaluate_s3d.main()
    assert calls[-1]["sem_mode"] == "consensus"
    assert calls[-1]["photo_sem"] is not None and calls[-1]["plan_sem"] is not None
    assert calls[-1]["photo_sem"][0].shape == (1, 518, 518)

    # 3. --photo_sem_source none disables semantics and runs baseline alignment without adapter
    monkeypatch.setattr(evaluate_s3d, "load_semantic_adapter", _fail_adapter)
    out_no_photo = tmp_path / "res_no_photo.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate_s3d.py",
            "--s3d_root",
            str(tmp_path / "s3d"),
            "--checkpoint_dir",
            str(tmp_path / "ckpt"),
            "--sem_mode",
            "uniform",
            "--photo_sem_source",
            "none",
            "--output",
            str(out_no_photo),
        ],
    )
    evaluate_s3d.main()
    assert calls[-1].get("sem_mode", "none") == "none"
    assert "photo_sem" not in calls[-1]

    # 4. Stage-1 adapter checkpoint (saved_lambda=0.0) falls back to 0.5 unless --lambda_sem is given
    monkeypatch.setattr(
        evaluate_s3d,
        "load_semantic_adapter",
        lambda p, device="cpu": (SemanticAdapter(in_dim=16, hidden_dim=16, out_dim=8), 0.0),
    )
    out_default_lam = tmp_path / "res_default_lam.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate_s3d.py",
            "--s3d_root",
            str(tmp_path / "s3d"),
            "--checkpoint_dir",
            str(tmp_path / "ckpt"),
            "--sem_mode",
            "consensus",
            "--output",
            str(out_default_lam),
        ],
    )
    evaluate_s3d.main()
    assert calls[-1]["lambda_sem"] == pytest.approx(0.5)

    out_override_lam = tmp_path / "res_override_lam.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate_s3d.py",
            "--s3d_root",
            str(tmp_path / "s3d"),
            "--checkpoint_dir",
            str(tmp_path / "ckpt"),
            "--sem_mode",
            "consensus",
            "--lambda_sem",
            "0.25",
            "--output",
            str(out_override_lam),
        ],
    )
    evaluate_s3d.main()
    assert calls[-1]["lambda_sem"] == pytest.approx(0.25)

    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate_s3d.py",
            "--s3d_root",
            str(tmp_path / "s3d"),
            "--sem_mode",
            "consensus",
            "--lambda_sem",
            "-0.1",
            "--output",
            str(tmp_path / "res_neg_lam.json"),
        ],
    )
    with pytest.raises(ValueError, match="lambda_sem"):
        evaluate_s3d.main()

    # 5. --include_rooms and --structural_photo_semantics CLI flags are forwarded
    room_flags: list[bool] = []
    photo_allowed: list[Any] = []
    monkeypatch.setattr(
        evaluate_s3d.s3d,
        "load_semantic_floorplan",
        lambda d, text_encoder=None, include_rooms=False, plan_params=None: (
            room_flags.append(include_rooms)
            or (np.zeros((512, 512), dtype=np.uint8), np.zeros((1, 16), dtype=np.float16), [])
        ),
    )
    monkeypatch.setattr(
        evaluate_s3d.s3d,
        "load_view_gt_semantics",
        lambda vd, text_encoder=None, allowed_labels=None: (
            photo_allowed.append(allowed_labels)
            or (np.zeros((518, 518), dtype=np.uint8), np.zeros((1, 16), dtype=np.float16), [])
        ),
    )
    out_flags = tmp_path / "res_flags.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate_s3d.py",
            "--s3d_root",
            str(tmp_path / "s3d"),
            "--checkpoint_dir",
            str(tmp_path / "ckpt"),
            "--sem_mode",
            "consensus",
            "--include_rooms",
            "--structural_photo_semantics",
            "--output",
            str(out_flags),
        ],
    )
    evaluate_s3d.main()
    assert json.loads(out_flags.read_text())["Pos@0.5m"] == pytest.approx(100.0)
    assert room_flags[-1] is True
    assert photo_allowed[-1] == s3d.STRUCTURAL_PHOTO_LABELS

    # 6. AlignmentFailureError triggers identity-matrix fallback instead of crashing
    seen_alignments: list[dict[str, Any]] = []
    monkeypatch.setattr(
        evaluate_s3d,
        "reconstruct_projection_geometry",
        lambda *a, **kw: ({"density_size": 512}, np.zeros((1, 3), dtype=np.float32), np.ones((1, 518, 518), dtype=bool)),
    )
    monkeypatch.setattr(
        evaluate_s3d,
        "pose_errors",
        lambda recon, aln, pp, vd: (seen_alignments.append(aln) or (0.0, 0.0)),
    )

    def _raise_align_fail(*a: Any, **kw: Any) -> Any:
        raise AlignmentFailureError("RANSAC failed")

    monkeypatch.setattr(evaluate_s3d, "align_to_floorplan", _raise_align_fail)
    out_align_fail = tmp_path / "res_align_fail.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate_s3d.py",
            "--s3d_root",
            str(tmp_path / "s3d"),
            "--checkpoint_dir",
            str(tmp_path / "ckpt"),
            "--sem_mode",
            "consensus",
            "--output",
            str(out_align_fail),
        ],
    )
    evaluate_s3d.main()
    assert out_align_fail.exists()
    assert seen_alignments and seen_alignments[-1]["M"].shape == (2, 3)
    assert np.allclose(seen_alignments[-1]["M"], np.eye(2, 3))

    # 7. Non-RANSAC ValueError must propagate rather than being swallowed as alignment failure
    def _raise_config_err(*a: Any, **kw: Any) -> Any:
        raise ValueError("bad semantic config")

    monkeypatch.setattr(evaluate_s3d, "align_to_floorplan", _raise_config_err)
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate_s3d.py",
            "--s3d_root",
            str(tmp_path / "s3d"),
            "--checkpoint_dir",
            str(tmp_path / "ckpt"),
            "--sem_mode",
            "consensus",
            "--output",
            str(tmp_path / "res_err.json"),
        ],
    )
    with pytest.raises(ValueError, match="bad semantic config"):
        evaluate_s3d.main()


def test_pool_semantic_patches_validation_errors() -> None:
    pal_np = np.zeros((3, 8), dtype=np.float32)
    idx_oob = np.zeros((16, 16), dtype=np.uint8)
    idx_oob[0, 0] = 3
    with pytest.raises(ValueError, match="sem_idx"):
        pool_semantic_patches(idx_oob, pal_np)

    idx_ok = np.zeros((16, 16), dtype=np.uint8)
    rel_bad = np.ones((8, 8), dtype=np.float32)
    with pytest.raises(ValueError, match="shape"):
        pool_semantic_patches(idx_ok, pal_np, sem_rel=rel_bad)

    with pytest.raises(ValueError, match="patch_size"):
        pool_semantic_patches(idx_ok, pal_np, patch_size=0)

    idx_t = torch.zeros((1, 16, 16), dtype=torch.long)
    pal_batch_mismatch = torch.zeros((2, 3, 8), dtype=torch.float32)
    with pytest.raises(ValueError, match="batch|shape"):
        pool_semantic_patches(idx_t, pal_batch_mismatch)

    with pytest.raises(TypeError):
        pool_semantic_patches(idx_ok, cast(Any, torch.zeros((3, 8), dtype=torch.float32)))


def test_compute_fused_sim_matrix_and_estimate_corr_errors() -> None:
    desc_geo_d = torch.randn(1, 16, 8)
    map_geo_p = torch.randn(1, 8, 4, 6)

    with pytest.raises(ValueError, match="sem_mode"):
        compute_fused_sim_matrix(desc_geo_d, map_geo_p, sem_mode=cast(Any, "invalid"))

    with pytest.raises(ValueError, match="lambda_sem"):
        compute_fused_sim_matrix(desc_geo_d, map_geo_p, sem_mode="none", lambda_sem=-0.5)

    class _DummyFeatModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.dummy = nn.Parameter(torch.zeros(1))

        def forward_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
            b, _, h, w = x.shape
            tokens = torch.randn(b, (h // 16) * (w // 16), 8, device=x.device)
            return {"x_norm_patchtokens": tokens}

    model = _DummyFeatModel()
    adapter = SemanticAdapter(in_dim=16, hidden_dim=16, out_dim=8)
    den_img = Image.new("RGB", (64, 64), (0, 0, 0))
    plan_img = Image.new("RGB", (96, 96), (255, 255, 255))
    pal = np.zeros((2, 16), dtype=np.float16)
    pal[1, 0] = 1.0
    den_sem = (np.ones((64, 64), dtype=np.uint8), pal, np.ones((64, 64), dtype=np.float32))
    plan_sem = (np.ones((96, 96), dtype=np.uint8), pal, None)

    with pytest.raises(ValueError):
        estimate_correspondences(
            model,
            den_img,
            plan_img,
            sem_mode="consensus",
            semantic_adapter=adapter,
            density_sem=den_sem,
            plan_sem=None,
        )

    with pytest.raises(ValueError):
        estimate_correspondences(
            model,
            den_img,
            plan_img,
            sem_mode="consensus",
            semantic_adapter=adapter,
            density_sem=None,
            plan_sem=plan_sem,
        )

    with pytest.raises(ValueError):
        estimate_correspondences(
            model,
            den_img,
            plan_img,
            sem_mode="consensus",
            semantic_adapter=None,
            density_sem=den_sem,
            plan_sem=plan_sem,
        )

    adapter.train()
    assert adapter.training is True
    pts_d, pts_p = estimate_correspondences(
        model,
        den_img,
        plan_img,
        sem_mode="consensus",
        semantic_adapter=adapter,
        density_sem=den_sem,
        plan_sem=plan_sem,
    )
    assert adapter.training is True
    assert pts_d.shape == pts_p.shape and pts_d.shape[1] == 2
    assert np.all(pts_d >= 0) and np.all(pts_d < 512)
    assert np.all(pts_p >= 0) and np.all(pts_p < 512)

    adapter.eval()
    estimate_correspondences(
        model,
        den_img,
        plan_img,
        sem_mode="uniform",
        semantic_adapter=adapter,
        density_sem=den_sem,
        plan_sem=plan_sem,
    )
    assert adapter.training is False


def test_align_to_floorplan_failure_and_empty_valid_loss(monkeypatch: pytest.MonkeyPatch) -> None:
    # Empty valid mask in feature_matching_loss returns zero scalar without NaNs (both with and without semantics)
    feat_d = torch.randn(2, 4, 8, requires_grad=True)
    feat_p = torch.randn(2, 4, 8, requires_grad=True)
    same_scene = torch.zeros((2, 2), dtype=torch.bool)
    valid = torch.zeros((2, 4), dtype=torch.bool)
    loss_geo = feature_matching_loss(feat_d, feat_p, same_scene, valid)
    assert loss_geo.item() == pytest.approx(0.0)
    assert torch.isfinite(loss_geo)

    sem_a = torch.randn(2, 4, 8, requires_grad=True)
    sem_b = torch.randn(2, 4, 8, requires_grad=True)
    loss_sem = feature_matching_loss(feat_d, feat_p, same_scene, valid, sem_a=sem_a, sem_b=sem_b, lambda_sem=0.5)
    assert loss_sem.item() == pytest.approx(0.0)
    assert torch.isfinite(loss_sem)
    loss_sem.backward()

    class _DummyFeatModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.p = nn.Parameter(torch.zeros(1))

        def forward_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
            b, _, h, w = x.shape
            return {"x_norm_patchtokens": torch.randn(b, (h // 16) * (w // 16), 8, device=x.device)}

    monkeypatch.setattr(cv2, "estimateAffinePartial2D", lambda *a, **kw: (None, None))
    gx, gz = np.meshgrid(np.linspace(-2.0, 2.0, 16, dtype=np.float32), np.linspace(-2.0, 2.0, 16, dtype=np.float32))
    gy = np.linspace(-1.0, 1.0, 256, dtype=np.float32).reshape(16, 16)
    pts_cam0 = np.stack([gx, gy, gz], axis=-1)[None, ...]
    recon: dict[str, Any] = {
        "points_cam0": pts_cam0,
        "conf": np.linspace(0.1, 1.0, 256, dtype=np.float32).reshape(1, 16, 16),
        "est_gravity_cam": np.array([0.0, 1.0, 0.0], dtype=np.float32),
    }
    plan_img = Image.new("RGB", (64, 64), (255, 255, 255))
    with pytest.raises(AlignmentFailureError):
        align_to_floorplan(recon, plan_img, _DummyFeatModel())


def test_dataset_single_res_boundary_rotations_and_rng_invariance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ds_dir = tmp_path / "ds_rot"
    ds_dir.mkdir()
    Image.fromarray(np.full((512, 512, 3), 128, dtype=np.uint8)).save(ds_dir / "den.png")
    Image.fromarray(np.full((512, 512, 3), 200, dtype=np.uint8)).save(ds_dir / "plan.png")

    pal = np.zeros((2, 16), dtype=np.float16)
    pal[1, 0] = 1.0
    den_idx = np.zeros((512, 512), dtype=np.uint8)
    den_rel = np.zeros((512, 512), dtype=np.float32)
    # 1-px stroke flush on top boundary y=0
    den_idx[0, 128:160] = 1
    den_rel[0, 128:160] = 1.0

    plan_idx = np.zeros((512, 512), dtype=np.uint8)
    plan_idx[0, 256:288] = 1

    save_semantic_map(str(ds_dir / "den_sem"), den_idx, pal, sem_rel=den_rel, labels=["wall"])
    save_semantic_map(str(ds_dir / "plan_sem"), plan_idx, pal, labels=["wall"])

    pts_plan = np.stack([np.linspace(258.0, 286.0, 8, dtype=np.float32), np.zeros(8, dtype=np.float32)], axis=1)
    pts_den = np.stack([np.linspace(130.0, 158.0, 8, dtype=np.float32), np.zeros(8, dtype=np.float32)], axis=1)
    np.save(ds_dir / "corr.npy", np.stack([pts_plan, pts_den]))

    sem_dir = tmp_path / "csv_sem"
    sem_dir.mkdir()
    with (sem_dir / "train.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scene_id", "density", "floorplan", "correspondences", "density_sem", "floorplan_sem"])
        w.writerow(["s0", "../ds_rot/den.png", "../ds_rot/plan.png", "../ds_rot/corr.npy", "../ds_rot/den_sem", "../ds_rot/plan_sem"])

    plain_dir = tmp_path / "csv_plain"
    plain_dir.mkdir()
    with (plain_dir / "train.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scene_id", "density", "floorplan", "correspondences"])
        w.writerow(["s0", "../ds_rot/den.png", "../ds_rot/plan.png", "../ds_rot/corr.npy"])

    # Baseline RNG-sequence invariance with load_semantics=False
    ds_plain = DensityFloorplanDataset(str(plain_dir), num_points=8, min_points=0, image_size=512, load_semantics=False)
    ds_sem_off = DensityFloorplanDataset(str(sem_dir), num_points=8, min_points=0, image_size=512, load_semantics=False)

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    s_plain = [ds_plain[0] for _ in range(4)]

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    s_off = [ds_sem_off[0] for _ in range(4)]

    for a, b in zip(s_plain, s_off):
        assert torch.equal(a["density_img"], b["density_img"])
        assert torch.equal(a["plan_img"], b["plan_img"])
        assert torch.allclose(a["plan_pts"], b["plan_pts"])
        assert torch.allclose(a["density_pts"], b["density_pts"])
        assert torch.equal(a["valid"], b["valid"])

    # Boundary-flush 1-px stroke alignment across all 4 rotations (k in 0..3) at single resolution 512
    ds_sem_on = DensityFloorplanDataset(str(sem_dir), num_points=8, min_points=0, image_size=512, load_semantics=True)
    monkeypatch.setattr(ds_sem_on, "_sample_crop", lambda *a, **kw: None)
    orig_np_randint = np.random.randint
    for k in range(4):
        def _fixed_rot(low: int, high: int | None = None, *a: Any, _k: int = k, **kw: Any) -> Any:
            if low == 0 and high == 4 and not a and not kw:
                return _k
            return orig_np_randint(low, high, *a, **kw)

        monkeypatch.setattr(np.random, "randint", _fixed_rot)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", UserWarning)
            sample = ds_sem_on[0]
        assert not [w for w in caught if issubclass(w.category, UserWarning)], f"Unexpected UserWarnings: {caught}"
        assert "density_sem_idx" in sample and "density_sem_rel" in sample and "density_sem_pal" in sample
        den_s_idx = sample["density_sem_idx"]
        den_s_rel = sample["density_sem_rel"]
        den_s_pal = sample["density_sem_pal"]
        pts_norm = sample["density_pts"][sample["valid"]].numpy()
        px = np.clip(np.round((pts_norm[:, 0] + 1.0) * 256.0 - 0.5).astype(int), 0, 511)
        py = np.clip(np.round((pts_norm[:, 1] + 1.0) * 256.0 - 0.5).astype(int), 0, 511)
        assert torch.all(den_s_idx[py, px] == 1), f"Pixel-aligned rotation failed for k={k}"
        pooled_d, rel_d = pool_semantic_patches(den_s_idx.unsqueeze(0), den_s_pal, sem_rel=den_s_rel.unsqueeze(0), patch_size=16)
        assert pooled_d.shape == (1, 16, 32, 32)
        assert torch.all(rel_d[0, 0, py // 16, px // 16] > 0)


def test_train_cli_validation_and_stage2_freeze_lora_and_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(train, "CLIP_EMBED_DIM", 16)
    ds_dir = tmp_path / "train_ds"
    ds_dir.mkdir()
    Image.fromarray(np.full((64, 64, 3), 100, dtype=np.uint8)).save(ds_dir / "den.png")
    Image.fromarray(np.full((64, 64, 3), 180, dtype=np.uint8)).save(ds_dir / "plan.png")

    pal = np.zeros((3, 16), dtype=np.float16)
    pal[1, 0] = 1.0
    pal[2, 1] = 1.0
    den_idx = np.zeros((64, 64), dtype=np.uint8)
    den_idx[16:32, 16:32] = 1
    den_idx[32:48, 32:48] = 2
    den_rel = np.zeros((64, 64), dtype=np.float32)
    den_rel[16:32, 16:32] = 1.0
    den_rel[32:48, 32:48] = 1.0

    plan_idx = np.zeros((64, 64), dtype=np.uint8)
    plan_idx[32:48, 32:48] = 1
    plan_idx[16:32, 16:32] = 2

    save_semantic_map(str(ds_dir / "den_sem"), den_idx, pal, sem_rel=den_rel, labels=["door", "window"])
    save_semantic_map(str(ds_dir / "plan_sem"), plan_idx, pal, labels=["door", "window"])
    c1_p = np.stack([np.linspace(34.0, 46.0, 8, dtype=np.float32), np.full(8, 40.0, dtype=np.float32)], axis=1)
    c1_d = np.stack([np.linspace(18.0, 30.0, 8, dtype=np.float32), np.full(8, 24.0, dtype=np.float32)], axis=1)
    c2_p = np.stack([np.linspace(18.0, 30.0, 8, dtype=np.float32), np.full(8, 24.0, dtype=np.float32)], axis=1)
    c2_d = np.stack([np.linspace(34.0, 46.0, 8, dtype=np.float32), np.full(8, 40.0, dtype=np.float32)], axis=1)
    pts_plan = np.concatenate([c1_p, c2_p], axis=0)
    pts_den = np.concatenate([c1_d, c2_d], axis=0)
    np.save(ds_dir / "corr.npy", np.stack([pts_plan, pts_den]))

    with (ds_dir / "train.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scene_id", "density", "floorplan", "correspondences", "density_sem", "floorplan_sem"])
        for i in range(4):
            w.writerow([f"s{i}", "den.png", "plan.png", "corr.npy", "den_sem", "plan_sem"])

    val_out = tmp_path / "val_out"
    monkeypatch.setattr(
        "sys.argv",
        ["train.py", "--data", str(ds_dir), "--output", str(val_out), "--sem_mode", "consensus", "--lambda_sem", "-0.1"],
    )
    with pytest.raises(ValueError, match="lambda_sem"):
        train.main()

    monkeypatch.setattr(
        "sys.argv",
        [
            "train.py",
            "--data",
            str(ds_dir),
            "--output",
            str(val_out),
            "--sem_mode",
            "consensus",
            "--lambda_sem",
            "0.0",
            "--alpha_sem",
            "0.0",
        ],
    )
    with pytest.raises(ValueError, match="lambda_sem|alpha_sem|supervise"):
        train.main()

    monkeypatch.setattr(
        "sys.argv",
        ["train.py", "--data", str(ds_dir), "--output", str(val_out), "--resolutions", "520"],
    )
    with pytest.raises(ValueError, match="16"):
        train.main()

    class _FakeBackbone(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(3, 8)
            self.lora_weight = nn.Parameter(torch.ones(8, 8))

        def forward_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
            b, _, h, w = x.shape
            pooled = torch.nn.functional.avg_pool2d(x, kernel_size=16, stride=16)
            tokens = pooled.flatten(2).transpose(1, 2)
            out = self.proj(tokens) @ self.lora_weight
            return {"x_norm_patchtokens": out.reshape(b, (h // 16) * (w // 16), 8)}

        def save_pretrained(self, output_dir: str) -> None:
            torch.save({"lora_weight": self.lora_weight.detach().clone()}, Path(output_dir) / "adapter_model.pt")

    init_lora_dir = tmp_path / "stage1_lora"
    init_lora_dir.mkdir()
    torch.save({"lora_weight": torch.full((8, 8), 2.0)}, init_lora_dir / "adapter_model.pt")

    captured_models: list[_FakeBackbone] = []

    def _fake_load_model(ckpt_dir: str = "pre-trained", dino: str | None = None) -> _FakeBackbone:
        m = _FakeBackbone()
        if ckpt_dir != "pre-trained":
            sd = torch.load(Path(ckpt_dir) / "adapter_model.pt", map_location="cpu", weights_only=True)
            with torch.no_grad():
                m.lora_weight.copy_(sd["lora_weight"])
        captured_models.append(m)
        return m

    monkeypatch.setattr(train, "load_model", _fake_load_model)
    monkeypatch.setattr(train, "get_peft_model", lambda base, cfg: base)
    monkeypatch.setattr(train, "get_peft_model_state_dict", lambda m: {"lora_weight": m.lora_weight.detach().clone()})

    def _fake_set_peft(m: Any, sd: dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            m.lora_weight.copy_(sd["lora_weight"])

    monkeypatch.setattr(train, "set_peft_model_state_dict", _fake_set_peft)

    def _single_worker_dl(*a: Any, **kw: Any) -> DataLoader[Any]:
        kw["num_workers"] = 0
        return DataLoader(*a, **kw)

    monkeypatch.setattr(train, "DataLoader", _single_worker_dl)

    out_dir = tmp_path / "stage2_out"
    monkeypatch.setattr(
        "sys.argv",
        [
            "train.py",
            "--data",
            str(ds_dir),
            "--output",
            str(out_dir),
            "--min_points",
            "8",
            "--sem_mode",
            "consensus",
            "--sem_dim",
            "8",
            "--init_lora",
            str(init_lora_dir),
            "--freeze_lora",
            "--resolutions",
            "64",
            "--epochs",
            "1",
            "--save_every",
            "1",
        ],
    )
    train.main()
    m1 = captured_models[-1]
    assert m1.training is False
    assert torch.allclose(m1.lora_weight, torch.full((8, 8), 2.0))
    assert (out_dir / "semantic_adapter.pt").exists()
    assert (out_dir / "checkpoint.pt").exists()
    log_text = (out_dir / "log.csv").read_text()
    assert "sem" in log_text.splitlines()[0].split(",")

    # Resume for epoch 2 and check semantic_adapter updates while frozen LoRA stays constant
    ckpt_before = torch.load(out_dir / "checkpoint.pt", map_location="cpu", weights_only=True)
    assert "semantic_adapter" in ckpt_before
    step_before = ckpt_before["step"]
    adapter_before = {k: v.clone() for k, v in ckpt_before["semantic_adapter"].items()}

    monkeypatch.setattr(
        "sys.argv",
        [
            "train.py",
            "--data",
            str(ds_dir),
            "--output",
            str(out_dir),
            "--min_points",
            "8",
            "--sem_mode",
            "consensus",
            "--sem_dim",
            "8",
            "--init_lora",
            str(init_lora_dir),
            "--freeze_lora",
            "--resume",
            "--resolutions",
            "64",
            "--epochs",
            "2",
            "--save_every",
            "1",
        ],
    )
    train.main()
    m2 = captured_models[-1]
    assert m2.training is False
    assert torch.allclose(m2.lora_weight, torch.full((8, 8), 2.0))
    ckpt_after = torch.load(out_dir / "checkpoint.pt", map_location="cpu", weights_only=True)
    assert ckpt_after["step"] > step_before
    assert any(not torch.equal(ckpt_after["semantic_adapter"][k], adapter_before[k]) for k in adapter_before)


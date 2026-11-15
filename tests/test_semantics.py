"""Tests for scenealigner.semantics: label normalization, palette I/O, patch pooling, and CLIP caching."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from scenealigner import semantics


class _DummyInputs(dict[str, torch.Tensor]):
    def to(self, device: str | torch.device) -> _DummyInputs:
        return self


class _DummyTokenizer:
    def __call__(
        self, prompts: list[str], padding: bool = True, truncation: bool = True, return_tensors: str = "pt"
    ) -> _DummyInputs:
        tokens = torch.arange(len(prompts), dtype=torch.long).unsqueeze(1)
        return _DummyInputs(input_ids=tokens)


class _DummyTextOutput:
    def __init__(self, text_embeds: torch.Tensor) -> None:
        self.text_embeds = text_embeds


class _DummyConfig:
    def __init__(self, projection_dim: int) -> None:
        self.projection_dim = projection_dim


class _DummyClipModel:
    def __init__(self, dim: int) -> None:
        self.config = _DummyConfig(dim)

    def __call__(self, input_ids: torch.Tensor, **kwargs: Any) -> _DummyTextOutput:
        batch = input_ids.shape[0]
        embeds = torch.zeros((batch, self.config.projection_dim), dtype=torch.float32)
        embeds[:, 0] = 3.0
        embeds[:, 1] = 4.0
        return _DummyTextOutput(embeds)


def test_normalize_label() -> None:
    assert semantics.normalize_label("  Living_Room-Area ") == "living room area"
    assert semantics.normalize_label("OUTWALL") == "wall"
    for ignored in ("", "   ", "unknown", "VOID", "undefined", "unlabeled", "none", "other"):
        assert semantics.normalize_label(ignored) == ""


def test_save_and_load_semantic_map_roundtrip_and_stale_rel_cleanup(tmp_path: Path) -> None:
    prefix = str(tmp_path / "scene_00000")
    sem_idx = np.array([[0, 1], [2, 1]], dtype=np.uint8)
    palette = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float16)
    sem_rel = np.array([[0.0, 1.0], [0.5, 0.25]], dtype=np.float32)
    labels = ["wall", "door"]

    semantics.save_semantic_map(prefix, sem_idx, palette, sem_rel=sem_rel, labels=labels)
    assert (tmp_path / "scene_00000_sem_rel.png").is_file()

    idx_loaded, pal_loaded, rel_loaded = semantics.load_semantic_map(prefix, load_rel=True)
    assert np.array_equal(idx_loaded, sem_idx)
    assert pal_loaded.dtype == np.float32
    assert np.allclose(pal_loaded, palette.astype(np.float32), atol=1e-3)
    assert np.allclose(rel_loaded, np.round(sem_rel * 255.0) / 255.0, atol=1e-5)

    # Re-saving without sem_rel must remove stale *_sem_rel.png so subsequent loads derive binary mask.
    semantics.save_semantic_map(prefix, sem_idx, palette, sem_rel=None, labels=labels)
    assert not (tmp_path / "scene_00000_sem_rel.png").exists()

    idx_derived, _, rel_derived = semantics.load_semantic_map(prefix, load_rel=True)
    assert np.array_equal(rel_derived, (idx_derived > 0).astype(np.float32))

    idx_only, pal_only = semantics.load_semantic_map(prefix, load_rel=False)
    assert np.array_equal(idx_only, sem_idx)
    assert pal_only.shape == (3, 2)


def test_save_semantic_map_rejects_out_of_bounds_indices(tmp_path: Path) -> None:
    prefix = str(tmp_path / "bad")
    palette_3 = np.zeros((3, 4), dtype=np.float16)

    with pytest.raises(ValueError):
        semantics.save_semantic_map(prefix, np.array([[-1, 0]], dtype=np.int16), palette_3)
    with pytest.raises(ValueError):
        semantics.save_semantic_map(prefix, np.array([[0, 3]], dtype=np.uint8), palette_3)
    with pytest.raises(ValueError):
        semantics.save_semantic_map(prefix, np.array([[0, 260]], dtype=np.int32), palette_3)
    with pytest.raises(ValueError):
        semantics.save_semantic_map(prefix, np.zeros((2, 2, 1), dtype=np.uint8), palette_3)
    with pytest.raises(ValueError):
        semantics.save_semantic_map(prefix, np.array([[0.0, 1.0]], dtype=np.float32), palette_3)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        semantics.save_semantic_map(prefix, np.zeros((2, 2), dtype=np.uint8), np.zeros((257, 4), dtype=np.float16))
    with pytest.raises(ValueError):
        semantics.save_semantic_map(prefix, np.zeros((2, 2), dtype=np.uint8), np.zeros((0, 4), dtype=np.float16))


def test_pool_semantic_patches_sparse_stroke_calibration() -> None:
    sem_idx = np.zeros((16, 16), dtype=np.uint8)
    sem_idx[7, 7:9] = 1  # 2-pixel wall stroke inside a 16x16 patch
    palette = np.array([[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=np.float32)

    feat_np, rel_np = semantics.pool_semantic_patches(sem_idx, palette, sem_rel=None, patch_size=16)
    assert feat_np.shape == (4, 1, 1)
    assert rel_np.shape == (1, 1, 1)
    assert float(rel_np[0, 0, 0]) == pytest.approx(1.0, abs=1e-6)
    assert np.allclose(feat_np[:, 0, 0], palette[1], atol=1e-6)

    sem_rel_partial = np.zeros((16, 16), dtype=np.float32)
    sem_rel_partial[7, 7:9] = 0.4
    _, rel_partial_np = semantics.pool_semantic_patches(sem_idx, palette, sem_rel=sem_rel_partial, patch_size=16)
    assert float(rel_partial_np[0, 0, 0]) == pytest.approx(0.4, abs=1e-6)

    feat_t, rel_t = semantics.pool_semantic_patches(
        torch.from_numpy(sem_idx), torch.from_numpy(palette), sem_rel=torch.from_numpy(sem_rel_partial), patch_size=16
    )
    assert float(rel_t[0, 0, 0]) == pytest.approx(0.4, abs=1e-6)
    assert np.allclose(feat_t[:, 0, 0].numpy(), palette[1], atol=1e-6)


def test_pool_semantic_patches_conflict_and_unlabeled_rel_masking() -> None:
    palette = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    # Case 1: unlabeled everywhere (sem_idx == 0), spurious sem_rel == 0.9 must be masked out.
    sem_unlabeled = np.zeros((16, 16), dtype=np.uint8)
    rel_spurious = np.full((16, 16), 0.9, dtype=np.float32)
    feat_u_np, rel_u_np = semantics.pool_semantic_patches(sem_unlabeled, palette, sem_rel=rel_spurious, patch_size=16)
    feat_u_t, rel_u_t = semantics.pool_semantic_patches(
        torch.from_numpy(sem_unlabeled), torch.from_numpy(palette), sem_rel=torch.from_numpy(rel_spurious), patch_size=16
    )
    assert float(rel_u_np[0, 0, 0]) == 0.0
    assert float(rel_u_t[0, 0, 0]) == 0.0
    assert np.allclose(feat_u_np, 0.0)
    assert np.allclose(feat_u_t.numpy(), 0.0)

    # Case 2: half +e1 and half orthogonal +e2 -> directional agreement ||(0.5, 0.5)|| = 1/sqrt(2).
    sem_split = np.ones((16, 16), dtype=np.uint8)
    sem_split[:, 8:] = 2
    feat_s_np, rel_s_np = semantics.pool_semantic_patches(sem_split, palette, sem_rel=None, patch_size=16)
    feat_s_t, rel_s_t = semantics.pool_semantic_patches(
        torch.from_numpy(sem_split), torch.from_numpy(palette), sem_rel=None, patch_size=16
    )
    expected_rel = float(1.0 / np.sqrt(2.0))
    assert float(rel_s_np[0, 0, 0]) == pytest.approx(expected_rel, abs=1e-5)
    assert float(rel_s_t[0, 0, 0]) == pytest.approx(expected_rel, abs=1e-5)
    expected_vec = np.array([expected_rel, expected_rel, 0.0, 0.0], dtype=np.float32)
    assert np.allclose(feat_s_np[:, 0, 0], expected_vec, atol=1e-5)
    assert np.allclose(feat_s_t[:, 0, 0].numpy(), expected_vec, atol=1e-5)


def test_pool_semantic_patches_numpy_pytorch_equivalence_and_noncontiguous_rot90() -> None:
    rng = np.random.default_rng(123)
    palette = rng.standard_normal((5, 8)).astype(np.float32)
    palette[0] = 0.0
    palette[1:] /= np.linalg.norm(palette[1:], axis=-1, keepdims=True)

    # Non-contiguous rot90 int64 batch (B=2, H=32, W=32) with shared (M, D) palette.
    idx_base = rng.integers(0, 5, size=(2, 32, 32), dtype=np.int64)
    rel_base = rng.random((2, 32, 32), dtype=np.float32)
    idx_rot_t = torch.rot90(torch.from_numpy(idx_base), k=1, dims=(-2, -1))
    rel_rot_t = torch.rot90(torch.from_numpy(rel_base), k=1, dims=(-2, -1))
    assert not idx_rot_t.is_contiguous()

    feat_np_3d, rel_np_3d = semantics.pool_semantic_patches(
        idx_rot_t.numpy(), palette, sem_rel=rel_rot_t.numpy(), patch_size=16
    )
    feat_t_3d, rel_t_3d = semantics.pool_semantic_patches(
        idx_rot_t, torch.from_numpy(palette), sem_rel=rel_rot_t, patch_size=16
    )
    assert feat_np_3d.shape == (2, 8, 2, 2)
    assert rel_np_3d.shape == (2, 1, 2, 2)
    assert np.allclose(feat_np_3d, feat_t_3d.numpy(), atol=1e-5)
    assert np.allclose(rel_np_3d, rel_t_3d.numpy(), atol=1e-5)

    # 2D slice parity.
    feat_np_2d, rel_np_2d = semantics.pool_semantic_patches(idx_base[0], palette, sem_rel=rel_base[0], patch_size=16)
    feat_t_2d, rel_t_2d = semantics.pool_semantic_patches(
        torch.from_numpy(idx_base[0]), torch.from_numpy(palette), sem_rel=torch.from_numpy(rel_base[0]), patch_size=16
    )
    assert np.allclose(feat_np_2d, feat_t_2d.numpy(), atol=1e-5)
    assert np.allclose(rel_np_2d, rel_t_2d.numpy(), atol=1e-5)

    # Non-divisible spatial dims must raise ValueError.
    with pytest.raises(ValueError):
        semantics.pool_semantic_patches(np.zeros((17, 16), dtype=np.uint8), palette, patch_size=16)
    with pytest.raises(ValueError):
        semantics.pool_semantic_patches(torch.zeros((16, 17), dtype=torch.uint8), torch.from_numpy(palette), patch_size=16)


def test_clip_text_encoder_dynamic_dim_and_readonly_cache() -> None:
    encoder = semantics.ClipTextEncoder(device="cpu")
    encoder._model = _DummyClipModel(768)
    encoder._tokenizer = _DummyTokenizer()

    palette = encoder.encode_palette(["wall", "door"])
    assert palette.shape == (3, 768)
    assert palette.dtype == np.float16
    assert np.allclose(palette[0], 0.0)
    assert float(np.linalg.norm(palette[1].astype(np.float32))) == pytest.approx(1.0, abs=1e-3)
    assert float(np.linalg.norm(palette[2].astype(np.float32))) == pytest.approx(1.0, abs=1e-3)

    cached_vec = encoder.encode_label("wall")
    assert cached_vec.shape == (768,)
    assert cached_vec.flags.writeable is False

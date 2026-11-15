"""Open-vocabulary semantic utilities: CLIP text encoding, compact palette I/O, and patch pooling."""
from __future__ import annotations

import json
import os
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal, overload

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image

if TYPE_CHECKING:
    import torch

    from .inference import ProjParams
    from .preprocess import ReconMode

DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch16"
CLIP_EMBED_DIM = 512
PROMPT_TEMPLATES = (
    "a photo of a {}.",
    "an indoor {}.",
    "a {} in a floorplan.",
)
IGNORED_LABELS = frozenset({"", "undefined", "unknown", "unlabeled", "void", "none", "other"})
LABEL_ALIASES = {"outwall": "wall"}


def normalize_label(label: object) -> str:
    """Canonicalize a raw category label string, returning '' for ignored/empty labels."""
    clean = " ".join(str(label).strip().lower().replace("_", " ").replace("-", " ").split())
    clean = LABEL_ALIASES.get(clean, clean)
    if clean in IGNORED_LABELS:
        return ""
    return clean


class ClipTextEncoder:
    """Lazy-loaded CLIP text encoder that averages L2-normalized prompt-template embeddings."""

    def __init__(
        self,
        model_name: str = DEFAULT_CLIP_MODEL,
        device: str | None = None,
        templates: Sequence[str] = PROMPT_TEMPLATES,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.templates = tuple(templates)
        self._model: Any = None
        self._tokenizer: Any = None
        self._embed_dim: int = CLIP_EMBED_DIM
        self._cache: dict[str, NDArray[np.float32]] = {}

    def _sync_embed_dim(self) -> None:
        proj_dim = getattr(getattr(self._model, "config", None), "projection_dim", None)
        if proj_dim is not None:
            self._embed_dim = int(proj_dim)

    def _load(self) -> None:
        if self._model is not None:
            self._sync_embed_dim()
            return
        import torch
        from transformers import CLIPTextModelWithProjection, CLIPTokenizer

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tokenizer = CLIPTokenizer.from_pretrained(self.model_name)
        self._model = CLIPTextModelWithProjection.from_pretrained(self.model_name).to(self.device).eval()  # type: ignore[arg-type]
        self._sync_embed_dim()

    def encode_label(self, label: str) -> NDArray[np.float32]:
        """Encode a single concept label into an L2-normalized (D,) float32 vector."""
        clean = normalize_label(label)
        if not clean:
            vec = np.zeros(self._embed_dim, dtype=np.float32)
            vec.setflags(write=False)
            return vec
        if clean in self._cache:
            return self._cache[clean]

        import torch
        import torch.nn.functional as F

        self._load()
        prompts = [template.format(clean) for template in self.templates]
        inputs = self._tokenizer(prompts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            embeds = self._model(**inputs).text_embeds.float()
            embeds = F.normalize(embeds, p=2, dim=-1)
            pooled = F.normalize(embeds.mean(dim=0, keepdim=True), p=2, dim=-1).squeeze(0)
            vec = pooled.cpu().numpy().astype(np.float32)
        vec.setflags(write=False)
        self._cache[clean] = vec
        return vec

    def encode_palette(self, labels: Sequence[str]) -> NDArray[np.float16]:
        """Encode labels for indices 1..N into a (N + 1, D) float16 palette with zero row at index 0."""
        if self._model is not None or any(normalize_label(label) for label in labels):
            self._load()
        rows = [np.zeros(self._embed_dim, dtype=np.float32)]
        for label in labels:
            rows.append(self.encode_label(label))
        return np.stack(rows, axis=0).astype(np.float16)


def save_semantic_map(
    prefix: str,
    sem_idx: NDArray[np.integer[Any]],
    palette: NDArray[np.floating[Any]],
    sem_rel: NDArray[np.floating[Any]] | NDArray[np.uint8] | None = None,
    labels: Sequence[str] | None = None,
) -> str:
    """Save indexed semantic map (*_sem_idx.png), CLIP palette (*_palette.npy), and optional reliability/labels."""
    idx_raw = np.asarray(sem_idx)
    if idx_raw.ndim != 2 or not np.issubdtype(idx_raw.dtype, np.integer):
        raise ValueError(f"sem_idx must be a 2D integer array, got shape={idx_raw.shape} dtype={idx_raw.dtype}")

    pal_arr = np.asarray(palette, dtype=np.float16)
    if pal_arr.ndim != 2 or not (1 <= pal_arr.shape[0] <= 256):
        raise ValueError(f"palette must be 2D with 1..256 entries, got shape={pal_arr.shape}")

    if idx_raw.size > 0 and (int(idx_raw.min()) < 0 or int(idx_raw.max()) >= pal_arr.shape[0]):
        raise ValueError(f"sem_idx values must lie in [0, {pal_arr.shape[0] - 1}]")

    parent = os.path.dirname(prefix)
    if parent:
        os.makedirs(parent, exist_ok=True)

    idx_arr = idx_raw.astype(np.uint8, copy=False)
    Image.fromarray(idx_arr).save(f"{prefix}_sem_idx.png")
    np.save(f"{prefix}_palette.npy", pal_arr)

    rel_path = f"{prefix}_sem_rel.png"
    if sem_rel is not None:
        rel_arr = np.asarray(sem_rel)
        if rel_arr.shape != idx_arr.shape:
            raise ValueError(f"sem_rel shape {rel_arr.shape} must match sem_idx shape {idx_arr.shape}")
        if rel_arr.dtype != np.uint8:
            rel_arr = np.clip(np.round(rel_arr.astype(np.float32) * 255.0), 0, 255).astype(np.uint8)
        Image.fromarray(rel_arr).save(rel_path)
    elif os.path.isfile(rel_path):
        os.remove(rel_path)

    if labels is not None:
        with open(f"{prefix}_labels.json", "w") as f:
            json.dump(list(labels), f)
    return prefix


@overload
def load_semantic_map(
    prefix: str, load_rel: Literal[True] = True
) -> tuple[NDArray[np.uint8], NDArray[np.float32], NDArray[np.float32]]: ...


@overload
def load_semantic_map(prefix: str, load_rel: Literal[False]) -> tuple[NDArray[np.uint8], NDArray[np.float32]]: ...


def load_semantic_map(
    prefix: str, load_rel: bool = True
) -> tuple[NDArray[np.uint8], NDArray[np.float32], NDArray[np.float32]] | tuple[NDArray[np.uint8], NDArray[np.float32]]:
    """Load indexed semantic map and palette; optionally load or derive reliability in [0, 1]."""
    sem_idx = np.array(Image.open(f"{prefix}_sem_idx.png").convert("L"), dtype=np.uint8)
    palette = np.load(f"{prefix}_palette.npy").astype(np.float32)
    if not load_rel:
        return sem_idx, palette

    rel_path = f"{prefix}_sem_rel.png"
    if os.path.isfile(rel_path):
        sem_rel = np.array(Image.open(rel_path).convert("L"), dtype=np.float32) / 255.0
    else:
        sem_rel = (sem_idx > 0).astype(np.float32)
    return sem_idx, palette, sem_rel


def _pool_patches_numpy(
    sem_idx: NDArray[Any],
    palette: NDArray[Any],
    sem_rel: NDArray[Any] | None,
    patch_size: int,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    if sem_idx.ndim not in (2, 3):
        raise ValueError(f"sem_idx must be 2D or 3D, got shape={sem_idx.shape}")
    if patch_size <= 0:
        raise ValueError(f"patch_size must be positive, got {patch_size}")
    if sem_rel is not None and sem_rel.shape != sem_idx.shape:
        raise ValueError(f"sem_rel shape {sem_rel.shape} must match sem_idx shape {sem_idx.shape}")

    squeeze = sem_idx.ndim == 2
    if squeeze:
        sem_idx = sem_idx[None, ...]
        if sem_rel is not None:
            sem_rel = sem_rel[None, ...]

    bsz, h, w = sem_idx.shape
    if h % patch_size != 0 or w % patch_size != 0:
        raise ValueError(f"sem_idx spatial dims {(h, w)} must be divisible by patch_size={patch_size}")
    if palette.ndim not in (2, 3):
        raise ValueError(f"palette must be 2D or 3D, got shape={palette.shape}")
    if palette.ndim == 3 and palette.shape[0] != bsz:
        raise ValueError(f"3D palette batch size ({palette.shape[0]}) must match sem_idx batch size ({bsz})")

    hp, wp = h // patch_size, w // patch_size
    n_patches = hp * wp
    p2 = patch_size * patch_size
    num_pal, embed_dim = palette.shape[-2], palette.shape[-1]
    if sem_idx.size > 0 and (int(sem_idx.min()) < 0 or int(sem_idx.max()) >= num_pal):
        raise ValueError(f"sem_idx values must lie in [0, {num_pal - 1}]")

    active = sem_idx > 0
    rel = active.astype(np.float32) if sem_rel is None else (sem_rel.astype(np.float32) * active)

    idx_p = (
        np.ascontiguousarray(sem_idx, dtype=np.int64)
        .reshape(bsz, hp, patch_size, wp, patch_size)
        .transpose(0, 1, 3, 2, 4)
        .reshape(bsz, n_patches, p2)
    )
    rel_p = (
        np.ascontiguousarray(rel, dtype=np.float32)
        .reshape(bsz, hp, patch_size, wp, patch_size)
        .transpose(0, 1, 3, 2, 4)
        .reshape(bsz, n_patches, p2)
    )
    b_ids = np.arange(bsz, dtype=np.int64)[:, None, None]
    p_ids = np.arange(n_patches, dtype=np.int64)[None, :, None]
    flat_bins = ((b_ids * n_patches + p_ids) * num_pal + idx_p).ravel()
    patch_weights = (
        np.bincount(flat_bins, weights=rel_p.ravel(), minlength=bsz * n_patches * num_pal)
        .astype(np.float32)
        .reshape(bsz, n_patches, num_pal)
    )
    patch_weights[:, :, 0] = 0.0

    pal_f32 = palette.astype(np.float32, copy=False)
    weighted_p = np.matmul(patch_weights, pal_f32).reshape(bsz, hp, wp, embed_dim)
    active_count = np.ascontiguousarray(active, dtype=np.float32).reshape(bsz, hp, patch_size, wp, patch_size).sum(axis=(2, 4))
    norms = np.linalg.norm(weighted_p, axis=-1, keepdims=True)
    patch_rel = np.clip(norms[..., 0] / np.maximum(active_count, 1.0), 0.0, 1.0)[:, None, :, :].astype(np.float32)

    patch_feat = np.divide(weighted_p, np.maximum(norms, 1e-6), out=np.zeros_like(weighted_p), where=norms > 1e-6)
    patch_feat = np.transpose(patch_feat, (0, 3, 1, 2)).astype(np.float32, copy=False)
    if squeeze:
        return patch_feat[0], patch_rel[0]
    return patch_feat, patch_rel


@overload
def pool_semantic_patches(
    sem_idx: NDArray[Any],
    palette: NDArray[Any],
    sem_rel: NDArray[Any] | None = None,
    patch_size: int = 16,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]: ...


@overload
def pool_semantic_patches(
    sem_idx: torch.Tensor,
    palette: torch.Tensor,
    sem_rel: torch.Tensor | None = None,
    patch_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]: ...


def pool_semantic_patches(
    sem_idx: NDArray[Any] | torch.Tensor,
    palette: NDArray[Any] | torch.Tensor,
    sem_rel: NDArray[Any] | torch.Tensor | None = None,
    patch_size: int = 16,
) -> tuple[NDArray[np.float32], NDArray[np.float32]] | tuple[torch.Tensor, torch.Tensor]:
    """Gather per-pixel CLIP embeddings and pool into (D, H/p, W/p) features + active-pixel reliability."""
    if isinstance(sem_idx, np.ndarray):
        if not isinstance(palette, np.ndarray):
            raise TypeError("palette must be a numpy.ndarray when sem_idx is a numpy.ndarray")
        if sem_rel is not None and not isinstance(sem_rel, np.ndarray):
            raise TypeError("sem_rel must be a numpy.ndarray when sem_idx is a numpy.ndarray")
        return _pool_patches_numpy(sem_idx, palette, sem_rel, patch_size)

    import torch  # lazy import to avoid possible cyclic deps

    if isinstance(palette, np.ndarray):
        raise TypeError("palette must be a torch.Tensor when sem_idx is a torch.Tensor")
    if isinstance(sem_rel, np.ndarray):
        raise TypeError("sem_rel must be a torch.Tensor when sem_idx is a torch.Tensor")

    if sem_idx.ndim not in (2, 3):
        raise ValueError(f"sem_idx must be 2D or 3D, got shape={tuple(sem_idx.shape)}")
    if patch_size <= 0:
        raise ValueError(f"patch_size must be positive, got {patch_size}")
    if sem_rel is not None and sem_rel.shape != sem_idx.shape:
        raise ValueError(f"sem_rel shape {tuple(sem_rel.shape)} must match sem_idx shape {tuple(sem_idx.shape)}")

    squeeze = sem_idx.ndim == 2
    if squeeze:
        sem_idx = sem_idx.unsqueeze(0)
        if sem_rel is not None:
            sem_rel = sem_rel.unsqueeze(0)

    bsz, h, w = sem_idx.shape
    if h % patch_size != 0 or w % patch_size != 0:
        raise ValueError(f"sem_idx spatial dims {(h, w)} must be divisible by patch_size={patch_size}")
    if palette.ndim not in (2, 3):
        raise ValueError(f"palette must be 2D or 3D, got shape={tuple(palette.shape)}")
    if palette.ndim == 3 and palette.shape[0] != bsz:
        raise ValueError(f"3D palette batch size ({palette.shape[0]}) must match sem_idx batch size ({bsz})")

    hp, wp = h // patch_size, w // patch_size
    n_patches = hp * wp
    p2 = patch_size * patch_size
    num_pal, embed_dim = palette.shape[-2], palette.shape[-1]
    if sem_idx.numel() > 0 and (int(sem_idx.min().item()) < 0 or int(sem_idx.max().item()) >= num_pal):
        raise ValueError(f"sem_idx values must lie in [0, {num_pal - 1}]")

    active = sem_idx > 0
    rel = active.float() if sem_rel is None else (sem_rel.float() * active)

    idx_p = sem_idx.long().reshape(bsz, hp, patch_size, wp, patch_size).permute(0, 1, 3, 2, 4).reshape(bsz, n_patches, p2)
    rel_p = rel.reshape(bsz, hp, patch_size, wp, patch_size).permute(0, 1, 3, 2, 4).reshape(bsz, n_patches, p2)
    patch_weights = torch.zeros(bsz, n_patches, num_pal, dtype=torch.float32, device=sem_idx.device)
    patch_weights.scatter_add_(2, idx_p, rel_p)
    patch_weights[:, :, 0] = 0.0

    pal_3d = palette.unsqueeze(0).expand(bsz, -1, -1) if palette.ndim == 2 else palette
    weighted_p = torch.bmm(patch_weights, pal_3d.float()).reshape(bsz, hp, wp, embed_dim).permute(0, 3, 1, 2)
    active_count = active.float().reshape(bsz, hp, patch_size, wp, patch_size).sum(dim=(2, 4)).unsqueeze(1)

    norms = weighted_p.norm(p=2, dim=1, keepdim=True)
    patch_feat = torch.where(norms > 1e-6, weighted_p / norms.clamp(min=1e-6), torch.zeros_like(weighted_p))
    patch_rel = (norms / active_count.clamp(min=1.0)).clamp(0.0, 1.0)
    if squeeze:
        return patch_feat.squeeze(0), patch_rel.squeeze(0)
    return patch_feat, patch_rel


def _resize_mask_nearest(mask_2d: NDArray[np.integer[Any]], new_w: int, new_h: int) -> NDArray[Any]:
    """Resize a 2D integer mask with OpenCV NEAREST, safely handling int32/int64 masks if needed."""
    if mask_2d.dtype in (np.uint8, np.uint16, np.int16):
        return cv2.resize(mask_2d, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    uniq, inv = np.unique(mask_2d, return_inverse=True)
    inv_2d = inv.reshape(mask_2d.shape)
    if len(uniq) <= 65536:
        resized_inv = np.asarray(
            cv2.resize(inv_2d.astype(np.uint16), (new_w, new_h), interpolation=cv2.INTER_NEAREST),
            dtype=np.intp,
        )
        return np.asarray(uniq[resized_inv], dtype=mask_2d.dtype)
    if int(mask_2d.min()) < -(2**31) or int(mask_2d.max()) > 2**31 - 1:
        raise OverflowError("Integer mask with >65536 unique labels exceeds 32-bit signed PIL 'I' range")
    pil_res = Image.fromarray(mask_2d.astype(np.int32), mode="I").resize((new_w, new_h), Image.Resampling.NEAREST)
    return np.asarray(pil_res, dtype=mask_2d.dtype)


@overload
def align_mask_to_grid(
    mask: NDArray[np.integer[Any]],
    target_size: int = 518,
    mode: ReconMode = "pad",
    conf: None = None,
) -> NDArray[Any]: ...


@overload
def align_mask_to_grid(
    mask: NDArray[np.integer[Any]],
    target_size: int = 518,
    mode: ReconMode = "pad",
    *,
    conf: NDArray[np.floating[Any]],
) -> tuple[NDArray[Any], NDArray[np.float32]]: ...


@overload
def align_mask_to_grid(
    mask: NDArray[np.integer[Any]],
    target_size: int,
    mode: ReconMode,
    conf: NDArray[np.floating[Any]],
) -> tuple[NDArray[Any], NDArray[np.float32]]: ...


def align_mask_to_grid(
    mask: NDArray[np.integer[Any]],
    target_size: int = 518,
    mode: ReconMode = "pad",
    conf: NDArray[np.floating[Any]] | None = None,
) -> NDArray[Any] | tuple[NDArray[Any], NDArray[np.float32]]:
    """Align (H, W) or (S, H, W) integer semantic masks (and optional confidence) identically to load_and_preprocess_images."""
    if mode not in ("pad", "crop"):
        raise ValueError(f"mode must be 'pad' or 'crop', got {mode!r}")
    if target_size <= 0:
        raise ValueError(f"target_size must be positive, got {target_size}")

    mask_arr = np.asarray(mask)
    if mask_arr.ndim not in (2, 3) or not np.issubdtype(mask_arr.dtype, np.integer):
        raise ValueError(f"mask must be a 2D or 3D integer array, got shape={mask_arr.shape} dtype={mask_arr.dtype}")

    conf_arr: NDArray[np.float32] | None = None
    if conf is not None:
        conf_arr = np.asarray(conf, dtype=np.float32)
        if conf_arr.shape != mask_arr.shape:
            raise ValueError(f"conf shape {conf_arr.shape} must match mask shape {mask_arr.shape}")

    squeeze = mask_arr.ndim == 2
    masks_3d = mask_arr[None, ...] if squeeze else mask_arr
    confs_3d = None if conf_arr is None else (conf_arr[None, ...] if squeeze else conf_arr)

    aligned_masks: list[NDArray[Any]] = []
    aligned_confs: list[NDArray[np.float32]] = []
    for idx in range(masks_3d.shape[0]):
        m_slice = masks_3d[idx]
        h, w = m_slice.shape
        if w >= h or mode == "crop":
            new_w, new_h = target_size, round(h * (target_size / w) / 14) * 14
        else:
            new_w, new_h = round(w * (target_size / h) / 14) * 14, target_size
        if new_w <= 0 or new_h <= 0:
            raise ValueError(f"Degenerate resized dimensions {(new_w, new_h)} for input shape {(h, w)}")

        m_res = _resize_mask_nearest(m_slice, new_w, new_h)
        c_res = (
            None
            if confs_3d is None
            else cv2.resize(confs_3d[idx], (new_w, new_h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        )

        eff_w, eff_h = new_w, new_h
        if eff_h > target_size:
            top = (eff_h - target_size) // 2
            m_res = m_res[top : top + target_size]
            if c_res is not None:
                c_res = c_res[top : top + target_size]
            eff_h = target_size
        if eff_w > target_size:
            left = (eff_w - target_size) // 2
            m_res = m_res[:, left : left + target_size]
            if c_res is not None:
                c_res = c_res[:, left : left + target_size]
            eff_w = target_size

        pad_w, pad_h = target_size - eff_w, target_size - eff_h
        pad_left, pad_top = pad_w // 2, pad_h // 2
        pad_right, pad_bottom = pad_w - pad_left, pad_h - pad_top
        m_pad = (
            np.pad(m_res, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="constant", constant_values=0)
            if (pad_top or pad_bottom or pad_left or pad_right)
            else m_res
        )
        aligned_masks.append(m_pad)

        if c_res is not None:
            c_pad = cv2.copyMakeBorder(c_res, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0.0)
            aligned_confs.append(c_pad.astype(np.float32, copy=False))

    out_masks = aligned_masks[0] if squeeze else np.stack(aligned_masks, axis=0)
    if conf is None:
        return out_masks
    out_confs = aligned_confs[0] if squeeze else np.stack(aligned_confs, axis=0)
    return out_masks, out_confs


def aggregate_topdown_semantics(
    row_px: NDArray[np.integer[Any]],
    col_px: NDArray[np.integer[Any]],
    point_sem_idx: NDArray[np.integer[Any]],
    palette: NDArray[np.floating[Any]],
    density_size: int,
    point_conf: NDArray[np.floating[Any]] | None = None,
    tau: float = 5.0,
    min_snap_cos: float = 0.0,
    labels: Sequence[str] | None = None,
    eps: float = 1e-8,
) -> tuple[NDArray[np.uint8], NDArray[np.float32], NDArray[np.float16], list[str]]:
    """Aggregate posed semantic points into top-down consensus index, reliability [0, 1], and compact palette."""
    if density_size <= 0:
        raise ValueError(f"density_size must be positive, got {density_size}")
    if tau <= 0:
        raise ValueError(f"tau must be positive, got {tau}")

    pal_arr = np.asarray(palette, dtype=np.float32)
    if pal_arr.ndim != 2 or pal_arr.shape[0] < 1:
        raise ValueError(f"palette must be 2D with at least row 0, got shape={pal_arr.shape}")
    num_pal, embed_dim = pal_arr.shape
    if labels is not None and len(labels) != num_pal - 1:
        raise ValueError(f"labels length ({len(labels)}) must equal palette rows - 1 ({num_pal - 1})")

    row_arr = np.asarray(row_px, dtype=np.int64).reshape(-1)
    col_arr = np.asarray(col_px, dtype=np.int64).reshape(-1)
    sem_arr = np.asarray(point_sem_idx, dtype=np.int64).reshape(-1)
    n_pts = len(row_arr)
    if len(col_arr) != n_pts or len(sem_arr) != n_pts:
        raise ValueError("row_px, col_px, and point_sem_idx must have identical lengths")

    if point_conf is None:
        conf_arr = np.ones(n_pts, dtype=np.float32)
    else:
        conf_arr = np.maximum(np.asarray(point_conf, dtype=np.float32).reshape(-1), 0.0)
        if len(conf_arr) != n_pts:
            raise ValueError(f"point_conf length ({len(conf_arr)}) must match points ({n_pts})")

    empty_idx = np.zeros((density_size, density_size), dtype=np.uint8)
    empty_rel = np.zeros((density_size, density_size), dtype=np.float32)
    empty_pal = np.zeros((1, embed_dim), dtype=np.float16)

    valid = (
        (row_arr >= 0) & (row_arr < density_size)
        & (col_arr >= 0) & (col_arr < density_size)
        & (sem_arr > 0) & (sem_arr < num_pal)
        & np.isfinite(conf_arr) & (conf_arr > 0.0)
    )
    if not np.any(valid):
        return empty_idx, empty_rel, empty_pal, []

    row_v, col_v = row_arr[valid], col_arr[valid]
    sem_v, conf_v = sem_arr[valid], conf_arr[valid]
    cell_flat = row_v * density_size + col_v
    unique_cells, cell_inv = np.unique(cell_flat, return_inverse=True)
    n_cells = len(unique_cells)

    n_obs = np.bincount(cell_inv, minlength=n_cells).astype(np.float32)
    total_w = np.bincount(cell_inv, weights=conf_v, minlength=n_cells).astype(np.float32)

    pal_norms = np.linalg.norm(pal_arr, axis=1, keepdims=True)
    pal_unit = np.divide(pal_arr, np.maximum(pal_norms, 1e-6), out=np.zeros_like(pal_arr), where=pal_norms > 1e-6)

    # Accumulate weighted embeddings per unique (cell, concept) pair to stay O(U * D) in memory
    unique_pairs, pair_inv = np.unique(cell_inv * num_pal + sem_v, return_inverse=True)
    pair_w = np.bincount(pair_inv, weights=conf_v).astype(np.float32)
    pair_cell = unique_pairs // num_pal
    pair_sem = unique_pairs % num_pal

    mu = np.zeros((n_cells, embed_dim), dtype=np.float32)
    np.add.at(mu, pair_cell, pair_w[:, None] * pal_unit[pair_sem])
    mu /= np.maximum(total_w[:, None], eps)

    mu_norm = np.linalg.norm(mu, axis=1)
    hat_e = np.divide(mu, np.maximum(mu_norm[:, None], eps), out=np.zeros_like(mu), where=mu_norm[:, None] > 1e-6)

    active_ids = np.unique(sem_v)
    active_ids = active_ids[pal_norms[active_ids, 0] > 1e-6]
    if len(active_ids) == 0:
        return empty_idx, empty_rel, empty_pal, []
    cand_vecs = pal_unit[active_ids]

    matched_id = np.zeros(n_cells, dtype=np.int64)
    best_cos = np.zeros(n_cells, dtype=np.float32)
    chunk_size = 4096
    for start in range(0, n_cells, chunk_size):
        end = min(start + chunk_size, n_cells)
        scores = hat_e[start:end] @ cand_vecs.T
        best_pos = np.argmax(scores, axis=1)
        matched_id[start:end] = active_ids[best_pos]
        best_cos[start:end] = scores[np.arange(end - start), best_pos]

    keep_cell = (mu_norm > 1e-6) & (best_cos > min_snap_cos)
    raw_rel = mu_norm * (1.0 - np.exp(-n_obs / np.float32(tau))) * np.maximum(best_cos, 0.0)
    cell_rel = np.where(keep_cell, np.clip(raw_rel, 0.0, 1.0), 0.0).astype(np.float32)
    keep_cell &= cell_rel > 0.0
    if not np.any(keep_cell):
        return empty_idx, empty_rel, empty_pal, []

    used_ids, id_counts = np.unique(matched_id[keep_cell], return_counts=True)
    if len(used_ids) > 255:
        top_order = np.argsort(-id_counts, kind="stable")[:255]
        used_ids = np.sort(used_ids[top_order])
        keep_cell &= np.isin(matched_id, used_ids)

    lut = np.zeros(num_pal, dtype=np.uint8)
    lut[used_ids] = np.arange(1, len(used_ids) + 1, dtype=np.uint8)

    kept_cells = unique_cells[keep_cell]
    kept_rows, kept_cols = kept_cells // density_size, kept_cells % density_size
    empty_idx[kept_rows, kept_cols] = lut[matched_id[keep_cell]]
    empty_rel[kept_rows, kept_cols] = cell_rel[keep_cell]

    compact_palette = np.vstack([np.zeros((1, embed_dim), dtype=np.float16), pal_unit[used_ids].astype(np.float16)])
    compact_labels = [labels[int(i) - 1] for i in used_ids] if labels is not None else []
    return empty_idx, empty_rel, compact_palette, compact_labels


def project_posed_semantics(
    points_ground: NDArray[np.floating[Any]],
    mask: NDArray[np.bool_],
    sem_idx_grid: NDArray[np.integer[Any]],
    palette: NDArray[np.floating[Any]],
    proj_params: ProjParams,
    conf_grid: NDArray[np.floating[Any]] | None = None,
    tau: float = 5.0,
    min_snap_cos: float = 0.0,
    training_rounding: bool = False,
    labels: Sequence[str] | None = None,
) -> tuple[NDArray[np.uint8], NDArray[np.float32], NDArray[np.float16], list[str]]:
    """Project 1 or S posed views' 2D semantic grids into a top-down consensus map matching density coordinates."""
    mask_arr = np.asarray(mask, dtype=bool)
    mask_3d = mask_arr[None, ...] if mask_arr.ndim == 2 else mask_arr
    if mask_3d.ndim != 3:
        raise ValueError(f"mask must be 2D or 3D, got shape={mask_arr.shape}")

    sem_arr = np.asarray(sem_idx_grid)
    if not np.issubdtype(sem_arr.dtype, np.integer):
        raise ValueError(f"sem_idx_grid must have integer dtype, got {sem_arr.dtype}")
    sem_3d = sem_arr[None, ...] if sem_arr.ndim == 2 else sem_arr
    if sem_3d.shape != mask_3d.shape:
        raise ValueError(f"sem_idx_grid shape {sem_3d.shape} must match mask shape {mask_3d.shape}")

    conf_3d: NDArray[np.floating[Any]] | None = None
    if conf_grid is not None:
        conf_arr = np.asarray(conf_grid, dtype=np.float32)
        conf_3d = conf_arr[None, ...] if conf_arr.ndim == 2 else conf_arr
        if conf_3d.shape != mask_3d.shape:
            raise ValueError(f"conf_grid shape {conf_3d.shape} must match mask shape {mask_3d.shape}")

    pts = np.asarray(points_ground, dtype=np.float64)
    expected_n = int(mask_3d.sum())
    if pts.ndim != 2 or pts.shape != (expected_n, 3):
        raise ValueError(f"points_ground must have shape ({expected_n}, 3) matching mask, got {pts.shape}")

    point_sem = sem_3d[mask_3d]
    point_conf = None if conf_3d is None else conf_3d[mask_3d]

    size = proj_params["density_size"]
    u_f = (pts[:, 0] - proj_params["min_x"]) * proj_params["scale"] + proj_params["cx"]
    v_f = (pts[:, 2] - proj_params["min_z"]) * proj_params["scale"] + proj_params["cz"]
    if training_rounding:
        col_px = u_f.astype(np.int64)
        row_px = (size - 1) - v_f.astype(np.int64)
    else:
        coords = np.stack([u_f, size - 1 - v_f], axis=1).astype(np.float32).astype(np.int32)
        col_px = coords[:, 0]
        row_px = coords[:, 1]

    return aggregate_topdown_semantics(
        row_px,
        col_px,
        point_sem,
        palette,
        density_size=size,
        point_conf=point_conf,
        tau=tau,
        min_snap_cos=min_snap_cos,
        labels=labels,
    )

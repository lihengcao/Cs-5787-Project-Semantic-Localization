"""Open-vocabulary semantic utilities: CLIP text encoding, compact palette I/O, and patch pooling."""
from __future__ import annotations

import json
import os
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal, overload

import numpy as np
from numpy.typing import NDArray
from PIL import Image

if TYPE_CHECKING:
    import torch

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

    hp, wp = h // patch_size, w // patch_size
    embed_dim = palette.shape[-1]

    active = sem_idx > 0
    rel = active.astype(np.float32) if sem_rel is None else (sem_rel.astype(np.float32) * active)

    if palette.ndim == 2:
        pix_emb = palette[sem_idx.astype(np.int64)].astype(np.float32)
    else:
        batch_ids = np.arange(bsz)[:, None, None]
        pix_emb = palette[batch_ids, sem_idx.astype(np.int64)].astype(np.float32)

    weighted = pix_emb * rel[..., None]
    weighted_p = weighted.reshape(bsz, hp, patch_size, wp, patch_size, embed_dim).sum(axis=(2, 4))
    active_count = active.astype(np.float32).reshape(bsz, hp, patch_size, wp, patch_size).sum(axis=(2, 4))
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
        assert isinstance(palette, np.ndarray)
        assert sem_rel is None or isinstance(sem_rel, np.ndarray)
        return _pool_patches_numpy(sem_idx, palette, sem_rel, patch_size)

    import torch

    assert not isinstance(palette, np.ndarray)
    assert sem_rel is None or not isinstance(sem_rel, np.ndarray)

    if sem_idx.ndim not in (2, 3):
        raise ValueError(f"sem_idx must be 2D or 3D, got shape={tuple(sem_idx.shape)}")
    if patch_size <= 0:
        raise ValueError(f"patch_size must be positive, got {patch_size}")

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

    hp, wp = h // patch_size, w // patch_size
    embed_dim = palette.shape[-1]

    active = sem_idx > 0
    rel = active.float() if sem_rel is None else (sem_rel.float() * active)

    pal_3d = palette.unsqueeze(0).expand(bsz, -1, -1) if palette.ndim == 2 else palette
    idx_flat = sem_idx.long().reshape(bsz, h * w)
    pix_emb = torch.gather(
        pal_3d.float(),
        1,
        idx_flat.unsqueeze(-1).expand(-1, -1, embed_dim),
    ).reshape(bsz, h, w, embed_dim)

    weighted = (pix_emb * rel.unsqueeze(-1)).permute(0, 3, 1, 2)  # (B, D, H, W)
    weighted_p = weighted.reshape(bsz, embed_dim, hp, patch_size, wp, patch_size).sum(dim=(3, 5))
    active_count = active.float().reshape(bsz, hp, patch_size, wp, patch_size).sum(dim=(2, 4)).unsqueeze(1)

    norms = weighted_p.norm(p=2, dim=1, keepdim=True)
    patch_feat = torch.where(norms > 1e-6, weighted_p / norms.clamp(min=1e-6), torch.zeros_like(weighted_p))
    patch_rel = (norms / active_count.clamp(min=1.0)).clamp(0.0, 1.0)
    if squeeze:
        return patch_feat.squeeze(0), patch_rel.squeeze(0)
    return patch_feat, patch_rel

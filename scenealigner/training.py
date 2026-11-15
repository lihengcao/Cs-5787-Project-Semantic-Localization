"""Training data and losses of the LoRA layers."""
from __future__ import annotations

import glob
import os
import random
import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, NotRequired, TypedDict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from .inference import norm_transform, normalize_active_descriptors
from .semantics import load_semantic_map, pool_semantic_patches

if TYPE_CHECKING:
    from .inference import ProjParams


class TrainingSampleDict(TypedDict):
    density_img: torch.Tensor
    plan_img: torch.Tensor
    density_pts: torch.Tensor
    plan_pts: torch.Tensor
    valid: torch.Tensor
    scene_id: str
    density_sem_idx: NotRequired[torch.Tensor]
    density_sem_pal: NotRequired[torch.Tensor]
    density_sem_rel: NotRequired[torch.Tensor]
    plan_sem_idx: NotRequired[torch.Tensor]
    plan_sem_pal: NotRequired[torch.Tensor]
    plan_sem_rel: NotRequired[torch.Tensor]


class TrainingBatchDict(TypedDict):
    density_img: torch.Tensor
    plan_img: torch.Tensor
    density_pts: torch.Tensor
    plan_pts: torch.Tensor
    valid: torch.Tensor
    scene_id: list[str]
    density_sem_feat: NotRequired[torch.Tensor]
    density_sem_rel: NotRequired[torch.Tensor]
    plan_sem_feat: NotRequired[torch.Tensor]
    plan_sem_rel: NotRequired[torch.Tensor]


def render_training_map(points: NDArray[np.floating[Any]], proj_params: ProjParams, gamma: float = 0.5) -> Image.Image:
    """Density map as rendered for the training data. Its rounding differs from the one at inference: the points
    are assigned to pixels before the vertical flip, and the density is truncated before the inversion."""
    size = proj_params["density_size"]
    u = ((points[:, 0] - proj_params["min_x"]) * proj_params["scale"] + proj_params["cx"]).astype(np.int64)
    v = ((points[:, 2] - proj_params["min_z"]) * proj_params["scale"] + proj_params["cz"]).astype(np.int64)
    density_map = np.bincount((size - 1 - v) * size + u, minlength=size * size).reshape(size, size)
    density_map = np.power(density_map.astype(np.float32), gamma)
    return Image.fromarray(255 - (density_map / density_map.max() * 255.0).astype(np.uint8))


def _rotate_coords(
    img: Image.Image, pts: NDArray[np.floating[Any]], k: int, *, pixel_centered: bool
) -> tuple[Image.Image, NDArray[np.floating[Any]]]:
    """Rotates an image by k * 90 degrees CCW and maps points using either corner (off=0) or center (off=1) indexing."""
    w, h = img.size
    off = 1 if pixel_centered else 0
    x, y = pts[:, 0], pts[:, 1]
    pts = np.stack([(x, y), (y, w - off - x), (w - off - x, h - off - y), (h - off - y, x)][k], axis=1)
    return img.rotate(k * 90, expand=True), pts


def random_rotate(
    img: Image.Image, pts: NDArray[np.floating[Any]], k: int
) -> tuple[Image.Image, NDArray[np.floating[Any]]]:
    """Rotates an image by k * 90 degrees (counter-clockwise) together with its points."""
    return _rotate_coords(img, pts, k, pixel_centered=False)


def _rotate_pixel_centered(
    img: Image.Image, pts: NDArray[np.floating[Any]], k: int
) -> tuple[Image.Image, NDArray[np.floating[Any]]]:
    """Rotates image and 0-indexed pixel coordinates by k * 90 deg CCW around pixel centers."""
    return _rotate_coords(img, pts, k, pixel_centered=True)


def _to_writable_tensor(arr: NDArray[Any]) -> torch.Tensor:
    """Converts a NumPy array into a contiguous, writable PyTorch tensor without UserWarnings."""
    if not arr.flags.writeable:
        return torch.from_numpy(np.array(arr, copy=True))
    return torch.from_numpy(np.ascontiguousarray(arr))


def _rotate_sem(
    sem_idx: NDArray[np.uint8], sem_rel: NDArray[np.float32], k: int
) -> tuple[NDArray[np.uint8], NDArray[np.float32]]:
    """Rotates dense semantic index and reliability maps by k * 90 degrees CCW."""
    return np.ascontiguousarray(np.rot90(sem_idx, k)), np.ascontiguousarray(np.rot90(sem_rel, k))


def _resize_sem_nearest(
    sem_idx: NDArray[np.uint8], sem_rel: NDArray[np.float32], size: int
) -> tuple[NDArray[np.uint8], NDArray[np.float32]]:
    """Resizes discrete semantic index and reliability maps using pixel-centered NEAREST resampling."""
    resized_idx = np.array(Image.fromarray(sem_idx).resize((size, size), Image.Resampling.NEAREST), dtype=np.uint8, copy=True)
    resized_rel = np.array(Image.fromarray(sem_rel).resize((size, size), Image.Resampling.NEAREST), dtype=np.float32, copy=True)
    return resized_idx, resized_rel


_TRAILING_INT_RE = re.compile(r"(\d+)$")
_SLICE_CSV_RE = re.compile(r"^scenes_(\d+)_(\d+)\.csv$")


def _parse_scene_num(scene_id: str) -> int | None:
    """Extract the trailing integer scene index from a scene_id like 'scene_00123'."""
    match = _TRAILING_INT_RE.search(str(scene_id).strip())
    return int(match.group(1)) if match else None


class DensityFloorplanDataset(Dataset[TrainingSampleDict]):
    """Pairs of a density map and a floorplan with their correspondences, listed in the CSV files of a folder."""

    def __init__(
        self,
        root: str,
        num_points: int = 1024,
        min_points: int = 10000,
        image_size: int | None = None,
        load_semantics: bool = False,
        scene_ranges: Sequence[tuple[int, int]] | None = None,
    ) -> None:
        self.root = root
        self.num_points = num_points
        self.min_points = min_points
        self.image_size = image_size
        self.load_semantics = load_semantics
        normalized_ranges: list[tuple[int, int]] | None = None
        if scene_ranges is not None:
            normalized_ranges = [(int(lo), int(hi)) for lo, hi in scene_ranges]
            for lo, hi in normalized_ranges:
                if lo < 0 or hi <= lo:
                    raise ValueError(f"Invalid scene range {(lo, hi)}: expected 0 <= start < end")
        self.scene_ranges = normalized_ranges

        all_files = sorted(glob.glob(os.path.join(root, "*.csv")))
        if not all_files:
            raise ValueError(f"No CSV files found in {root}")
        if normalized_ranges is not None:
            files = []
            for f in all_files:
                m = _SLICE_CSV_RE.match(os.path.basename(f))
                if m:
                    s0, s1 = int(m.group(1)), int(m.group(2))
                    if not any(s0 < hi and s1 > lo for lo, hi in normalized_ranges):
                        continue
                files.append(f)
            if not files:
                raise ValueError(f"No CSV files in {root} overlap scene_ranges={normalized_ranges}")
        else:
            files = all_files

        self.data = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        # Drop duplicate view entries when slice CSVs overlap across resumed runs
        self.data = self.data.drop_duplicates(subset=["scene_id", "density"], ignore_index=True)
        if normalized_ranges is not None:
            keep_mask: list[bool] = []
            for sid in self.data["scene_id"]:
                num = _parse_scene_num(str(sid))
                if num is None:
                    raise ValueError(
                        f"Cannot filter by scene_ranges={normalized_ranges}: non-numeric scene_id {sid!r}"
                    )
                keep_mask.append(any(lo <= num < hi for lo, hi in normalized_ranges))
            self.data = self.data[np.asarray(keep_mask, dtype=bool)].reset_index(drop=True)
        if len(self.data) == 0:
            raise ValueError(f"No training samples found in {root} for scene_ranges={normalized_ranges}")
        if self.load_semantics:
            missing = [col for col in ("density_sem", "floorplan_sem") if col not in self.data.columns]
            if missing:
                raise ValueError(f"load_semantics=True requires columns {missing} in {root} CSVs")
            if self.image_size is not None and self.image_size % 16 != 0:
                raise ValueError(f"image_size={self.image_size} must be divisible by 16 when load_semantics=True")
        self.jitter_plan = T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)
        self.jitter_density = T.ColorJitter(brightness=0.2, contrast=0.2)

    def __len__(self) -> int:
        return len(self.data)

    def _sample_crop(
        self,
        img: Image.Image,
        pts: NDArray[np.floating[Any]],
        crop_range: tuple[float, float] = (0.7, 1.0),
    ) -> tuple[tuple[int, int, int, int], NDArray[np.floating[Any]], NDArray[np.bool_]] | None:
        w, h = img.size
        scale = random.uniform(*crop_range)
        crop_w, crop_h = int(w * scale), int(h * scale)
        left, top = random.randint(0, w - crop_w), random.randint(0, h - crop_h)
        cropped = pts - np.float32([left, top])  # type: ignore[arg-type]
        inside = (cropped[:, 0] >= 0) & (cropped[:, 0] < crop_w) & (cropped[:, 1] >= 0) & (cropped[:, 1] < crop_h)
        if inside.sum() < self.min_points:
            return None
        return (left, top, left + crop_w, top + crop_h), cropped[inside], inside

    def random_crop(
        self,
        img: Image.Image,
        pts: NDArray[np.floating[Any]],
        other_pts: NDArray[np.floating[Any]],
        crop_range: tuple[float, float] = (0.7, 1.0),
    ) -> tuple[Image.Image, NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
        """Random crop of an image, kept only if at least min_points correspondences stay inside."""
        sampled = self._sample_crop(img, pts, crop_range)
        if sampled is None:
            return img, pts, other_pts
        box, cropped_pts, inside = sampled
        return img.crop(box), cropped_pts, other_pts[inside]

    def _random_crop_joint(
        self,
        img: Image.Image,
        pts: NDArray[np.floating[Any]],
        other_pts: NDArray[np.floating[Any]],
        sem_idx: NDArray[np.uint8],
        sem_rel: NDArray[np.float32],
        crop_range: tuple[float, float] = (0.7, 1.0),
    ) -> tuple[
        Image.Image,
        NDArray[np.floating[Any]],
        NDArray[np.floating[Any]],
        NDArray[np.uint8],
        NDArray[np.float32],
    ]:
        sampled = self._sample_crop(img, pts, crop_range)
        if sampled is None:
            return img, pts, other_pts, sem_idx, sem_rel
        (left, top, right, bottom), cropped_pts, inside = sampled
        return (
            img.crop((left, top, right, bottom)),
            cropped_pts,
            other_pts[inside],
            np.ascontiguousarray(sem_idx[top:bottom, left:right]),
            np.ascontiguousarray(sem_rel[top:bottom, left:right]),
        )

    def __getitem__(self, idx: int) -> TrainingSampleDict:
        row = self.data.iloc[idx]
        density = Image.open(os.path.join(self.root, str(row.density))).convert("RGB")
        plan = Image.open(os.path.join(self.root, str(row.floorplan))).convert("RGB")
        pts_plan, pts_density = np.load(os.path.join(self.root, str(row.correspondences)))  # (N, 2) pixels each

        d_idx: NDArray[np.uint8] | None = None
        d_pal: NDArray[np.float32] | None = None
        d_rel: NDArray[np.float32] | None = None
        p_idx: NDArray[np.uint8] | None = None
        p_pal: NDArray[np.float32] | None = None
        p_rel: NDArray[np.float32] | None = None
        if self.load_semantics:
            d_idx, d_pal, d_rel = load_semantic_map(os.path.join(self.root, str(row.density_sem)), load_rel=True)
            p_idx, p_pal, p_rel = load_semantic_map(os.path.join(self.root, str(row.floorplan_sem)), load_rel=True)
            if d_idx.shape != (density.height, density.width) or p_idx.shape != (plan.height, plan.width):
                raise ValueError(
                    f"Semantic map dimensions must match RGB images: density {d_idx.shape} vs "
                    f"{(density.height, density.width)}, plan {p_idx.shape} vs {(plan.height, plan.width)}"
                )

        # Augmentation: independent rotations and crops of the two images, colour jitter
        k_density = int(np.random.randint(0, 4))
        k_plan = int(np.random.randint(0, 4))
        if self.load_semantics:
            assert d_idx is not None and d_rel is not None and p_idx is not None and p_rel is not None
            density, pts_density = _rotate_pixel_centered(density, pts_density, k_density)
            plan, pts_plan = _rotate_pixel_centered(plan, pts_plan, k_plan)
            d_idx, d_rel = _rotate_sem(d_idx, d_rel, k_density)
            p_idx, p_rel = _rotate_sem(p_idx, p_rel, k_plan)
            plan, pts_plan, pts_density, p_idx, p_rel = self._random_crop_joint(plan, pts_plan, pts_density, p_idx, p_rel)
            density, pts_density, pts_plan, d_idx, d_rel = self._random_crop_joint(density, pts_density, pts_plan, d_idx, d_rel)
        else:
            density, pts_density = random_rotate(density, pts_density, k_density)
            plan, pts_plan = random_rotate(plan, pts_plan, k_plan)
            plan, pts_plan, pts_density = self.random_crop(plan, pts_plan, pts_density)
            density, pts_density, pts_plan = self.random_crop(density, pts_density, pts_plan)
        plan, density = self.jitter_plan(plan), self.jitter_density(density)

        # Pixels -> [-1, 1] (grid_sample), without duplicates
        pts_density = np.clip(2 * ((pts_density + 0.5) / np.float32(density.size)) - 1, -1, 1)
        pts_plan = np.clip(2 * ((pts_plan + 0.5) / np.float32(plan.size)) - 1, -1, 1)
        if self.image_size:  # training at a single resolution
            density = density.resize((self.image_size, self.image_size), Image.Resampling.BICUBIC)
            plan = plan.resize((self.image_size, self.image_size), Image.Resampling.BICUBIC)
            if self.load_semantics:
                assert d_idx is not None and d_rel is not None and p_idx is not None and p_rel is not None
                d_idx, d_rel = _resize_sem_nearest(d_idx, d_rel, self.image_size)
                p_idx, p_rel = _resize_sem_nearest(p_idx, p_rel, self.image_size)
        _, unique = np.unique(np.round(np.hstack([pts_density, pts_plan]), decimals=3), axis=0, return_index=True)
        pts_density, pts_plan = pts_density[np.sort(unique)], pts_plan[np.sort(unique)]

        # num_points correspondences; missing ones are random points, used as negatives only
        n = len(pts_density)
        if n >= self.num_points:
            keep = np.random.choice(n, self.num_points, replace=False)
            pts_density_t, pts_plan_t = torch.tensor(pts_density[keep]), torch.tensor(pts_plan[keep])
        else:
            pts_density_t = torch.cat([torch.tensor(pts_density), torch.rand(self.num_points - n, 2) * 2 - 1])
            pts_plan_t = torch.cat([torch.tensor(pts_plan), torch.rand(self.num_points - n, 2) * 2 - 1])
        valid = torch.arange(self.num_points) < n
        sample: TrainingSampleDict = {
            "density_img": norm_transform(density),
            "plan_img": norm_transform(plan),
            "density_pts": pts_density_t,
            "plan_pts": pts_plan_t,
            "valid": valid,
            "scene_id": str(row.scene_id),
        }
        if self.load_semantics:
            assert d_idx is not None and d_pal is not None and d_rel is not None
            assert p_idx is not None and p_pal is not None and p_rel is not None
            sample["density_sem_idx"] = _to_writable_tensor(d_idx)
            sample["density_sem_pal"] = _to_writable_tensor(d_pal)
            sample["density_sem_rel"] = _to_writable_tensor(d_rel)
            sample["plan_sem_idx"] = _to_writable_tensor(p_idx)
            sample["plan_sem_pal"] = _to_writable_tensor(p_pal)
            sample["plan_sem_rel"] = _to_writable_tensor(p_rel)
        return sample


def collate(batch: Sequence[TrainingSampleDict], resolutions: Sequence[int]) -> TrainingBatchDict:
    """All images of a batch are resized to one resolution, drawn at random (mixed-resolution training)."""
    size = random.choice(resolutions)
    pts_tensors = {k: torch.stack([sample[k] for sample in batch]) for k in ("density_pts", "plan_pts", "valid")}
    img_tensors = {k: torch.cat([F.interpolate(sample[k][None], size=(size, size), mode="bicubic", antialias=True) for sample in batch])
                   for k in ("density_img", "plan_img")}  # the points are normalized: no rescaling
    batch_out: TrainingBatchDict = {
        "density_img": img_tensors["density_img"],
        "plan_img": img_tensors["plan_img"],
        "density_pts": pts_tensors["density_pts"],
        "plan_pts": pts_tensors["plan_pts"],
        "valid": pts_tensors["valid"],
        "scene_id": [sample["scene_id"] for sample in batch],
    }
    if any("density_sem_idx" in s or "plan_sem_idx" in s for s in batch):
        if size % 16 != 0:
            raise ValueError(f"Batch resolution {size} must be divisible by patch_size=16 when semantics are enabled.")

        def _pool_side(is_density: bool) -> tuple[torch.Tensor, torch.Tensor]:
            feats: list[torch.Tensor] = []
            rels: list[torch.Tensor] = []
            for sample in batch:
                if is_density:
                    if (
                        "density_sem_idx" not in sample
                        or "density_sem_pal" not in sample
                        or "density_sem_rel" not in sample
                    ):
                        raise ValueError(
                            "All samples in a batch must contain density_sem_idx/pal/rel when semantics are enabled."
                        )
                    idx_t, pal_t, rel_t = sample["density_sem_idx"], sample["density_sem_pal"], sample["density_sem_rel"]
                else:
                    if "plan_sem_idx" not in sample or "plan_sem_pal" not in sample or "plan_sem_rel" not in sample:
                        raise ValueError(
                            "All samples in a batch must contain plan_sem_idx/pal/rel when semantics are enabled."
                        )
                    idx_t, pal_t, rel_t = sample["plan_sem_idx"], sample["plan_sem_pal"], sample["plan_sem_rel"]
                idx_np = np.asarray(idx_t, dtype=np.uint8)
                pal_np = np.asarray(pal_t, dtype=np.float32)
                rel_np = np.asarray(rel_t, dtype=np.float32)
                if idx_np.shape != (size, size):
                    idx_np, rel_np = _resize_sem_nearest(idx_np, rel_np, size)
                pooled_feat, pooled_rel = pool_semantic_patches(idx_np, pal_np, rel_np, patch_size=16)
                assert pooled_rel is not None
                feats.append(torch.from_numpy(np.ascontiguousarray(pooled_feat)).to(torch.float32))
                rels.append(torch.from_numpy(np.ascontiguousarray(pooled_rel)).to(torch.float32))
            return torch.stack(feats), torch.stack(rels)

        d_feat, d_rel = _pool_side(True)
        p_feat, p_rel = _pool_side(False)
        batch_out["density_sem_feat"] = d_feat
        batch_out["density_sem_rel"] = d_rel
        batch_out["plan_sem_feat"] = p_feat
        batch_out["plan_sem_rel"] = p_rel
    return batch_out


def feature_matching_loss(
    feat_a: torch.Tensor,
    feat_b: torch.Tensor,
    same_scene: torch.Tensor,
    valid: torch.Tensor,
    temperature: float = 0.07,
    *,
    sem_a: torch.Tensor | None = None,
    sem_b: torch.Tensor | None = None,
    rel_a: torch.Tensor | None = None,
    rel_b: torch.Tensor | None = None,
    lambda_sem: float = 0.0,
) -> torch.Tensor:
    """Symmetric InfoNCE over all points of the batch (Eq. 2). feat_a, feat_b: (B, N, C).

    same_scene (B, B): pairs of different samples of one scene, whose points are not used as negatives.
    """
    if lambda_sem < 0.0:
        raise ValueError(f"lambda_sem must be non-negative, got {lambda_sem}")
    B, N, C = feat_a.shape
    sim_geo = torch.matmul(
        F.normalize(feat_a, dim=-1).reshape(B * N, C),
        F.normalize(feat_b, dim=-1).reshape(B * N, C).T,
    )
    if lambda_sem > 0.0:
        if sem_a is None or sem_b is None:
            raise ValueError("sem_a and sem_b are required when lambda_sem > 0")
        if (rel_a is None) != (rel_b is None):
            raise ValueError("rel_a and rel_b must both be provided or both be None")
        sem_dim = sem_a.shape[-1]
        sem_a_norm = normalize_active_descriptors(sem_a).reshape(B * N, sem_dim)
        sem_b_norm = normalize_active_descriptors(sem_b).reshape(B * N, sem_dim)
        sim_sem = torch.matmul(sem_a_norm, sem_b_norm.T)
        if rel_a is not None and rel_b is not None:
            gate = rel_a.reshape(B * N, 1).clamp(0.0, 1.0) * rel_b.reshape(1, B * N).clamp(0.0, 1.0)
            sim_sem = gate * sim_sem
        logits = (sim_geo + lambda_sem * sim_sem) / temperature
    else:
        logits = sim_geo / temperature
    logits = logits.masked_fill(same_scene[:, None, :, None].expand(B, N, B, N).reshape(B * N, B * N), float("-inf"))
    idx = torch.nonzero(valid.reshape(B * N)).squeeze(1)
    if idx.numel() == 0:
        zero = (feat_a.sum() + feat_b.sum()) * 0.0
        if sem_a is not None and sem_b is not None and lambda_sem > 0.0:
            zero = zero + (sem_a.sum() + sem_b.sum()) * 0.0
        return zero
    return (F.cross_entropy(logits[idx], idx) + F.cross_entropy(logits.T[idx], idx)) / 2


def semantic_contrastive_loss(
    sem_a: torch.Tensor,
    sem_b: torch.Tensor,
    raw_sem_a: torch.Tensor,
    raw_sem_b: torch.Tensor,
    valid: torch.Tensor,
    temperature: float = 0.07,
    same_concept_threshold: float = 0.90,
    *,
    rel_a: torch.Tensor | None = None,
    rel_b: torch.Tensor | None = None,
) -> torch.Tensor:
    """Symmetric InfoNCE for projected semantics with same-concept negative masking."""
    if (rel_a is None) != (rel_b is None):
        raise ValueError("rel_a and rel_b must both be provided or both be None")
    bsz, num_pts, dim = sem_a.shape
    total_pts = bsz * num_pts
    z_a = normalize_active_descriptors(sem_a).reshape(total_pts, dim)
    z_b = normalize_active_descriptors(sem_b).reshape(total_pts, dim)

    raw_a = raw_sem_a.reshape(total_pts, -1)
    raw_b = raw_sem_b.reshape(total_pts, -1)
    active_a = raw_a.norm(p=2, dim=-1) > 1e-6
    active_b = raw_b.norm(p=2, dim=-1) > 1e-6
    raw_a_norm = normalize_active_descriptors(raw_a)
    raw_b_norm = normalize_active_descriptors(raw_b)

    anchors = valid.reshape(total_pts).bool() & active_a & active_b
    anchor_w: torch.Tensor | None = None
    if rel_a is not None and rel_b is not None:
        anchor_w = rel_a.reshape(total_pts).clamp(0.0, 1.0) * rel_b.reshape(total_pts).clamp(0.0, 1.0)
        anchors = anchors & (anchor_w > 0.0)
    if not bool(anchors.any()):
        return (sem_a.sum() + sem_b.sum()) * 0.0

    off_diag = ~torch.eye(total_pts, dtype=torch.bool, device=sem_a.device)
    logits_ab = torch.matmul(z_a, z_b.T) / temperature
    logits_ba = logits_ab.T

    sim_ab = torch.matmul(raw_a_norm, raw_b_norm.T)
    sim_bb = torch.matmul(raw_b_norm, raw_b_norm.T)
    sim_aa = torch.matmul(raw_a_norm, raw_a_norm.T)

    mask_ab = off_diag & ((~active_b)[None, :] | (sim_ab >= same_concept_threshold) | (sim_bb >= same_concept_threshold))
    mask_ba = off_diag & ((~active_a)[None, :] | (sim_ab.T >= same_concept_threshold) | (sim_aa >= same_concept_threshold))

    logits_ab = logits_ab.masked_fill(mask_ab, float("-inf"))
    logits_ba = logits_ba.masked_fill(mask_ba, float("-inf"))

    has_neg_ab = (off_diag & ~mask_ab).any(dim=-1)
    has_neg_ba = (off_diag & ~mask_ba).any(dim=-1)
    idx_ab = torch.nonzero(anchors & has_neg_ab).squeeze(1)
    idx_ba = torch.nonzero(anchors & has_neg_ba).squeeze(1)
    if idx_ab.numel() == 0 and idx_ba.numel() == 0:
        return (sem_a.sum() + sem_b.sum()) * 0.0

    terms: list[torch.Tensor] = []
    if idx_ab.numel() > 0:
        ce_ab = F.cross_entropy(logits_ab[idx_ab], idx_ab, reduction="none")
        if anchor_w is not None:
            w_ab = anchor_w[idx_ab]
            terms.append((ce_ab * w_ab).sum() / w_ab.sum().clamp(min=1e-6))
        else:
            terms.append(ce_ab.mean())
    if idx_ba.numel() > 0:
        ce_ba = F.cross_entropy(logits_ba[idx_ba], idx_ba, reduction="none")
        if anchor_w is not None:
            w_ba = anchor_w[idx_ba]
            terms.append((ce_ba * w_ba).sum() / w_ba.sum().clamp(min=1e-6))
        else:
            terms.append(ce_ba.mean())
    return torch.stack(terms).mean()


def regression_loss(pred: torch.Tensor, gt: torch.Tensor, weights: torch.Tensor, delta: float = 0.02) -> torch.Tensor:
    """Confidence-weighted Huber loss on the predicted floorplan coordinates (Eq. 3)."""
    dist = torch.norm(pred - gt, p=2, dim=-1)
    return (F.huber_loss(dist, torch.zeros_like(dist), delta=delta, reduction="none") * weights).sum() / (weights.sum() + 1e-6)


def sample_groups(B: int, N: int, size: int, num_groups: int, device: torch.device) -> torch.Tensor:
    """Random groups of `size` distinct points per sample: indices (B, K, size)."""
    K = min(num_groups, N // size)
    return torch.argsort(torch.rand(B, N, device=device), dim=1)[:, :size * K].view(B, K, size)


def geometry_loss(
    src: torch.Tensor, pred: torch.Tensor, weights: torch.Tensor, num_pairs: int = 512, delta: float = 0.02, eps: float = 1e-6
) -> torch.Tensor:
    """Consistent distance ratios between point pairs of the density map and their predictions (Eq. 5)."""
    idx = sample_groups(*src.shape[:2], 2, num_pairs, src.device)  # type: ignore[call-arg]
    batch = torch.arange(len(src), device=src.device).unsqueeze(1)
    i, j = idx[:, :, 0], idx[:, :, 1]
    w = weights[batch, i] * weights[batch, j]
    ds = torch.norm(src[batch, i] - src[batch, j], dim=-1).clamp(min=eps)
    dp = torch.norm(pred[batch, i] - pred[batch, j], dim=-1).clamp(min=eps)
    diff = torch.log(dp) - torch.log(ds)
    mean = (diff * w).sum(dim=1, keepdim=True) / (w.sum(dim=1, keepdim=True) + eps)
    return (F.huber_loss(diff, mean.detach().expand_as(diff), delta=delta, reduction="none") * w).sum() / (w.sum() + eps)


def topology_loss(
    src: torch.Tensor,
    pred: torch.Tensor,
    weights: torch.Tensor,
    num_triplets: int = 512,
    delta: float = 0.02,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Consistent angles (cosine and sine) of point triplets of the density map and their predictions (Eq. 4)."""
    idx = sample_groups(*src.shape[:2], 3, num_triplets, src.device)  # type: ignore[call-arg]
    batch = torch.arange(len(src), device=src.device).unsqueeze(1)
    i, j, k = idx[:, :, 0], idx[:, :, 1], idx[:, :, 2]
    w = weights[batch, i] * weights[batch, j] * weights[batch, k]
    u_src, v_src = src[batch, j] - src[batch, i], src[batch, k] - src[batch, i]
    u_pred, v_pred = pred[batch, j] - pred[batch, i], pred[batch, k] - pred[batch, i]

    def sine(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        u, v = F.normalize(u, dim=-1), F.normalize(v, dim=-1)
        return u[:, :, 0] * v[:, :, 1] - u[:, :, 1] * v[:, :, 0]

    cos_src, cos_pred = F.cosine_similarity(u_src, v_src, dim=-1), F.cosine_similarity(u_pred, v_pred, dim=-1)
    loss = F.huber_loss(cos_pred, cos_src.detach(), delta=delta, reduction="none") + \
        F.huber_loss(sine(u_pred, v_pred), sine(u_src, v_src).detach(), delta=delta, reduction="none")
    return (loss * w).sum() / (w.sum() + eps)

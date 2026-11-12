"""Training data and losses of the LoRA layers."""
from __future__ import annotations

import glob
import os
import random
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, TypedDict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from .inference import norm_transform

if TYPE_CHECKING:
    from .inference import ProjParams


class TrainingSampleDict(TypedDict):
    density_img: torch.Tensor
    plan_img: torch.Tensor
    density_pts: torch.Tensor
    plan_pts: torch.Tensor
    valid: torch.Tensor
    scene_id: str


class TrainingBatchDict(TypedDict):
    density_img: torch.Tensor
    plan_img: torch.Tensor
    density_pts: torch.Tensor
    plan_pts: torch.Tensor
    valid: torch.Tensor
    scene_id: list[str]


def render_training_map(points: NDArray[np.floating[Any]], proj_params: ProjParams, gamma: float = 0.5) -> Image.Image:
    """Density map as rendered for the training data. Its rounding differs from the one at inference: the points
    are assigned to pixels before the vertical flip, and the density is truncated before the inversion."""
    size = proj_params["density_size"]
    u = ((points[:, 0] - proj_params["min_x"]) * proj_params["scale"] + proj_params["cx"]).astype(np.int64)
    v = ((points[:, 2] - proj_params["min_z"]) * proj_params["scale"] + proj_params["cz"]).astype(np.int64)
    density_map = np.bincount((size - 1 - v) * size + u, minlength=size * size).reshape(size, size)
    density_map = np.power(density_map.astype(np.float32), gamma)
    return Image.fromarray(255 - (density_map / density_map.max() * 255.0).astype(np.uint8))


def random_rotate(
    img: Image.Image, pts: NDArray[np.floating[Any]], k: int
) -> tuple[Image.Image, NDArray[np.floating[Any]]]:
    """Rotates an image by k * 90 degrees (counter-clockwise) together with its points."""
    w, h = img.size
    x, y = pts[:, 0], pts[:, 1]
    pts = np.stack([(x, y), (y, w - x), (w - x, h - y), (h - y, x)][k], axis=1)
    return img.rotate(k * 90, expand=True), pts


class DensityFloorplanDataset(Dataset[TrainingSampleDict]):
    """Pairs of a density map and a floorplan with their correspondences, listed in the CSV files of a folder."""

    def __init__(self, root: str, num_points: int = 1024, min_points: int = 10000, image_size: int | None = None) -> None:
        self.root, self.num_points, self.min_points, self.image_size = root, num_points, min_points, image_size
        files = sorted(glob.glob(os.path.join(root, "*.csv")))
        self.data = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        self.jitter_plan = T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)
        self.jitter_density = T.ColorJitter(brightness=0.2, contrast=0.2)

    def __len__(self) -> int:
        return len(self.data)

    def random_crop(
        self,
        img: Image.Image,
        pts: NDArray[np.floating[Any]],
        other_pts: NDArray[np.floating[Any]],
        crop_range: tuple[float, float] = (0.7, 1.0),
    ) -> tuple[Image.Image, NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
        """Random crop of an image, kept only if at least min_points correspondences stay inside."""
        w, h = img.size
        scale = random.uniform(*crop_range)
        crop_w, crop_h = int(w * scale), int(h * scale)
        left, top = random.randint(0, w - crop_w), random.randint(0, h - crop_h)
        cropped = pts - np.float32([left, top])  # type: ignore[arg-type]
        inside = (cropped[:, 0] >= 0) & (cropped[:, 0] < crop_w) & (cropped[:, 1] >= 0) & (cropped[:, 1] < crop_h)
        if inside.sum() < self.min_points:
            return img, pts, other_pts
        return img.crop((left, top, left + crop_w, top + crop_h)), cropped[inside], other_pts[inside]

    def __getitem__(self, idx: int) -> TrainingSampleDict:
        row = self.data.iloc[idx]
        density = Image.open(os.path.join(self.root, str(row.density))).convert("RGB")
        plan = Image.open(os.path.join(self.root, str(row.floorplan))).convert("RGB")
        pts_plan, pts_density = np.load(os.path.join(self.root, str(row.correspondences)))  # (N, 2) pixels each

        # Augmentation: independent rotations and crops of the two images, colour jitter
        density, pts_density = random_rotate(density, pts_density, int(np.random.randint(0, 4)))
        plan, pts_plan = random_rotate(plan, pts_plan, int(np.random.randint(0, 4)))
        plan, pts_plan, pts_density = self.random_crop(plan, pts_plan, pts_density)
        density, pts_density, pts_plan = self.random_crop(density, pts_density, pts_plan)
        plan, density = self.jitter_plan(plan), self.jitter_density(density)

        # Pixels -> [-1, 1] (grid_sample), without duplicates
        pts_density = np.clip(2 * ((pts_density + 0.5) / np.float32(density.size)) - 1, -1, 1)
        pts_plan = np.clip(2 * ((pts_plan + 0.5) / np.float32(plan.size)) - 1, -1, 1)
        if self.image_size:  # training at a single resolution
            density = density.resize((self.image_size, self.image_size), Image.Resampling.BICUBIC)
            plan = plan.resize((self.image_size, self.image_size), Image.Resampling.BICUBIC)
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
        return {"density_img": norm_transform(density), "plan_img": norm_transform(plan), "density_pts": pts_density_t,
                "plan_pts": pts_plan_t, "valid": valid, "scene_id": str(row.scene_id)}


def collate(batch: Sequence[TrainingSampleDict], resolutions: Sequence[int]) -> TrainingBatchDict:
    """All images of a batch are resized to one resolution, drawn at random (mixed-resolution training)."""
    size = random.choice(resolutions)
    pts_tensors = {k: torch.stack([sample[k] for sample in batch]) for k in ("density_pts", "plan_pts", "valid")}
    img_tensors = {k: torch.cat([F.interpolate(sample[k][None], size=(size, size), mode="bicubic", antialias=True) for sample in batch])
                   for k in ("density_img", "plan_img")}  # the points are normalized: no rescaling
    return {
        "density_img": img_tensors["density_img"],
        "plan_img": img_tensors["plan_img"],
        "density_pts": pts_tensors["density_pts"],
        "plan_pts": pts_tensors["plan_pts"],
        "valid": pts_tensors["valid"],
        "scene_id": [sample["scene_id"] for sample in batch],
    }


def feature_matching_loss(
    feat_a: torch.Tensor,
    feat_b: torch.Tensor,
    same_scene: torch.Tensor,
    valid: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Symmetric InfoNCE over all points of the batch (Eq. 2). feat_a, feat_b: (B, N, C).

    same_scene (B, B): pairs of different samples of one scene, whose points are not used as negatives.
    """
    B, N, C = feat_a.shape
    logits = torch.matmul(F.normalize(feat_a, dim=-1).reshape(B * N, C), F.normalize(feat_b, dim=-1).reshape(B * N, C).T) / temperature
    logits = logits.masked_fill(same_scene[:, None, :, None].expand(B, N, B, N).reshape(B * N, B * N), float("-inf"))
    idx = torch.nonzero(valid.reshape(B * N)).squeeze(1)
    return (F.cross_entropy(logits[idx], idx) + F.cross_entropy(logits.T[idx], idx)) / 2


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

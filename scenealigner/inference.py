"""Floorplan alignment: density map (Sec. 3.2), cross-modal correspondences and 2D similarity transform M
(Sec. 3.3), and the floorplan-aligned 3D scene."""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Literal, NotRequired, TypedDict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dinov3.hub.backbones import dinov3_vitb16
from huggingface_hub import hf_hub_download, snapshot_download
from numpy.typing import NDArray
from peft import PeftModel
from PIL import Image
from safetensors.torch import load_file
from torchvision import transforms as T

from .semantics import CLIP_EMBED_DIM, pool_semantic_patches, project_posed_semantics

if TYPE_CHECKING:
    from .preprocess import ReconstructionDict


class ProjParams(TypedDict):
    R_rect: NDArray[np.floating[Any]]
    min_x: float
    min_z: float
    scale: float
    cx: float
    cz: float
    density_size: int


class PoseAlignment(TypedDict):
    proj_params: ProjParams
    M: NDArray[Any]


class AlignmentDict(PoseAlignment):
    density_img: Image.Image
    src_px: NDArray[np.float32]
    dst_px: NDArray[np.float32]


class AlignedSceneDict(TypedDict):
    points: NDArray[np.float32]
    colors: NDArray[np.uint8]
    cam_to_world: NDArray[np.float32]
    thumbnails: NotRequired[NDArray[np.uint8]]
    image_sizes: NotRequired[NDArray[np.integer[Any]]]
    color: NotRequired[NDArray[np.integer[Any]]]


SemMode = Literal["none", "uniform", "consensus"]
SemanticMapInput = tuple[NDArray[np.integer[Any]], NDArray[np.floating[Any]], NDArray[np.floating[Any]] | None]


class AlignmentFailureError(ValueError):
    """Raised when geometric estimation (e.g., RANSAC) fails to find a valid transform."""

MODEL_REPO = "jhcho99/SceneAligner"  # our LoRA layers
S3D_MODEL_REPO = "jhcho99/SceneAligner/structured3d"  # our LoRA layers for Structured3D
# DINOv3 ViT-B/16 weights from timm (https://github.com/facebookresearch/dinov3/issues/145)
DINOV3_TIMM_REPO = "timm/vit_base_patch16_dinov3.lvd1689m"
DINOV3_TIMM_REVISION = "c6a5fb7d12bbd3cf3b0079253141c3332aaed7da"
EVAL_IMG_RES = 512  # resolution of density maps and floorplans
SOFT_ARGMAX_TEMPERATURE = 0.02

norm_transform = T.Compose([T.ToTensor(), T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])


def load_timm_weights(base_model: Any) -> dict[str, torch.Tensor]:
    """timm's DINOv3 weights with the parameter names of the DINOv3 code."""
    state_dict = load_file(hf_hub_download(DINOV3_TIMM_REPO, "model.safetensors", revision=DINOV3_TIMM_REVISION))
    state_dict = {k.replace("reg_token", "storage_tokens").replace("gamma_1", "ls1.gamma").replace("gamma_2", "ls2.gamma"): v
                  for k, v in state_dict.items()}
    # Not in timm's file: q/k/v biases and their masks, mask token (all zero), RoPE periods (bfloat16 in Meta's file)
    for name, tensor in base_model.state_dict().items():
        if name.endswith(("qkv.bias", "qkv.bias_mask")) or name == "mask_token":
            state_dict[name] = torch.zeros_like(tensor)
    state_dict["rope_embed.periods"] = base_model.rope_embed.periods.bfloat16()
    return state_dict


def load_model(checkpoint_dir: str = MODEL_REPO, dinov3_weights: str | None = None) -> Any:
    """DINOv3 ViT-B/16 with our LoRA layers, or the pretrained DINOv3 if checkpoint_dir is "pre-trained".

    checkpoint_dir: local directory, or model repository on the Hugging Face Hub, optionally with a subfolder.
    dinov3_weights: Meta's weight file (default: timm's copy).
    """
    base_model = dinov3_vitb16(pretrained=False)
    state_dict = load_timm_weights(base_model) if dinov3_weights is None else torch.load(dinov3_weights, map_location="cpu")
    base_model.load_state_dict(state_dict)
    if checkpoint_dir == "pre-trained":
        return base_model.eval().to("cuda")
    if not os.path.isdir(checkpoint_dir) and checkpoint_dir.count("/") > 1:  # <user>/<repository>/<subfolder>
        parts = checkpoint_dir.split("/")
        repo, subfolder = "/".join(parts[:2]), "/".join(parts[2:])
        checkpoint_dir = os.path.join(snapshot_download(repo, allow_patterns=f"{subfolder}/*"), subfolder)
    return PeftModel.from_pretrained(base_model, checkpoint_dir).eval().to("cuda")


def reconstruct_projection_geometry(
    points_cam0: NDArray[np.floating[Any]],
    points_conf: NDArray[np.floating[Any]],
    est_gravity_cam: NDArray[np.floating[Any]],
    conf_threshold_percent: float = 45.0,
    xz_slice_val: float = 2.5,
    density_size: int = EVAL_IMG_RES,
    padding: float = 0.1,
    eps: float = 1e-8,
) -> tuple[ProjParams, NDArray[np.floating[Any]], NDArray[np.bool_]]:
    """Rotates the points to the gravity-aligned ground frame, filters them and fits the top-down projection.

    Returns the projection parameters, the filtered points (N, 3) in the ground frame and their mask (S, H, W).
    """
    # Ground frame (Gram-Schmidt): +Y along gravity, +X close to the x-axis of camera 0
    y_target = est_gravity_cam / (np.linalg.norm(est_gravity_cam) + eps)
    x_ref = np.array([1.0, 0.0, 0.0])
    if np.abs(np.dot(y_target, x_ref)) > 0.9999:  # gravity parallel to the x-axis
        x_ref = np.array([0.0, 0.0, 1.0])
    z_temp = np.cross(x_ref, y_target)
    z_temp /= np.linalg.norm(z_temp)
    x_target = np.cross(y_target, z_temp)
    x_target /= np.linalg.norm(x_target)
    z_target = np.cross(x_target, y_target)
    z_target /= np.linalg.norm(z_target)
    R_rect = np.stack([x_target, y_target, z_target], axis=1).T  # camera 0 -> ground
    points = (points_cam0 @ R_rect.T).reshape(-1, 3)
    conf = points_conf.reshape(-1)

    # Confidence, height (+Y is down: top 5% and bottom 20% removed) and x/z percentile filters
    mask = conf > np.percentile(conf, conf_threshold_percent)
    ceil_cutoff, floor_cutoff = np.percentile(points[mask, 1], [5.0, 80.0])
    mask &= (points[:, 1] >= ceil_cutoff) & (points[:, 1] <= floor_cutoff)
    x_min, x_max = np.percentile(points[mask, 0], [xz_slice_val, 100 - xz_slice_val])
    z_min, z_max = np.percentile(points[mask, 2], [xz_slice_val, 100 - xz_slice_val])
    mask &= (points[:, 0] >= x_min) & (points[:, 0] <= x_max) & (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
    points = points[mask]

    # Scale and offsets that center the points in the map, with a margin
    min_x, min_z = points[:, 0].min(), points[:, 2].min()
    width_m, height_m = points[:, 0].max() - min_x, points[:, 2].max() - min_z
    scale = (density_size * (1 - padding * 2)) / max(width_m, height_m)
    proj_params: ProjParams = {"R_rect": R_rect, "min_x": min_x, "min_z": min_z, "scale": scale,
                               "cx": (density_size - width_m * scale) / 2, "cz": (density_size - height_m * scale) / 2,
                               "density_size": density_size}
    return proj_params, points, mask.reshape(points_conf.shape)


def get_density_coordinates(points: NDArray[np.floating[Any]], proj_params: ProjParams) -> NDArray[np.float32]:
    """Ground-frame 3D points (N, 3) -> density-map pixel coordinates (N, 2)."""
    u = (points[:, 0] - proj_params["min_x"]) * proj_params["scale"] + proj_params["cx"]
    v = (points[:, 2] - proj_params["min_z"]) * proj_params["scale"] + proj_params["cz"]
    return np.stack([u, proj_params["density_size"] - 1 - v], axis=1).astype(np.float32)


def render_density_map(points: NDArray[np.floating[Any]], proj_params: ProjParams, gamma_val: float = 0.5) -> Image.Image:
    """Counts the points per pixel, applies gamma correction and normalizes. Returns an RGB image (dark = dense)."""
    size = proj_params["density_size"]
    coords = get_density_coordinates(points, proj_params).astype(np.int32)
    coords = coords[((coords >= 0) & (coords <= size - 1)).all(axis=1)]
    density_map = np.bincount(coords[:, 1].astype(np.int64) * size + coords[:, 0], minlength=size * size)
    density_map = np.power(density_map.astype(np.float32).reshape(size, size), gamma_val)
    if density_map.max() > 0:
        density_map = (density_map / density_map.max()) * 255.0
    return Image.fromarray(np.clip(255.0 - density_map, 0, 255).astype(np.uint8)).convert("RGB")


def intensity_sampling(density_img: Image.Image, num_sample_points: int = 10000) -> torch.Tensor:
    """Samples query points with a probability proportional to the density (darker pixels).

    Returns normalized coordinates in [-1, 1], (1, N, 2), without duplicates.
    """
    img_arr = np.array(density_img.convert("L"))
    H, W = img_arr.shape
    prob_map = 255.0 - img_arr.astype(np.float32)
    prob_map[prob_map < 20] = 0  # background noise
    flat_indices = np.random.choice(H * W, size=num_sample_points, replace=True, p=(prob_map / prob_map.sum()).flatten())
    norm_pts = np.stack([2 * ((flat_indices % W + 0.5) / W) - 1, 2 * ((flat_indices // W + 0.5) / H) - 1], axis=1)
    norm_pts = norm_pts.astype(np.float32)
    _, unique_indices = np.unique(np.round(norm_pts, decimals=3), axis=0, return_index=True)  # ignores float jitter
    return torch.tensor(norm_pts[np.sort(unique_indices)]).unsqueeze(0)


def sample_descriptors(features: torch.Tensor, grid_points: torch.Tensor) -> torch.Tensor:
    """Bilinear interpolation of features (B, C, H, W) at normalized coordinates (B, N, 2). Returns (B, N, C)."""
    sampled = F.grid_sample(features, grid_points.unsqueeze(1), mode="bilinear", align_corners=False)  # (B, C, 1, N)
    return sampled.squeeze(2).permute(0, 2, 1)


def normalize_active_descriptors(desc: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """L2-normalize non-zero descriptors along the last axis while preserving zero vectors."""
    norms = desc.norm(p=2, dim=-1, keepdim=True)
    return torch.where(norms > eps, desc / norms.clamp(min=eps), torch.zeros_like(desc))


class SemanticAdapter(nn.Module):
    """Shared projection MLP mapping pooled 512-D CLIP patch vectors to L2-normalized semantic descriptors."""

    def __init__(self, in_dim: int = CLIP_EMBED_DIM, hidden_dim: int = 256, out_dim: int = 128) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        active = x.norm(p=2, dim=-1, keepdim=True) > 1e-6
        projected = F.normalize(self.net(x), p=2, dim=-1, eps=1e-6)
        return torch.where(active, projected, torch.zeros_like(projected))

    def forward_map(self, feat_map: torch.Tensor) -> torch.Tensor:
        """Project spatial CLIP patch features (B, C, H, W) -> (B, out_dim, H, W)."""
        return self.forward(feat_map.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


def save_semantic_adapter(output_dir: str, adapter: SemanticAdapter, lambda_sem: float | None = 0.5) -> str:
    """Atomically save SemanticAdapter weights and config metadata into <output_dir>/semantic_adapter.pt."""
    if lambda_sem is not None and lambda_sem < 0.0:
        raise ValueError(f"lambda_sem must be non-negative, got {lambda_sem}")
    os.makedirs(output_dir, exist_ok=True)
    ckpt_path = os.path.join(output_dir, "semantic_adapter.pt")
    tmp_path = f"{ckpt_path}.tmp"
    torch.save(
        {
            "state_dict": adapter.state_dict(),
            "in_dim": adapter.in_dim,
            "hidden_dim": adapter.hidden_dim,
            "out_dim": adapter.out_dim,
            "lambda_sem": None if lambda_sem is None else float(lambda_sem),
        },
        tmp_path,
    )
    os.replace(tmp_path, ckpt_path)
    return ckpt_path


def load_semantic_adapter(
    path_or_dir: str, device: torch.device | str = "cuda"
) -> tuple[SemanticAdapter, float | None]:
    """Load SemanticAdapter from a directory, checkpoint file, or HuggingFace repo with weights_only=True."""
    if os.path.isdir(path_or_dir):
        ckpt_path = os.path.join(path_or_dir, "semantic_adapter.pt")
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"SemanticAdapter checkpoint not found: {ckpt_path}")
    elif os.path.isfile(path_or_dir):
        ckpt_path = path_or_dir
    elif path_or_dir.startswith((".", "/", "~")) or path_or_dir.endswith(".pt") or "/" not in path_or_dir:
        if "/" not in path_or_dir:
            ckpt_path = hf_hub_download(path_or_dir, filename="semantic_adapter.pt")
        else:
            raise FileNotFoundError(f"SemanticAdapter checkpoint path not found: {path_or_dir}")
    elif path_or_dir.count("/") > 1:
        parts = path_or_dir.split("/")
        repo, subfolder = "/".join(parts[:2]), "/".join(parts[2:])
        ckpt_path = hf_hub_download(repo, filename=f"{subfolder}/semantic_adapter.pt")
    else:
        ckpt_path = hf_hub_download(path_or_dir, filename="semantic_adapter.pt")

    payload = torch.load(ckpt_path, map_location=device, weights_only=True)
    if isinstance(payload, dict) and "state_dict" in payload:
        state_dict = payload["state_dict"]
        in_dim = int(payload.get("in_dim", CLIP_EMBED_DIM))
        hidden_dim = int(payload.get("hidden_dim", 256))
        out_dim = int(payload.get("out_dim", 128))
        raw_lambda = payload.get("lambda_sem")
        lambda_sem = float(raw_lambda) if raw_lambda is not None else None
    else:
        state_dict = payload
        in_dim, hidden_dim, out_dim, lambda_sem = CLIP_EMBED_DIM, 256, 128, None

    adapter = SemanticAdapter(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim)
    adapter.load_state_dict(state_dict)
    return adapter.to(device).eval(), lambda_sem


def compute_fused_sim_matrix(
    desc_geo_d: torch.Tensor,
    map_geo_p: torch.Tensor,
    *,
    sem_mode: SemMode = "none",
    desc_sem_d: torch.Tensor | None = None,
    map_sem_p: torch.Tensor | None = None,
    rel_d: torch.Tensor | None = None,
    map_rel_p: torch.Tensor | None = None,
    lambda_sem: float = 0.5,
) -> torch.Tensor:
    """Compute patch similarity S_ij = <g_d, g_p> + lambda_sem * w_ij * <s_d, s_p>."""
    if sem_mode not in ("none", "uniform", "consensus"):
        raise ValueError(f"Unsupported sem_mode: {sem_mode!r}")
    if lambda_sem < 0.0:
        raise ValueError(f"lambda_sem must be non-negative, got {lambda_sem}")
    flat_p = F.normalize(map_geo_p.flatten(2).permute(0, 2, 1), dim=-1)
    sim = torch.matmul(F.normalize(desc_geo_d, dim=-1), flat_p.transpose(1, 2))
    if sem_mode == "none" or lambda_sem == 0.0 or desc_sem_d is None or map_sem_p is None:
        return sim

    s_d = normalize_active_descriptors(desc_sem_d)
    flat_s_p = normalize_active_descriptors(map_sem_p.flatten(2).permute(0, 2, 1))
    sem_sim = torch.matmul(s_d, flat_s_p.transpose(1, 2))
    if sem_mode == "consensus":
        if rel_d is None or map_rel_p is None:
            raise ValueError("sem_mode='consensus' requires both rel_d and map_rel_p")
        bsz = desc_geo_d.shape[0]
        gate = rel_d.reshape(bsz, -1, 1).clamp(0.0, 1.0) * map_rel_p.flatten(2).clamp(0.0, 1.0)
        sem_sim = sem_sim * gate
    return sim + lambda_sem * sem_sim


def _pool_and_adapt_map(
    sem_tuple: SemanticMapInput,
    adapter: SemanticAdapter,
    target_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    sem_idx, palette, sem_rel = sem_tuple
    raw_idx = np.asarray(sem_idx)
    pal_arr = np.asarray(palette, dtype=np.float32)
    if raw_idx.ndim != 2 or not np.issubdtype(raw_idx.dtype, np.integer):
        raise ValueError(f"Expected 2D integer semantic index map, got shape={raw_idx.shape} dtype={raw_idx.dtype}")
    if pal_arr.ndim != 2:
        raise ValueError(f"Expected 2D palette array, got shape={pal_arr.shape}")
    num_pal = pal_arr.shape[0]
    if raw_idx.size > 0 and (int(raw_idx.min()) < 0 or int(raw_idx.max()) >= num_pal):
        raise ValueError(f"Semantic index values must lie in [0, {num_pal - 1}]")
    idx_arr = raw_idx.astype(np.uint8, copy=False)
    if idx_arr.shape != (target_size, target_size):
        idx_arr = np.asarray(
            Image.fromarray(idx_arr).resize((target_size, target_size), Image.Resampling.NEAREST), dtype=np.uint8
        )
    rel_arr: NDArray[np.float32] | None = None
    if sem_rel is not None:
        rel_arr = np.asarray(sem_rel, dtype=np.float32)
        if rel_arr.shape != raw_idx.shape:
            raise ValueError(f"sem_rel shape {rel_arr.shape} must match sem_idx shape {raw_idx.shape}")
        if rel_arr.shape != (target_size, target_size):
            rel_arr = np.asarray(
                Image.fromarray(rel_arr).resize((target_size, target_size), Image.Resampling.NEAREST), dtype=np.float32
            )
    patch_feat, patch_rel = pool_semantic_patches(idx_arr, pal_arr, rel_arr, patch_size=16)
    feat_t = torch.from_numpy(patch_feat).unsqueeze(0).to(device)
    rel_t = torch.from_numpy(patch_rel).unsqueeze(0).to(device)
    return adapter.forward_map(feat_t), rel_t


@torch.no_grad()
def estimate_correspondences(
    model: Any,
    density_img: Image.Image,
    plan_img: Image.Image,
    *,
    sem_mode: SemMode = "none",
    semantic_adapter: SemanticAdapter | None = None,
    density_sem: SemanticMapInput | None = None,
    plan_sem: SemanticMapInput | None = None,
    lambda_sem: float = 0.5,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Reliable correspondences from the density map to the floorplan, in pixels (Sec. 3.3)."""
    if sem_mode not in ("none", "uniform", "consensus"):
        raise ValueError(f"Unsupported sem_mode: {sem_mode!r}")
    if lambda_sem < 0.0:
        raise ValueError(f"lambda_sem must be non-negative, got {lambda_sem}")
    if sem_mode != "none":
        if (density_sem is None) != (plan_sem is None):
            raise ValueError("Both density_sem and plan_sem must be provided together when sem_mode != 'none'")
        if semantic_adapter is not None and (density_sem is None or plan_sem is None):
            raise ValueError("density_sem and plan_sem are required when semantic_adapter is provided with sem_mode != 'none'")

    device = next(model.parameters()).device
    feature_maps = []
    for img in (density_img, plan_img):
        tokens = model.forward_features(norm_transform(img).unsqueeze(0).to(device))["x_norm_patchtokens"]  # type: ignore[attr-defined]
        B, N_tokens, C = tokens.shape
        grid_dim = int(np.sqrt(N_tokens))
        feature_maps.append(tokens.permute(0, 2, 1).reshape(B, C, grid_dim, grid_dim))
    map_d, map_p = feature_maps

    # Cosine similarity between query descriptors of the density map and all floorplan patches
    src_pts_norm = intensity_sampling(density_img).to(device)  # (1, N, 2)
    desc_d = F.normalize(sample_descriptors(map_d, src_pts_norm), dim=-1)  # (1, N, C)
    desc_sem_d: torch.Tensor | None = None
    map_sem_p: torch.Tensor | None = None
    rel_d: torch.Tensor | None = None
    map_rel_p: torch.Tensor | None = None
    if sem_mode != "none" and density_sem is not None and plan_sem is not None:
        if semantic_adapter is None:
            raise ValueError("semantic_adapter is required when sem_mode != 'none' and semantic maps are provided")
        was_training = semantic_adapter.training
        semantic_adapter.eval()
        try:
            map_sem_d, map_rel_d = _pool_and_adapt_map(density_sem, semantic_adapter, map_d.shape[-1] * 16, device)
            map_sem_p, map_rel_p = _pool_and_adapt_map(plan_sem, semantic_adapter, map_p.shape[-1] * 16, device)
            desc_sem_d = sample_descriptors(map_sem_d, src_pts_norm)
            if sem_mode == "consensus":
                rel_d = sample_descriptors(map_rel_d, src_pts_norm)
        finally:
            semantic_adapter.train(was_training)
    sim_matrix = compute_fused_sim_matrix(
        desc_d,
        map_p,
        sem_mode=sem_mode,
        desc_sem_d=desc_sem_d,
        map_sem_p=map_sem_p,
        rel_d=rel_d,
        map_rel_p=map_rel_p,
        lambda_sem=lambda_sem,
    )  # (1, N, HW)

    # Reliable correspondences: top 50% confidence (peak of the softmax) and mutual nearest neighbors
    prob = F.softmax(sim_matrix / SOFT_ARGMAX_TEMPERATURE, dim=-1)
    conf_vals = prob.max(dim=-1).values.squeeze(0)
    fwd_idx = sim_matrix.argmax(dim=-1).squeeze(0)  # density map -> floorplan
    bwd_idx = sim_matrix.argmax(dim=1).squeeze(0)  # floorplan -> density map
    mask_mnn = bwd_idx[fwd_idx] == torch.arange(len(fwd_idx), device=device)
    mask_conf = conf_vals >= torch.topk(conf_vals, int(len(conf_vals) * 0.5)).values[-1]
    final_mask = mask_conf & mask_mnn
    if final_mask.sum() < 10:  # fallback: confidence only
        final_mask = mask_conf

    # Sub-patch floorplan coordinates: expectation of the patch centers (soft-argmax)
    gy, gx = torch.meshgrid(torch.arange(grid_dim), torch.arange(grid_dim), indexing="ij")
    grid_coords = (torch.stack([gx, gy], dim=-1).float().to(device).reshape(-1, 2) + 0.5) / grid_dim  # [0, 1]
    dst_pts_norm = torch.matmul(prob, grid_coords).squeeze(0)  # (N, 2)
    src_px = (((src_pts_norm.squeeze(0) + 1) / 2)[final_mask].cpu().numpy() * EVAL_IMG_RES).astype(np.float32)
    dst_px = (dst_pts_norm[final_mask].cpu().numpy() * EVAL_IMG_RES).astype(np.float32)
    return src_px, dst_px


def align_to_floorplan(
    recon: ReconstructionDict | dict[str, Any],
    plan_img: Image.Image,
    model: Any,
    *,
    sem_mode: SemMode = "none",
    semantic_adapter: SemanticAdapter | None = None,
    photo_sem: SemanticMapInput | None = None,
    plan_sem: SemanticMapInput | None = None,
    lambda_sem: float = 0.5,
    tau: float = 5.0,
) -> AlignmentDict:
    """Density map of a reconstruction, its reliable correspondences and M (density map -> floorplan pixels)."""
    if sem_mode not in ("none", "uniform", "consensus"):
        raise ValueError(f"Unsupported sem_mode: {sem_mode!r}")
    if lambda_sem < 0.0:
        raise ValueError(f"lambda_sem must be non-negative, got {lambda_sem}")
    if sem_mode != "none":
        if (photo_sem is None) != (plan_sem is None):
            raise ValueError("Both photo_sem and plan_sem must be provided together when sem_mode != 'none'")
        if semantic_adapter is not None and (photo_sem is None or plan_sem is None):
            raise ValueError("photo_sem and plan_sem are required when semantic_adapter is provided with sem_mode != 'none'")
    proj_params, points, mask = reconstruct_projection_geometry(recon["points_cam0"], recon["conf"], recon["est_gravity_cam"])
    density_img = render_density_map(points, proj_params)
    density_sem: SemanticMapInput | None = None
    if sem_mode != "none" and photo_sem is not None and plan_sem is not None:
        if semantic_adapter is None:
            raise ValueError("semantic_adapter is required when sem_mode != 'none' and semantic maps are provided")
        sem_idx_grid, photo_palette, conf_grid = photo_sem
        den_sem_idx, den_sem_rel, den_palette, _ = project_posed_semantics(
            points,
            mask,
            sem_idx_grid,
            photo_palette,
            proj_params,
            conf_grid=conf_grid,
            tau=tau,
            training_rounding=False,
        )
        density_sem = (den_sem_idx, den_palette, den_sem_rel)

    src_px, dst_px = estimate_correspondences(
        model,
        density_img,
        plan_img,
        sem_mode=sem_mode,
        semantic_adapter=semantic_adapter,
        density_sem=density_sem,
        plan_sem=plan_sem,
        lambda_sem=lambda_sem,
    )
    M, _ = cv2.estimateAffinePartial2D(src_px, dst_px, method=cv2.RANSAC, ransacReprojThreshold=0.04 * EVAL_IMG_RES)
    if M is None:
        raise AlignmentFailureError("RANSAC failed to estimate the similarity transform.")
    return {"proj_params": proj_params, "density_img": density_img, "src_px": src_px, "dst_px": dst_px, "M": M}


def get_floorplan_aligned_scene(
    recon: ReconstructionDict | dict[str, Any],
    alignment: PoseAlignment,
    conf_threshold_percent: float = 90.0,
    y_percentiles: tuple[float, float] = (1.0, 99.0),
    xz_slice_val: float = 2.5,
) -> AlignedSceneDict:
    """3D points and cameras in the floorplan frame: horizontal coordinates follow M, heights use the scale of M.

    Frame: Z up, the floorplan on the unit square [-0.5, 0.5]^2 at z = 0 (512 pixels = 1, top edge at y = 0.5).
    The floor, the lower bound of the height filter, is at z = 0. The points are filtered for display by confidence,
    height and x/z percentiles. Cameras are camera-to-world poses (OpenCV convention).
    """
    proj_params, M = alignment["proj_params"], alignment["M"]
    R_rect, res = proj_params["R_rect"], proj_params["density_size"]
    s_M = np.sqrt(np.abs(np.linalg.det(M[:2, :2])))  # scale of M

    # Displayed points: confidence, height and x/z percentile filters in the ground frame (+Y down)
    points = (recon["points_cam0"] @ R_rect.T).reshape(-1, 3)
    conf = recon["conf"].reshape(-1)
    mask = conf > np.percentile(conf, conf_threshold_percent)
    points, colors = points[mask], recon["preprocessed_images"].reshape(-1, 3)[mask]
    y_min, y_floor = np.percentile(points[:, 1], y_percentiles)
    mask = (points[:, 1] >= y_min) & (points[:, 1] <= y_floor)
    points, colors = points[mask], colors[mask]
    x_min, x_max = np.percentile(points[:, 0], [xz_slice_val, 100 - xz_slice_val])
    z_min, z_max = np.percentile(points[:, 2], [xz_slice_val, 100 - xz_slice_val])
    mask = (points[:, 0] >= x_min) & (points[:, 0] <= x_max) & (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
    points, colors = points[mask], colors[mask]

    def to_floorplan_frame(p: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        plan_px = cv2.transform(get_density_coordinates(p, proj_params).reshape(-1, 1, 2), M).reshape(-1, 2)
        return np.stack([plan_px[:, 0] / res - 0.5, 0.5 - plan_px[:, 1] / res,
                         (y_floor - p[:, 1]) * proj_params["scale"] * s_M / res], axis=1)

    # Cameras: rotation of M about the vertical axis (the v-axis of the maps points down)
    F_flip = np.diag([1.0, -1.0])
    R2d = F_flip @ (M[:2, :2] / s_M) @ F_flip
    R_yaw = np.array([[R2d[0, 0], 0.0, R2d[0, 1]], [0.0, 1.0, 0.0], [R2d[1, 0], 0.0, R2d[1, 1]]])
    to_z_up = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])  # (x, y down, z) -> (x, z, -y)
    cam_to_world = np.tile(np.eye(4), (len(recon["extrinsics"]), 1, 1))
    cam_to_world[:, :3, :3] = to_z_up @ R_yaw @ R_rect @ recon["extrinsics"][:, :3, :3]
    cam_to_world[:, :3, 3] = to_floorplan_frame(recon["extrinsics"][:, :3, 3] @ R_rect.T)
    return {"points": to_floorplan_frame(points).astype(np.float32), "colors": colors,
            "cam_to_world": cam_to_world.astype(np.float32)}

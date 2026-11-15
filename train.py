"""Training of the LoRA layers on pairs of density maps and floorplans.

    python train.py --data <data> --output <run>                                                   # C3
    python train.py --data <data> --output <run> --epochs 20 --resolutions 512 --min_points 1000   # Structured3D
"""
from __future__ import annotations

import argparse
import csv
import os
import random
from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from scenealigner.inference import (SOFT_ARGMAX_TEMPERATURE, SemanticAdapter, SemMode, compute_fused_sim_matrix,
                                   load_model, sample_descriptors, save_semantic_adapter)
from scenealigner.semantics import CLIP_EMBED_DIM
from scenealigner.training import (DensityFloorplanDataset, collate, feature_matching_loss, geometry_loss,
                                   regression_loss, semantic_contrastive_loss, topology_loss)

if TYPE_CHECKING:
    from scenealigner.training import TrainingBatchDict

LAMBDA_REGR, LAMBDA_TOPO, LAMBDA_GEO = 50.0, 10.0, 10.0  # the weight of the feature matching loss is 1


def ramp(step: int, start: int, duration: int) -> float:
    """Weight of a loss that is introduced linearly from `start` over `duration` iterations."""
    if step < start:
        return 0.0
    if step >= start + duration:
        return 1.0
    return (step - start) / duration


def _model_device(model: Any, fallback: torch.device) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return fallback


def compute_loss(
    model: Any,
    batch: TrainingBatchDict,
    step: int,
    total_steps: int,
    *,
    semantic_adapter: SemanticAdapter | None = None,
    sem_mode: SemMode = "none",
    lambda_sem: float = 0.0,
    alpha_sem: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Loss of a batch at a training step, and its terms."""
    if sem_mode not in ("none", "uniform", "consensus"):
        raise ValueError(f"Unsupported sem_mode: {sem_mode!r}")
    if lambda_sem < 0.0:
        raise ValueError(f"lambda_sem must be non-negative, got {lambda_sem}")
    if alpha_sem < 0.0:
        raise ValueError(f"alpha_sem must be non-negative, got {alpha_sem}")
    device = _model_device(model, batch["density_img"].device)
    density_img = batch["density_img"].to(device)
    plan_img = batch["plan_img"].to(device)
    pts_density = batch["density_pts"].to(device)
    pts_plan = batch["plan_pts"].to(device)
    valid = batch["valid"].to(device)
    scene_ids = np.array(batch["scene_id"])
    same_scene = torch.tensor(
        (scene_ids[:, None] == scene_ids[None]) & ~np.eye(len(scene_ids), dtype=bool),
        device=device,
    )

    feature_maps = []
    grid_dim = density_img.shape[-1] // 16
    for img in (density_img, plan_img):
        tokens = model.forward_features(img)["x_norm_patchtokens"]
        grid_dim = img.shape[-1] // 16
        feature_maps.append(tokens.permute(0, 2, 1).reshape(len(img), -1, grid_dim, grid_dim))
    map_density, map_plan = feature_maps
    desc_density = sample_descriptors(map_density, pts_density)
    desc_plan = sample_descriptors(map_plan, pts_plan)

    use_sem_fusion = sem_mode != "none" and lambda_sem > 0.0
    use_sem_loss = alpha_sem > 0.0
    desc_sem_d: torch.Tensor | None = None
    desc_sem_p: torch.Tensor | None = None
    map_sem_p: torch.Tensor | None = None
    rel_d: torch.Tensor | None = None
    rel_p: torch.Tensor | None = None
    map_rel_p: torch.Tensor | None = None
    raw_map_sem_d: torch.Tensor | None = None
    raw_map_sem_p: torch.Tensor | None = None

    if use_sem_fusion or use_sem_loss:
        if semantic_adapter is None:
            raise ValueError("semantic_adapter is required when sem_mode != 'none' (with lambda_sem > 0) or alpha_sem > 0.")
        if (
            "density_sem_feat" not in batch
            or "plan_sem_feat" not in batch
            or "density_sem_rel" not in batch
            or "plan_sem_rel" not in batch
        ):
            raise ValueError("Batch is missing pooled semantic fields required for semantic training.")
        raw_map_sem_d = batch["density_sem_feat"].to(device)
        raw_map_sem_p = batch["plan_sem_feat"].to(device)
        map_sem_d = semantic_adapter.forward_map(raw_map_sem_d)
        map_sem_p = semantic_adapter.forward_map(raw_map_sem_p)
        desc_sem_d = sample_descriptors(map_sem_d, pts_density)
        desc_sem_p = sample_descriptors(map_sem_p, pts_plan)
        if sem_mode == "consensus":
            map_rel_d = batch["density_sem_rel"].to(device)
            map_rel_p = batch["plan_sem_rel"].to(device)
            rel_d = sample_descriptors(map_rel_d, pts_density).squeeze(-1)
            rel_p = sample_descriptors(map_rel_p, pts_plan).squeeze(-1)

    eff_lambda = lambda_sem if use_sem_fusion else 0.0
    losses = {
        "feat": feature_matching_loss(
            desc_density,
            desc_plan,
            same_scene,
            valid,
            sem_a=desc_sem_d if use_sem_fusion else None,
            sem_b=desc_sem_p if use_sem_fusion else None,
            rel_a=rel_d if use_sem_fusion else None,
            rel_b=rel_p if use_sem_fusion else None,
            lambda_sem=eff_lambda,
        )
    }
    loss = losses["feat"]

    # Curriculum: the regression loss comes in after 10 % of the iterations, the structural ones after 20 %
    w_regr = ramp(step, total_steps // 10, total_steps // 10)
    w_struct = ramp(step, total_steps // 5, total_steps // 10)
    if w_regr > 0:
        # Predicted floorplan coordinates in [0, 1]: soft-argmax over the patch centers
        if use_sem_fusion:
            sim_matrix = compute_fused_sim_matrix(
                desc_density,
                map_plan,
                sem_mode=sem_mode,
                desc_sem_d=desc_sem_d,
                map_sem_p=map_sem_p,
                rel_d=rel_d,
                map_rel_p=map_rel_p,
                lambda_sem=lambda_sem,
            )
        else:
            flat_plan = F.normalize(map_plan.flatten(2).permute(0, 2, 1), dim=-1)
            sim_matrix = torch.matmul(F.normalize(desc_density, dim=-1), flat_plan.transpose(1, 2))
        prob = F.softmax(sim_matrix / SOFT_ARGMAX_TEMPERATURE, dim=-1)
        gy, gx = torch.meshgrid(
            torch.arange(grid_dim, device=device),
            torch.arange(grid_dim, device=device),
            indexing="ij",
        )
        grid_coords = (torch.stack([gx, gy], dim=-1).float().reshape(-1, 2) + 0.5) / grid_dim
        pred = torch.matmul(prob, grid_coords)
        weights = prob.max(dim=-1).values.detach() * valid.float()  # confidence of the valid correspondences
        src = (pts_density + 1) / 2
        losses["regr"] = regression_loss(pred, (pts_plan + 1) / 2, weights)
        loss = loss + LAMBDA_REGR * w_regr * losses["regr"]
        if w_struct > 0:
            losses["geo"] = geometry_loss(src, pred, weights)
            loss = loss + LAMBDA_GEO * w_struct * losses["geo"]
            losses["topo"] = topology_loss(src, pred, weights)
            loss = loss + LAMBDA_TOPO * w_struct * losses["topo"]

    if use_sem_loss:
        assert desc_sem_d is not None and desc_sem_p is not None
        assert raw_map_sem_d is not None and raw_map_sem_p is not None
        raw_sem_d = sample_descriptors(raw_map_sem_d, pts_density)
        raw_sem_p = sample_descriptors(raw_map_sem_p, pts_plan)
        losses["sem"] = semantic_contrastive_loss(
            desc_sem_d,
            desc_sem_p,
            raw_sem_d,
            raw_sem_p,
            valid,
            rel_a=rel_d if sem_mode == "consensus" else None,
            rel_b=rel_p if sem_mode == "consensus" else None,
        )
        loss = loss + alpha_sem * losses["sem"]
    return loss, losses


def save_checkpoint(
    output_dir: str,
    model: Any,
    optimizer: torch.optim.Optimizer,
    step: int,
    *,
    semantic_adapter: SemanticAdapter | None = None,
    lambda_sem: float | None = 0.0,
) -> None:
    """Saves the LoRA adapter, optional semantic adapter, and an atomic training state checkpoint."""
    model.save_pretrained(output_dir)
    payload: dict[str, Any] = {
        "step": step,
        "lora": get_peft_model_state_dict(model),
        "optimizer": optimizer.state_dict(),
    }
    if semantic_adapter is not None:
        save_semantic_adapter(output_dir, semantic_adapter, lambda_sem=lambda_sem)
        payload["semantic_adapter"] = semantic_adapter.state_dict()
    ckpt_path = os.path.join(output_dir, "checkpoint.pt")
    tmp_path = f"{ckpt_path}.tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, ckpt_path)


def compute_epoch_schedule(dataset_len: int, epochs: int, batch_size: int = 4) -> tuple[int, int]:
    """Computes (steps_per_epoch, total_steps), rejecting empty datasets."""
    if dataset_len <= 0:
        raise ValueError("Training dataset is empty.")
    steps_per_epoch = (dataset_len + batch_size - 1) // batch_size
    return steps_per_epoch, steps_per_epoch * epochs


def should_save_mid_epoch(step: int, save_every: int, steps_per_epoch: int) -> bool:
    """True on mid-epoch save_every boundaries, avoiding duplicate saves at epoch end."""
    return save_every > 0 and step % save_every == 0 and step % steps_per_epoch != 0


def epoch_sample_order(dataset_len: int, epoch: int, seed: int) -> list[int]:
    """Deterministic per-epoch sample permutation so interrupted epochs resume identically."""
    return torch.randperm(dataset_len, generator=torch.Generator().manual_seed(seed + epoch)).tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="training data (prepare_c3.py or prepare_s3d.py)")
    parser.add_argument("--output", required=True, help="directory of the LoRA layers and the training log")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--resolutions", type=int, nargs="+", default=[512, 768, 1024, 1280], help="one per batch, at random")
    parser.add_argument("--min_points", type=int, default=10000, help="correspondences that a random crop must keep")
    parser.add_argument("--dinov3_weights", default=None, help="Meta's DINOv3 ViT-B/16 weights")
    parser.add_argument("--save_every", type=int, default=0, help="save a resumable checkpoint every N steps (0 = epoch end only)")
    parser.add_argument("--resume", action="store_true", help="resume from checkpoint.pt in --output if present")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sem_mode", choices=["none", "uniform", "consensus"], default="none")
    parser.add_argument("--lambda_sem", type=float, default=0.5, help="weight of semantic similarity in fused matching")
    parser.add_argument(
        "--alpha_sem",
        type=float,
        default=None,
        help="weight of auxiliary semantic contrastive loss (default: 0.2 when --sem_mode != none, else 0.0)",
    )
    parser.add_argument("--sem_dim", type=int, default=128, help="output dimensionality of SemanticAdapter")
    parser.add_argument("--init_lora", default="pre-trained", help="LoRA checkpoint dir/HF repo or 'pre-trained'")
    parser.add_argument("--freeze_lora", action="store_true", help="freeze LoRA weights and train only SemanticAdapter")
    args = parser.parse_args()

    sem_mode: SemMode = args.sem_mode
    if args.lambda_sem < 0.0:
        raise ValueError(f"--lambda_sem must be non-negative, got {args.lambda_sem}")
    alpha_sem = (0.2 if sem_mode != "none" else 0.0) if args.alpha_sem is None else float(args.alpha_sem)
    if alpha_sem < 0.0:
        raise ValueError(f"--alpha_sem must be non-negative, got {alpha_sem}")
    eff_lambda = float(args.lambda_sem) if sem_mode != "none" else 0.0
    if sem_mode != "none" and eff_lambda == 0.0 and alpha_sem == 0.0:
        raise ValueError("--sem_mode != 'none' requires --lambda_sem > 0 or --alpha_sem > 0 to supervise SemanticAdapter.")
    if any(r <= 0 or r % 16 != 0 for r in args.resolutions):
        raise ValueError(f"All --resolutions must be positive multiples of 16, got {args.resolutions}")
    use_semantics = (sem_mode != "none" and eff_lambda > 0.0) or alpha_sem > 0.0

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    image_size = args.resolutions[0] if len(args.resolutions) == 1 else None
    dataset = DensityFloorplanDataset(
        args.data, min_points=args.min_points, image_size=image_size, load_semantics=use_semantics
    )
    collate_fn = partial(collate, resolutions=args.resolutions)
    steps_per_epoch, total_steps = compute_epoch_schedule(len(dataset), args.epochs, batch_size=4)

    if args.init_lora == "pre-trained":
        lora = LoraConfig(r=16, lora_alpha=16, lora_dropout=0.1, target_modules=["qkv", "proj", "fc1", "fc2"])
        model = get_peft_model(load_model("pre-trained", args.dinov3_weights), lora)
    else:
        model = load_model(args.init_lora, args.dinov3_weights)
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.requires_grad_(not args.freeze_lora)
    if args.freeze_lora:
        for param in model.parameters():
            param.requires_grad_(False)
        model.eval()
    else:
        model.train()
    device = _model_device(model, torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    semantic_adapter: SemanticAdapter | None = None
    if use_semantics:
        semantic_adapter = SemanticAdapter(CLIP_EMBED_DIM, out_dim=args.sem_dim).to(device)
        semantic_adapter.train()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if semantic_adapter is not None:
        trainable_params.extend(p for p in semantic_adapter.parameters() if p.requires_grad)
    if not trainable_params:
        raise ValueError("No trainable parameters remain (cannot --freeze_lora without semantic training).")
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4)

    os.makedirs(args.output, exist_ok=True)
    ckpt_path = os.path.join(args.output, "checkpoint.pt")
    step = 0
    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        set_peft_model_state_dict(model, ckpt["lora"])
        if semantic_adapter is not None and "semantic_adapter" in ckpt:
            semantic_adapter.load_state_dict(ckpt["semantic_adapter"])
        optimizer.load_state_dict(ckpt["optimizer"])
        step = ckpt["step"]
        print(f"Resumed from {ckpt_path} at step {step}/{total_steps}")

    log_path = os.path.join(args.output, "log.csv")
    append_log = step > 0 and os.path.exists(log_path)
    loss_cols = ("feat", "regr", "geo", "topo", "sem") if use_semantics else ("feat", "regr", "geo", "topo")
    log = csv.writer(open(log_path, "a" if append_log else "w", buffering=1))
    if not append_log:
        log.writerow(["step", "loss", *loss_cols])

    start_epoch = step // steps_per_epoch
    saved_lambda = eff_lambda if sem_mode != "none" else None
    for epoch in range(start_epoch, args.epochs):
        skip_steps = step % steps_per_epoch if epoch == start_epoch else 0
        perm = epoch_sample_order(len(dataset), epoch=epoch, seed=args.seed)
        loader = DataLoader(Subset(dataset, perm[skip_steps * 4:]), batch_size=4, shuffle=False, num_workers=4, collate_fn=collate_fn)
        for batch in tqdm(loader, desc=f"epoch {epoch + 1}/{args.epochs}", initial=skip_steps, total=steps_per_epoch):
            optimizer.zero_grad()
            loss, losses = compute_loss(
                model,
                batch,
                step,
                total_steps,
                semantic_adapter=semantic_adapter,
                sem_mode=sem_mode,
                lambda_sem=eff_lambda,
                alpha_sem=alpha_sem,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            step += 1
            if step % 100 == 0:
                log.writerow([step, loss.item()] + [losses[k].item() if k in losses else "" for k in loss_cols])
            if should_save_mid_epoch(step, args.save_every, steps_per_epoch):
                save_checkpoint(
                    args.output, model, optimizer, step, semantic_adapter=semantic_adapter, lambda_sem=saved_lambda
                )
        save_checkpoint(args.output, model, optimizer, step, semantic_adapter=semantic_adapter, lambda_sem=saved_lambda)


if __name__ == "__main__":
    main()

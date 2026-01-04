#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage-3 (minimal loop): Train a stats-conditioned low-level feature generator for CIFAR10+CNNCifar.

Loss = lambda_high_mean * L_high_mean + alpha_teacher * L_teacher_ce + eta_div * L_div

Key design choices (as per user spec):
  - Teacher ensemble: all local client models that contain samples of class y; per-sample weights w_{y,u} ∝ count_{u,y}
  - High mean target: mean-then-norm (mu = normalize(mean(h_raw))) stored in Stage-2 global_stats['high']['class_means']
  - Teacher loss: CE on ensemble logits
  - Output: generator.pt (state_dict) that generates low_level_features_raw (flatten)
  - Logging: log.log + TensorBoard scalars for each loss term
"""

from __future__ import annotations

import os
import sys

# Ensure project root is on sys.path when running as a script:
#   python exps/run_stage3_lowgen_minloop.py ...
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import argparse
import json
import logging
import random
import time
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from lib.checkpoint import load_checkpoint
from lib.feature_generator import DiversityLoss, StatsConditionedFeatureGenerator, gather_by_label, stack_low_global_stats
from lib.models.models import CNNCifar
from lib.split_manager import load_split
from lib.utils import get_dataset


def _setup_logging(out_dir: str) -> logging.Logger:
    os.makedirs(out_dir, exist_ok=True)
    logger = logging.getLogger("stage3_lowgen_minloop")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(os.path.join(out_dir, "log.log"), encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


@torch.no_grad()
def _infer_feature_dims(model: nn.Module, device: str) -> Tuple[int, int]:
    """
    Infer (high_dim, low_dim) from a dummy forward. Assumes model forward returns:
      (logits, log_probs, high_level_features_raw, low_level_features_raw, projected_features)
    """
    model = model.to(device).eval()
    dummy = torch.randn(2, 3, 32, 32, device=device)
    out = model(dummy)
    if not (isinstance(out, tuple) and len(out) >= 4):
        raise ValueError("Unexpected model output format; expected tuple with >=4 elements.")
    high_dim = int(out[2].shape[1])
    low_dim = int(out[3].shape[1])
    return high_dim, low_dim


def _build_teacher_models(
    *,
    ckpt_state: Dict,
    num_users: int,
    args_like,
    device: str,
) -> List[CNNCifar]:
    local_sd = ckpt_state.get("local_models_full_state_dicts", None)
    if not isinstance(local_sd, dict):
        raise ValueError("Checkpoint missing state['local_models_full_state_dicts'].")

    teachers: List[CNNCifar] = []
    for cid in range(num_users):
        m = CNNCifar(args=args_like)
        key = cid
        if key not in local_sd and str(cid) in local_sd:
            key = str(cid)
        if key not in local_sd:
            raise KeyError(f"local_models_full_state_dicts missing client id={cid}")
        m.load_state_dict(local_sd[key], strict=True)
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
        m.to(device)
        teachers.append(m)
    return teachers


def _compute_client_class_counts(
    *,
    train_dataset,
    user_groups: Dict[int, List[int]],
    num_users: int,
    num_classes: int,
) -> torch.Tensor:
    """
    Returns counts tensor of shape (num_users, num_classes) in int64.
    This uses dataset labels without running models.
    """
    counts = torch.zeros((num_users, num_classes), dtype=torch.long)
    # CIFAR datasets return (img, label)
    for uid in range(num_users):
        idxs = user_groups[uid]
        # Fast path: iterate indices and read labels
        for j in idxs:
            _, y = train_dataset[int(j)]
            counts[uid, int(y)] += 1
    return counts


def _weights_for_batch_labels(
    y: torch.Tensor,
    *,
    counts: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    y: (B,) int64
    counts: (U, C) long
    Returns weight matrix W: (B, U) float32 where W[i,u] ∝ counts[u, y_i], normalized across u.
    If sum is 0 (shouldn't happen for CIFAR10), falls back to uniform across all teachers.
    """
    y = y.long()
    U, C = counts.shape
    B = y.shape[0]
    # gather counts for each label: (U, B)
    c = counts[:, y].to(torch.float32)  # (U,B)
    s = c.sum(dim=0, keepdim=True)  # (1,B)
    # mask where class is absent across all teachers
    zero_mask = (s <= eps)
    s = torch.where(zero_mask, torch.ones_like(s), s)
    w = (c / s).t().contiguous()  # (B,U)
    if torch.any(zero_mask):
        w[zero_mask.view(-1)] = (1.0 / float(U))
    return w


def _teacher_forward_from_low(
    teacher: CNNCifar,
    x_low: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Given low-level feature vector x_low (raw, unnormalized), compute:
      - h_raw: ReLU(fc0(x_low))               (B,120)
      - logits: fc2(ReLU(fc1(norm(h_raw))))   (B,num_classes)

    NOTE:
      This must match CNNCifar.forward() semantics when feeding REAL images:
        low_level_features_raw is the flatten output of (conv+pool) blocks (no normalize),
        then high path applies fc0->normalize->fc1->fc2.
      Therefore, x_low here should be the same "raw low feature" (not F.normalize(x_low)).
    """
    # Delegate to the canonical model API to avoid any normalization/activation drift.
    logits, _log_probs, h_raw, _proj = teacher.forward_from_low(x_low)
    return h_raw, logits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1_ckpt_path", type=str, required=True, help="Path to Stage-1 checkpoint (best-wo.pt / latest.pt)")
    parser.add_argument("--stage2_stats_path", type=str, required=True, help="Path to Stage-2 global_stats.pt")
    parser.add_argument("--split_path", type=str, default=None, help="Path to split.pkl (if None, infer from ckpt meta when possible)")
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory (default: <ckpt.meta.logdir>/stage3/lowgen_minloop)")

    parser.add_argument("--device", type=str, default=None, help="cuda | cpu (default: auto)")
    parser.add_argument("--seed", type=int, default=1234)

    # Generator training (defaults aligned with existing Stage-3)
    parser.add_argument("--gen_steps", type=int, default=2000)
    parser.add_argument("--gen_batch_size", type=int, default=256)
    parser.add_argument("--gen_lr", type=float, default=1e-3)

    # Generator architecture
    parser.add_argument("--gen_noise_dim", type=int, default=64)
    parser.add_argument("--gen_y_emb_dim", type=int, default=32)
    parser.add_argument("--gen_stat_emb_dim", type=int, default=128)
    parser.add_argument("--gen_hidden_dim", type=int, default=256)
    parser.add_argument("--gen_n_hidden_layers", type=int, default=2)
    parser.add_argument("--gen_relu_output", type=int, default=1)
    parser.add_argument("--gen_use_cov_diag", type=int, default=1)

    # Loss weights (minimal loop)
    parser.add_argument("--lambda_high_mean", type=float, default=1.0)
    parser.add_argument("--alpha_teacher", type=float, default=1.0)
    parser.add_argument("--eta_div", type=float, default=0.01)

    # Optional: also include low-stats matching from existing Stage-3 (off by default in minimal loop)
    parser.add_argument("--enable_low_stats_losses", type=int, default=0, help="If 1, also use low mean/var/rff losses from existing Stage-3")
    parser.add_argument("--gen_w_mean", type=float, default=1.0)
    parser.add_argument("--gen_w_var", type=float, default=0.1)
    parser.add_argument("--gen_w_rff", type=float, default=1.0)

    args = parser.parse_args()

    # Device
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load Stage-1 ckpt
    ckpt = load_checkpoint(args.stage1_ckpt_path, map_location="cpu")
    ckpt_meta = (ckpt.get("meta", {}) or {}) if isinstance(ckpt, dict) else {}
    ckpt_state = (ckpt.get("state", {}) or {}) if isinstance(ckpt, dict) else {}
    ckpt_args = (ckpt.get("args", {}) or {}) if isinstance(ckpt, dict) else {}

    dataset = str(ckpt_meta.get("dataset", ckpt_args.get("dataset", "cifar10")))
    if dataset != "cifar10":
        raise ValueError(f"This minimal script currently targets cifar10+CNNCifar. Got dataset={dataset}")
    num_users = int(ckpt_meta.get("num_users", ckpt_args.get("num_users", 20)))
    num_classes = int(ckpt_meta.get("num_classes", ckpt_args.get("num_classes", 10)))

    # Resolve split_path/out_dir
    if args.split_path is None:
        args.split_path = ckpt_meta.get("split_path", None) or ckpt_args.get("split_path", None)
    if not args.split_path or not os.path.exists(args.split_path):
        raise FileNotFoundError(f"split_path not found: {args.split_path}")

    logdir = ckpt_meta.get("logdir", None) or ckpt_args.get("log_dir", None) or os.path.dirname(os.path.abspath(args.stage1_ckpt_path))
    if args.out_dir is None:
        args.out_dir = os.path.join(logdir, "stage3", "lowgen_minloop")
    os.makedirs(args.out_dir, exist_ok=True)
    logger = _setup_logging(args.out_dir)
    tb_dir = os.path.join(args.out_dir, "tb")
    os.makedirs(tb_dir, exist_ok=True)
    writer = SummaryWriter(tb_dir)

    logger.info(f"device={args.device} dataset={dataset} num_users={num_users} num_classes={num_classes}")
    logger.info(f"stage1_ckpt_path={args.stage1_ckpt_path}")
    logger.info(f"stage2_stats_path={args.stage2_stats_path}")
    logger.info(f"split_path={args.split_path}")
    logger.info(f"out_dir={args.out_dir}")

    # Seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Build teachers
    args_like = argparse.Namespace(**{
        "num_classes": num_classes,
    })
    teachers = _build_teacher_models(ckpt_state=ckpt_state, num_users=num_users, args_like=args_like, device=args.device)
    logger.info(f"Loaded teacher models: {len(teachers)}")

    # Infer dims and assert CNNCifar low dim == 400 for CIFAR10 setting
    high_dim, low_dim = _infer_feature_dims(teachers[0], args.device)
    logger.info(f"Inferred dims: low_dim={low_dim}, high_dim={high_dim}")
    if low_dim != 16 * 5 * 5:
        logger.warning(f"Expected low_dim=400 for CNNCifar, got {low_dim}. Proceeding anyway.")

    # Load Stage-2 stats
    payload2 = torch.load(args.stage2_stats_path, map_location="cpu")
    meta2 = payload2.get("meta", {}) or {}
    state2 = payload2.get("state", {}) or {}
    global_stats = state2.get("global_stats", None)
    if global_stats is None:
        raise KeyError("Stage-2 payload missing state['global_stats']")
    if "high" not in global_stats or "low" not in global_stats:
        raise KeyError("Stage-2 global_stats missing 'high' or 'low'. Please rerun Stage-2 after enabling both-level stats.")

    # Target high means (mean_then_norm) for L_high_mean
    mu_high_list = global_stats["high"].get("class_means", None)
    if not (isinstance(mu_high_list, list) and len(mu_high_list) == num_classes):
        raise ValueError("Stage-2 global_stats['high']['class_means'] missing or invalid.")
    mu_high = torch.stack([m.detach().clone() for m in mu_high_list], dim=0).to(args.device)  # (C, high_dim)
    if mu_high.shape[1] != high_dim:
        raise ValueError(f"mu_high dim mismatch: expected {high_dim}, got {mu_high.shape[1]}")

    # Low stats for generator conditioning
    stats_low = stack_low_global_stats(global_stats)
    stats_low = type(stats_low)(
        mu=stats_low.mu.to(args.device),
        cov_diag=stats_low.cov_diag.to(args.device),
        rf_mean=stats_low.rf_mean.to(args.device),
        sample_per_class=stats_low.sample_per_class.to(args.device),
    )
    qualified = torch.nonzero(stats_low.sample_per_class > 0, as_tuple=False).view(-1)
    if qualified.numel() == 0:
        raise ValueError("No qualified classes with sample_per_class > 0.")

    # Compute teacher weights counts_{u,c} from split + dataset
    split = load_split(args.split_path)
    n_list = split["n_list"]
    k_list = split["k_list"]
    user_groups = split["user_groups"]
    # Build a minimal args object for get_dataset
    data_dir = ckpt_args.get("data_dir", "../data/")
    iid = int(ckpt_args.get("iid", 0))
    unequal = int(ckpt_args.get("unequal", 0))
    # get_dataset expects a Namespace-like args. Provide few-shot fields to keep
    # `lib/sampling.py` happy even when this stage-3 script is run standalone.
    ds_args = argparse.Namespace(
        dataset=dataset,
        data_dir=data_dir,
        iid=iid,
        unequal=unequal,
        num_users=num_users,
        num_classes=num_classes,
        train_shots_max=int(ckpt_args.get("train_shots_max", 110)),
        test_shots=int(ckpt_args.get("test_shots", 15)),
    )
    train_dataset, _, _, _, _, _ = get_dataset(ds_args, n_list, k_list)
    counts = _compute_client_class_counts(train_dataset=train_dataset, user_groups=user_groups, num_users=num_users, num_classes=num_classes).to(args.device)
    logger.info("Computed counts_{u,c} from split+train_dataset.")

    # Build generator
    gen = StatsConditionedFeatureGenerator(
        num_classes=num_classes,
        feature_dim=low_dim,
        noise_dim=int(args.gen_noise_dim),
        y_emb_dim=int(args.gen_y_emb_dim),
        stat_emb_dim=int(args.gen_stat_emb_dim),
        hidden_dim=int(args.gen_hidden_dim),
        n_hidden_layers=int(args.gen_n_hidden_layers),
        relu_output=int(args.gen_relu_output) == 1,
        use_cov_diag=int(args.gen_use_cov_diag) == 1,
    ).to(args.device)

    diversity_loss_fn = DiversityLoss(metric="l1").to(args.device)
    opt = torch.optim.Adam(gen.parameters(), lr=float(args.gen_lr))

    logger.info(
        f"Train gen: steps={args.gen_steps}, bs={args.gen_batch_size}, lr={args.gen_lr}, "
        f"lambda_high_mean={args.lambda_high_mean}, alpha_teacher={args.alpha_teacher}, eta_div={args.eta_div}, "
        f"enable_low_stats_losses={int(args.enable_low_stats_losses)}"
    )

    start_ts = time.time()
    for step in range(int(args.gen_steps)):
        # sample labels from qualified classes (uniform)
        idx = torch.randint(low=0, high=qualified.numel(), size=(int(args.gen_batch_size),), device=args.device)
        y = qualified[idx].long()

        mu_b, cov_diag_b, _rf_b = gather_by_label(stats_low, y)
        gen_res = gen(y, mu=mu_b, cov_diag=cov_diag_b, verbose=True)
        x_low = gen_res["output"]  # (B, low_dim)
        eps = gen_res["eps"]

        # teacher weights for this batch: (B, U)
        w_bu = _weights_for_batch_labels(y, counts=counts)  # float32

        # Teacher ensemble forward (keep grad wrt x_low, but teachers are frozen)
        h_ens = torch.zeros((x_low.shape[0], high_dim), device=args.device, dtype=x_low.dtype)
        logits_ens = torch.zeros((x_low.shape[0], num_classes), device=args.device, dtype=x_low.dtype)
        for u, t in enumerate(teachers):
            h_u, logits_u = _teacher_forward_from_low(t, x_low)
            wu = w_bu[:, u].unsqueeze(1).type_as(x_low)
            h_ens = h_ens + wu * h_u
            logits_ens = logits_ens + wu * logits_u

        # L_teacher (CE on ensemble logits)
        teacher_ce = F.cross_entropy(logits_ens, y, reduction="mean")

        # L_high_mean (mean_then_norm on h_raw ensemble)
        uniq = torch.unique(y)
        high_mean_loss = torch.tensor(0.0, device=args.device)
        n_used = 0
        for c in uniq.tolist():
            mask = (y == c)
            if not torch.any(mask):
                continue
            hs = h_ens[mask]
            mu_hat = hs.mean(dim=0)
            if torch.allclose(mu_hat, torch.zeros_like(mu_hat)):
                mu_hat_n = mu_hat
            else:
                mu_hat_n = F.normalize(mu_hat.unsqueeze(0), dim=1).squeeze(0)
            high_mean_loss = high_mean_loss + F.mse_loss(mu_hat_n, mu_high[c], reduction="mean")
            n_used += 1
        if n_used > 0:
            high_mean_loss = high_mean_loss / n_used

        # L_div
        div_loss = diversity_loss_fn(eps, x_low)

        # Optional: include low-level stats losses (existing Stage-3 terms)
        low_mean_loss = torch.tensor(0.0, device=args.device)
        low_var_loss = torch.tensor(0.0, device=args.device)
        low_rff_loss = torch.tensor(0.0, device=args.device)
        if int(args.enable_low_stats_losses) == 1:
            # Match low-level mean/var and RFF mean (if available)
            # Note: rf_mean exists in stats_low but this script does not reconstruct rf_model_low; keep rff off by default.
            uniq2 = torch.unique(y)
            n2 = 0
            for c in uniq2.tolist():
                mask = (y == c)
                if not torch.any(mask):
                    continue
                xs = x_low[mask]
                low_mean_loss = low_mean_loss + F.mse_loss(xs.mean(dim=0), stats_low.mu[c], reduction="mean")
                if xs.shape[0] >= 2:
                    v = xs.var(dim=0, unbiased=False)
                    low_var_loss = low_var_loss + F.l1_loss(v, stats_low.cov_diag[c], reduction="mean")
                n2 += 1
            if n2 > 0:
                low_mean_loss = low_mean_loss / n2
                low_var_loss = low_var_loss / n2

        loss = (
            float(args.lambda_high_mean) * high_mean_loss
            + float(args.alpha_teacher) * teacher_ce
            + float(args.eta_div) * div_loss
        )
        if int(args.enable_low_stats_losses) == 1:
            loss = loss + float(args.gen_w_mean) * low_mean_loss + float(args.gen_w_var) * low_var_loss + float(args.gen_w_rff) * low_rff_loss

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        # Logging
        if step % 25 == 0 or step == int(args.gen_steps) - 1:
            elapsed = time.time() - start_ts
            msg = (
                f"[minloop][{step:05d}/{int(args.gen_steps)}] "
                f"loss={loss.item():.6f} "
                f"high_mean={high_mean_loss.item():.6f} "
                f"teacher_ce={teacher_ce.item():.6f} "
                f"div={div_loss.item():.6f} "
                f"sec={elapsed:.1f}"
            )
            logger.info(msg)

        writer.add_scalar("loss/total", float(loss.item()), step)
        writer.add_scalar("loss/high_mean", float(high_mean_loss.item()), step)
        writer.add_scalar("loss/teacher_ce", float(teacher_ce.item()), step)
        writer.add_scalar("loss/div", float(div_loss.item()), step)
        if int(args.enable_low_stats_losses) == 1:
            writer.add_scalar("loss/low_mean", float(low_mean_loss.item()), step)
            writer.add_scalar("loss/low_var", float(low_var_loss.item()), step)
            writer.add_scalar("loss/low_rff", float(low_rff_loss.item()), step)
        writer.add_scalar("opt/lr", float(opt.param_groups[0]["lr"]), step)

    writer.flush()
    writer.close()

    # Save artifacts
    gen_path = os.path.join(args.out_dir, "generator.pt")
    torch.save(gen.state_dict(), gen_path)

    meta_out = {
        "stage": 3,
        "variant": "lowgen_minloop",
        "dataset": dataset,
        "num_users": num_users,
        "num_classes": num_classes,
        "low_feature_dim": low_dim,
        "high_feature_dim": high_dim,
        "stage1_ckpt_path": args.stage1_ckpt_path,
        "stage2_stats_path": args.stage2_stats_path,
        "split_path": args.split_path,
        "seed": args.seed,
        "losses": {
            "high_mean_mode": str(meta2.get("high_mean_mode", "unknown")),
            "lambda_high_mean": float(args.lambda_high_mean),
            "alpha_teacher": float(args.alpha_teacher),
            "eta_div": float(args.eta_div),
        },
        "train": {
            "gen_steps": int(args.gen_steps),
            "gen_batch_size": int(args.gen_batch_size),
            "gen_lr": float(args.gen_lr),
        },
        "gen_arch": {
            "gen_noise_dim": int(args.gen_noise_dim),
            "gen_y_emb_dim": int(args.gen_y_emb_dim),
            "gen_stat_emb_dim": int(args.gen_stat_emb_dim),
            "gen_hidden_dim": int(args.gen_hidden_dim),
            "gen_n_hidden_layers": int(args.gen_n_hidden_layers),
            "gen_relu_output": int(args.gen_relu_output),
            "gen_use_cov_diag": int(args.gen_use_cov_diag),
        },
        "teacher_ensemble": {
            "type": "all_teachers_with_counts_weighted_logits",
            "weight_source": "counts_{u,c}_from_split",
        },
        "enable_low_stats_losses": int(args.enable_low_stats_losses),
    }
    with open(os.path.join(args.out_dir, "generator_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta_out, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved generator state_dict to: {gen_path}")
    logger.info(f"Saved generator meta to: {os.path.join(args.out_dir, 'generator_meta.json')}")


if __name__ == "__main__":
    main()



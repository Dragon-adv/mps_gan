#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage-4 (CIFAR10): client-side finetune of high-level encoder + classifier head
using a mixture of real and synthetic low-level features.

Design:
  - Load Stage-1 checkpoint (local client models)
  - Load Stage-2 global_stats.pt (class-wise low-level mu/cov_diag)
  - Load Stage-3 generator.pt (StatsConditionedFeatureGenerator)
  - For each client:
      * Freeze low encoder (conv1/conv2) and projector
      * Train only fc0 + fc1 + fc2 using:
          - real low_level_features_raw extracted from real images
          - synthetic low features from generator with labels sampled from real batch labels
          - (optional) OOC synthetic low features from client-missing classes for fc0-only regularization
      * Evaluate on real client test split (images) using full forward()
  - Save finetuned weights per client

Usage (example):
  python exps/run_stage4.py ^
    --stage1_ckpt_path ..\\newresults\\ours\\<LOGDIR>\\stage1_ckpts\\best-wo.pt ^
    --stage2_stats_path ..\\newresults\\ours\\<LOGDIR>\\stage2_stats\\global_stats.pt ^
    --gen_path ..\\newresults\\ours\\<LOGDIR>\\stage3_gen\\generator.pt ^
    --syn_ratio 0.2 --steps 2000 --batch_size 128 --lr 1e-4 --gpu 0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import random
import time
import logging
from datetime import datetime
from dataclasses import asdict
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

# Ensure project root is on sys.path when running as a script:
#   python exps/run_stage4.py ...
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from lib.checkpoint import load_checkpoint
from lib.feature_generator import StatsConditionedFeatureGenerator, gather_by_label, stack_low_global_stats
from lib.models.models import CNNCifar
from lib.split_manager import load_split
from lib.utils import get_dataset


def _resolve_device(gpu: int) -> str:
    if torch.cuda.is_available():
        torch.cuda.set_device(int(gpu))
        return "cuda"
    return "cpu"


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _format_float_for_tag(x: float) -> str:
    """
    Format a float into a short, filesystem-friendly token.
    Example: 0.15 -> "0p15", 1e-4 -> "0p0001"
    """
    s = f"{float(x):.6g}"  # compact, stable
    s = s.replace("+", "")
    return s.replace(".", "p")


def _make_run_tag(args: argparse.Namespace) -> str:
    """
    Deterministic tag from key hparams to avoid accidental overwrite across runs.
    """
    return (
        f"syn{_format_float_for_tag(args.syn_ratio)}"
        f"_ooc{_format_float_for_tag(args.ooc_ratio)}"
        f"_lam{_format_float_for_tag(args.ooc_lambda)}"
        f"_{str(args.ooc_mode)}"
        f"_steps{int(args.steps)}"
        f"_bs{int(args.batch_size)}"
        f"_lr{_format_float_for_tag(args.lr)}"
        f"_wd{_format_float_for_tag(args.weight_decay)}"
        f"_seed{int(args.seed)}"
    )


def _setup_logger(out_dir: str) -> logging.Logger:
    """
    Tee logs to both console and <out_dir>/stage4.log.
    """
    os.makedirs(out_dir, exist_ok=True)
    logger = logging.getLogger("stage4")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Ensure file handler always points to the current out_dir (even if reused in same process).
    if logger.handlers:
        for h in list(logger.handlers):
            try:
                h.flush()
                h.close()
            except Exception:
                pass
            logger.removeHandler(h)

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(os.path.join(out_dir, "stage4.log"), encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def _as_int_indices(idxs) -> List[int]:
    """
    Convert split indices (possibly numpy float arrays from older split.pkl)
    into a clean Python List[int] suitable for torch.utils.data.Subset.
    """
    def _flatten(v) -> List[int]:
        if v is None:
            return []

        # torch Tensor
        if torch.is_tensor(v):
            v = v.detach().cpu()
            if v.numel() == 0:
                return []
            if v.numel() == 1:
                return [int(v.item())]
            return [int(x) for x in v.flatten().tolist()]

        # numpy ndarray / scalar
        if isinstance(v, np.ndarray):
            if v.size == 0:
                return []
            if v.ndim == 0:
                return _flatten(v.item())
            # Object arrays can contain nested lists/arrays; flatten recursively.
            if v.dtype == object:
                out: List[int] = []
                for e in v.tolist():
                    out.extend(_flatten(e))
                return out
            if np.issubdtype(v.dtype, np.floating):
                if not np.all(np.isfinite(v)):
                    raise ValueError("Found non-finite indices in split")
                if not np.allclose(v, np.round(v)):
                    raise ValueError("Found non-integer float indices in split")
                v = np.round(v)
            v = v.astype(np.int64, copy=False).reshape(-1)
            return [int(x) for x in v.tolist()]

        # Python containers
        if isinstance(v, (list, tuple, set)):
            out: List[int] = []
            for e in v:
                out.extend(_flatten(e))
            return out

        # numpy / python scalars
        if isinstance(v, (np.integer, int)):
            return [int(v)]
        if isinstance(v, (np.floating, float)):
            fv = float(v)
            if not np.isfinite(fv):
                raise ValueError("Found non-finite float index in split")
            if not float(fv).is_integer():
                raise ValueError(f"Found non-integer float index in split: {fv}")
            return [int(round(fv))]

        raise TypeError(f"Unsupported index type in split: {type(v)}")

    out = _flatten(idxs)
    # Final safety: must be plain Python ints, not tensors/ndarrays
    if any(not isinstance(x, int) for x in out):
        bad = [type(x) for x in out if not isinstance(x, int)][:5]
        raise TypeError(f"Indices normalization failed; non-int types remain: {bad}")
    return out


@torch.no_grad()
def _eval_client_on_images(
    model: nn.Module,
    dl: DataLoader,
    device: str,
    num_classes: int,
) -> Tuple[float, float]:
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    for images, labels in dl:
        images = images.to(device)
        labels = labels.to(device)
        logits, _log_probs, _h_raw, _l_raw, _proj = model(images)
        logits = logits[:, 0:num_classes]
        loss = F.cross_entropy(logits, labels, reduction="sum")
        loss_sum += float(loss.item())
        pred = logits.argmax(dim=1)
        correct += int((pred == labels).sum().item())
        total += int(labels.numel())
    acc = (correct / total) if total > 0 else 0.0
    avg_loss = (loss_sum / total) if total > 0 else 0.0
    return acc, avg_loss


def _load_generator_hparams(gen_path: str, cli_args: argparse.Namespace) -> Dict[str, int]:
    """
    Try to load generator_meta.json from the same directory, otherwise fall back to CLI/defaults.
    """
    d = os.path.dirname(os.path.abspath(gen_path))
    meta_path = os.path.join(d, "generator_meta.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            h = (meta.get("gen_hparams", {}) or {})
            return {
                "gen_noise_dim": int(h.get("gen_noise_dim", cli_args.gen_noise_dim)),
                "gen_y_emb_dim": int(h.get("gen_y_emb_dim", cli_args.gen_y_emb_dim)),
                "gen_stat_emb_dim": int(h.get("gen_stat_emb_dim", cli_args.gen_stat_emb_dim)),
                "gen_hidden_dim": int(h.get("gen_hidden_dim", cli_args.gen_hidden_dim)),
                "gen_n_hidden_layers": int(h.get("gen_n_hidden_layers", cli_args.gen_n_hidden_layers)),
                "gen_relu_output": int(h.get("gen_relu_output", cli_args.gen_relu_output)),
                "gen_use_cov_diag": int(h.get("gen_use_cov_diag", cli_args.gen_use_cov_diag)),
            }
        except Exception:
            pass
    return {
        "gen_noise_dim": int(cli_args.gen_noise_dim),
        "gen_y_emb_dim": int(cli_args.gen_y_emb_dim),
        "gen_stat_emb_dim": int(cli_args.gen_stat_emb_dim),
        "gen_hidden_dim": int(cli_args.gen_hidden_dim),
        "gen_n_hidden_layers": int(cli_args.gen_n_hidden_layers),
        "gen_relu_output": int(cli_args.gen_relu_output),
        "gen_use_cov_diag": int(cli_args.gen_use_cov_diag),
    }


def _resolve_existing_path(path_like: str) -> str:
    """
    Normalize a user-provided path:
      - expanduser
      - if relative, resolve from current working directory
    """
    if path_like is None:
        return path_like
    p = os.path.expanduser(str(path_like))
    if not os.path.isabs(p):
        p = os.path.abspath(p)
    return p


def _resolve_generator_path(
    gen_path_cli: str,
    *,
    logdir: str,
) -> str:
    """
    Resolve generator.pt path.
    - If user passes a directory, look for generator.pt inside it.
    - If user passes a non-existing path, try common candidates under logdir:
        1) <logdir>/stage3_gen/generator.pt
        2) <logdir>/stage3_lowgen_minloop/generator.pt
    """
    gen_path_cli = _resolve_existing_path(gen_path_cli)

    # If a directory is provided, assume generator.pt inside it.
    if gen_path_cli and os.path.isdir(gen_path_cli):
        cand = os.path.join(gen_path_cli, "generator.pt")
        if os.path.exists(cand):
            return cand
        raise FileNotFoundError(f"gen_path is a directory but generator.pt not found: {cand}")

    # If exact file exists, use it.
    if gen_path_cli and os.path.exists(gen_path_cli):
        return gen_path_cli

    # Fallback to typical locations under logdir.
    cands = [
        os.path.join(logdir, "stage3_gen", "generator.pt"),
        os.path.join(logdir, "stage3_lowgen_minloop", "generator.pt"),
    ]
    existing = [p for p in cands if os.path.exists(p)]
    if len(existing) == 1:
        print(f"[stage4] Warning: --gen_path not found. Fallback to: {existing[0]}")
        return existing[0]
    if len(existing) > 1:
        raise FileNotFoundError(
            "Multiple generator.pt candidates found under logdir; please pass --gen_path explicitly:\n"
            + "\n".join(existing)
        )

    raise FileNotFoundError(
        "generator.pt not found. Checked:\n"
        f"  - {gen_path_cli}\n"
        + "\n".join([f"  - {p}" for p in cands])
    )


def _assert_optimizer_params(optim: torch.optim.Optimizer, allowed: List[torch.nn.Parameter]) -> None:
    allowed_ids = {id(p) for p in allowed}
    for g in optim.param_groups:
        for p in g.get("params", []):
            if id(p) not in allowed_ids:
                raise RuntimeError("Optimizer contains parameters outside (fc0, fc1, fc2).")

def _any_grad(params: List[torch.nn.Parameter]) -> bool:
    for p in params:
        if p.grad is not None:
            return True
    return False


def _assert_freeze_setup(model: CNNCifar) -> None:
    """
    Stage-4 contract:
      - conv1/conv2 + projector are frozen (requires_grad=False)
      - fc0/fc1/fc2 are trainable (requires_grad=True)
      - model must implement forward_from_low
    """
    if not (hasattr(model, "forward_from_low") and callable(getattr(model, "forward_from_low"))):
        raise RuntimeError("Model missing required API: forward_from_low(x_low).")
    # low encoder
    for p in list(model.conv1.parameters()) + list(model.conv2.parameters()):
        if p.requires_grad:
            raise RuntimeError("Freeze check failed: conv1/conv2 should have requires_grad=False in Stage-4.")
    # projector
    for p in model.projector.parameters():
        if p.requires_grad:
            raise RuntimeError("Freeze check failed: projector should have requires_grad=False in Stage-4.")
    # trainable parts
    for p in model.fc0.parameters():
        if not p.requires_grad:
            raise RuntimeError("Trainable check failed: fc0 should have requires_grad=True in Stage-4.")
    for p in model.fc1.parameters():
        if not p.requires_grad:
            raise RuntimeError("Trainable check failed: fc1 should have requires_grad=True in Stage-4.")
    for p in model.fc2.parameters():
        if not p.requires_grad:
            raise RuntimeError("Trainable check failed: fc2 should have requires_grad=True in Stage-4.")


def _assert_grad_flow(model: CNNCifar) -> None:
    """
    After backward():
      - fc0/fc1/fc2 should receive gradients
      - conv1/conv2/projector must not receive gradients
    """
    if not _any_grad(list(model.fc0.parameters())):
        raise RuntimeError("Grad check failed: fc0 has no gradients (expected trainable).")
    if not _any_grad(list(model.fc1.parameters())):
        raise RuntimeError("Grad check failed: fc1 has no gradients (expected trainable).")
    if not _any_grad(list(model.fc2.parameters())):
        raise RuntimeError("Grad check failed: fc2 has no gradients (expected trainable).")

    if _any_grad(list(model.conv1.parameters())) or _any_grad(list(model.conv2.parameters())):
        raise RuntimeError("Grad check failed: conv1/conv2 received gradients but should be frozen.")
    if _any_grad(list(model.projector.parameters())):
        raise RuntimeError("Grad check failed: projector received gradients but should be frozen.")


def _ooc_high_mean_loss_fc0_only(
    model: CNNCifar,
    x_low_ooc: torch.Tensor,
    y_ooc: torch.Tensor,
    *,
    mu_high: torch.Tensor,
) -> torch.Tensor:
    """
    OOC (out-of-client-class) feature-space regularization for fc0 only.

    Align normalized high features produced by fc0(x_low_ooc) to Stage-2 global high means mu_high[y].
    This does NOT train fc1/fc2 to classify OOC labels.
    """
    y_ooc = y_ooc.long()
    # High-level encoder
    high_raw = F.relu(model.fc0(x_low_ooc))
    high = F.normalize(high_raw, dim=1)

    # Target global high mean (expected to be mean-then-norm; normalize again for safety)
    target = mu_high.to(high.device)[y_ooc]
    target = F.normalize(target, dim=1)

    # 1 - cosine similarity
    return (1.0 - (high * target).sum(dim=1)).mean()


def _ooc_uniform_loss_fc0_only(model: CNNCifar, x_low_ooc: torch.Tensor, *, num_classes: int) -> torch.Tensor:
    """
    OOC (out-of-client-class) unlabeled regularization.

    Intended behavior:
      - Let high-level encoder (fc0) "see" OOC feature distributions
      - Do NOT train classifier head (fc1/fc2) to predict those OOC labels

    We implement a uniform-target loss on predictions, but compute logits using detached
    fc1/fc2 parameters so gradients flow ONLY into fc0 (via the normalized high features).
    """
    # High-level encoder (trainable)
    high_raw = F.relu(model.fc0(x_low_ooc))
    high = F.normalize(high_raw, dim=1)

    # Classifier head forward with detached params (no grads to fc1/fc2)
    h1 = F.relu(F.linear(high, model.fc1.weight.detach(), model.fc1.bias.detach()))
    logits = F.linear(h1, model.fc2.weight.detach(), model.fc2.bias.detach())
    logits = logits[:, 0:int(num_classes)]

    logp = F.log_softmax(logits, dim=1)  # (B, C)
    # Cross-entropy to uniform target: -mean_c log p_c
    return -(logp.mean(dim=1)).mean()


def main() -> int:
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("--stage1_ckpt_path", type=str, required=True, help="Stage-1 checkpoint path (best-wo.pt / latest.pt)")
    p.add_argument("--stage2_stats_path", type=str, required=True, help="Stage-2 global_stats.pt path")
    p.add_argument("--gen_path", type=str, required=True, help="Stage-3 generator.pt path")
    p.add_argument("--split_path", type=str, default=None, help="split.pkl path (if None, infer from ckpt meta/args)")
    p.add_argument("--out_dir", type=str, default=None, help="Output dir (default: <ckpt.meta.logdir>/stage4_finetune)")
    p.add_argument(
        "--auto_run_dir",
        type=int,
        default=1,
        help="If 1, create a unique subdir under out_dir/stage4_finetune to avoid overwrites (default: 1).",
    )
    p.add_argument("--run_name", type=str, default=None, help="Optional run name suffix for output dir.")
    p.add_argument(
        "--run_tag",
        type=str,
        default=None,
        help="Optional run tag. If None, auto-generated from hparams (recommended).",
    )
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=1234)

    # Stage-4 training hyperparams
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--syn_ratio", type=float, default=0.2, help="Synthetic ratio within each step (0..1)")
    p.add_argument(
        "--ooc_ratio",
        type=float,
        default=0.0,
        help="Within synthetic batch, ratio sampled from client-missing classes (OOC). Default 0 disables OOC.",
    )
    p.add_argument(
        "--ooc_lambda",
        type=float,
        default=0.2,
        help="Weight for OOC unlabeled uniform loss (applied to fc0 only).",
    )
    p.add_argument(
        "--ooc_mode",
        type=str,
        default="high_mean",
        choices=["high_mean", "uniform"],
        help="OOC regularization mode. 'high_mean' aligns fc0 features to Stage-2 global high means; "
        "'uniform' pushes predictions to uniform using detached head params (fc0-only grads).",
    )
    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--eval_interval", type=int, default=200)
    p.add_argument(
        "--save_best",
        type=int,
        default=1,
        help="If 1, track and save per-client best checkpoint during stage-4 based on test acc (default: 1).",
    )
    p.add_argument(
        "--select_best_for_after",
        type=int,
        default=1,
        help="If 1, report acc_after/loss_after as the best checkpoint metrics; otherwise report final-step metrics (default: 1).",
    )

    # Generator architecture (used if generator_meta.json is missing)
    p.add_argument("--gen_noise_dim", type=int, default=64)
    p.add_argument("--gen_y_emb_dim", type=int, default=32)
    p.add_argument("--gen_stat_emb_dim", type=int, default=128)
    p.add_argument("--gen_hidden_dim", type=int, default=256)
    p.add_argument("--gen_n_hidden_layers", type=int, default=2)
    p.add_argument("--gen_relu_output", type=int, default=1)
    p.add_argument("--gen_use_cov_diag", type=int, default=1)

    args = p.parse_args()

    device = _resolve_device(args.gpu)
    _seed_all(int(args.seed))

    # Load Stage-1 ckpt
    ckpt = load_checkpoint(args.stage1_ckpt_path, map_location="cpu")
    ckpt_meta = (ckpt.get("meta", {}) or {}) if isinstance(ckpt, dict) else {}
    ckpt_state = (ckpt.get("state", {}) or {}) if isinstance(ckpt, dict) else {}
    ckpt_args = (ckpt.get("args", {}) or {}) if isinstance(ckpt, dict) else {}

    dataset = str(ckpt_meta.get("dataset", ckpt_args.get("dataset", "cifar10")))
    if dataset != "cifar10":
        raise ValueError(f"Stage-4 currently supports cifar10 only. Got dataset={dataset}")
    num_users = int(ckpt_meta.get("num_users", ckpt_args.get("num_users", 20)))
    num_classes = int(ckpt_meta.get("num_classes", ckpt_args.get("num_classes", 10)))

    # Resolve split_path
    split_path = args.split_path or ckpt_meta.get("split_path", None) or ckpt_args.get("split_path", None)
    if not split_path or not os.path.exists(split_path):
        raise FileNotFoundError(f"split_path not found: {split_path}")

    # Resolve logdir/out_dir
    logdir = ckpt_meta.get("logdir", None) or ckpt_args.get("log_dir", None) or os.path.dirname(os.path.abspath(args.stage1_ckpt_path))
    base_out_dir = args.out_dir or os.path.join(logdir, "stage4_finetune")
    run_tag = str(args.run_tag).strip() if args.run_tag is not None else _make_run_tag(args)
    if args.run_name:
        run_tag = f"{run_tag}_{str(args.run_name).strip()}"

    if int(args.auto_run_dir) == 1:
        out_dir = os.path.join(base_out_dir, run_tag)
        # If non-empty dir exists (repeat run), add timestamp to avoid overwrite.
        if os.path.exists(out_dir) and os.listdir(out_dir):
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            out_dir = f"{out_dir}_{ts}"
    else:
        out_dir = base_out_dir

    os.makedirs(out_dir, exist_ok=True)
    logger = _setup_logger(out_dir)
    logger.info(f"[stage4] out_dir={out_dir}")
    logger.info(f"[stage4] run_tag={run_tag} auto_run_dir={int(args.auto_run_dir)} run_name={args.run_name}")
    logger.info(f"[stage4] cli_args={vars(args)}")

    # Resolve generator path (support stage3_gen and stage3_lowgen_minloop)
    gen_path = _resolve_generator_path(args.gen_path, logdir=logdir)
    logger.info(f"[stage4] resolved_paths: stage1={args.stage1_ckpt_path} stage2={args.stage2_stats_path} gen={gen_path} split={split_path}")

    # Load split + datasets
    split = load_split(split_path)
    n_list = split["n_list"]
    k_list = split["k_list"]
    user_groups: Dict[int, List[int]] = split["user_groups"]
    user_groups_lt: Dict[int, List[int]] = split["user_groups_lt"]

    # Build args namespace for get_dataset (reuse ckpt args as much as possible)
    base = dict(ckpt_args)
    base["dataset"] = "cifar10"
    base["num_classes"] = num_classes
    base["num_users"] = num_users
    base["gpu"] = int(args.gpu)
    base["device"] = device
    args_ds = SimpleNamespace(**base)
    train_dataset, test_dataset, _, _, _, _ = get_dataset(args_ds, n_list, k_list)

    # Load Stage-2 stats
    payload2 = torch.load(args.stage2_stats_path, map_location="cpu")
    state2 = payload2.get("state", {}) or {}
    global_stats = state2.get("global_stats", None)
    if global_stats is None or "low" not in global_stats:
        raise KeyError("Stage-2 global_stats missing 'low'. Please rerun Stage-2 with low stats enabled.")
    stats_low = stack_low_global_stats(global_stats)
    stats_low = type(stats_low)(
        mu=stats_low.mu.to(device),
        cov_diag=stats_low.cov_diag.to(device),
        rf_mean=stats_low.rf_mean.to(device),
        sample_per_class=stats_low.sample_per_class.to(device),
    )

    # Optional: load Stage-2 global high means for OOC regularization (recommended)
    mu_high = None
    if float(args.ooc_ratio) > 0.0 and str(args.ooc_mode).lower() == "high_mean":
        if "high" not in global_stats:
            raise KeyError("OOC high_mean mode requires Stage-2 global_stats['high'].")
        mu_high_list = global_stats["high"].get("class_means", None)
        if not (isinstance(mu_high_list, list) and len(mu_high_list) == int(num_classes)):
            raise KeyError("OOC high_mean mode requires global_stats['high']['class_means'] as a list of length num_classes.")
        mu_high = torch.stack([m.detach().clone() for m in mu_high_list], dim=0).to(device)

    # Build generator
    # Infer low_feature_dim from stats (canonical)
    low_feature_dim = int(stats_low.mu.shape[1])
    gen_h = _load_generator_hparams(gen_path, args)
    gen = StatsConditionedFeatureGenerator(
        num_classes=num_classes,
        feature_dim=low_feature_dim,
        noise_dim=int(gen_h["gen_noise_dim"]),
        y_emb_dim=int(gen_h["gen_y_emb_dim"]),
        stat_emb_dim=int(gen_h["gen_stat_emb_dim"]),
        hidden_dim=int(gen_h["gen_hidden_dim"]),
        n_hidden_layers=int(gen_h["gen_n_hidden_layers"]),
        relu_output=int(gen_h["gen_relu_output"]) == 1,
        use_cov_diag=int(gen_h["gen_use_cov_diag"]) == 1,
    ).to(device)
    gen.load_state_dict(torch.load(gen_path, map_location="cpu"), strict=True)
    gen.eval()
    for pgen in gen.parameters():
        pgen.requires_grad_(False)

    # Load local client models from Stage-1 ckpt
    local_sd = ckpt_state.get("local_models_full_state_dicts", None)
    if not isinstance(local_sd, dict):
        raise ValueError("Stage-1 checkpoint missing state['local_models_full_state_dicts']")

    # Persist a meta file for reproducibility
    meta_path = os.path.join(out_dir, "stage4_meta.json")
    meta_out = {
        "stage": 4,
        "dataset": "cifar10",
        "num_users": num_users,
        "num_classes": num_classes,
        "stage1_ckpt_path": args.stage1_ckpt_path,
        "stage2_stats_path": args.stage2_stats_path,
        "gen_path": gen_path,
        "split_path": split_path,
        "out_dir": out_dir,
        "run_tag": run_tag,
        "run_name": args.run_name,
        "auto_run_dir": int(args.auto_run_dir),
        "hparams": {
            "steps": int(args.steps),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "syn_ratio": float(args.syn_ratio),
            "ooc_ratio": float(args.ooc_ratio),
            "ooc_lambda": float(args.ooc_lambda),
            "ooc_mode": str(args.ooc_mode),
            "seed": int(args.seed),
        },
        "gen_hparams": gen_h,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_out, f, ensure_ascii=False, indent=2)

    # Stage-4 training loop per client
    syn_ratio = float(args.syn_ratio)
    if not (0.0 <= syn_ratio <= 1.0):
        raise ValueError("--syn_ratio must be within [0,1]")
    if not (0.0 <= float(args.ooc_ratio) <= 1.0):
        raise ValueError("--ooc_ratio must be within [0,1]")
    if int(args.save_best) not in (0, 1):
        raise ValueError("--save_best must be 0 or 1")
    if int(args.select_best_for_after) not in (0, 1):
        raise ValueError("--select_best_for_after must be 0 or 1")

    # Pre-compute client-missing classes (OOC candidates) from split.pkl (prefer classes_list if present)
    classes_list = split.get("classes_list", None)
    # Only sample OOC from classes that exist globally (sample_per_class>0), otherwise generator conditioning may be meaningless
    qualified_global = set(torch.nonzero(stats_low.sample_per_class > 0, as_tuple=False).view(-1).detach().cpu().tolist())
    all_classes = set(range(int(num_classes)))
    client_missing: Dict[int, List[int]] = {}
    if classes_list is not None:
        for cid in range(int(num_users)):
            if isinstance(classes_list, dict):
                present = set(int(x) for x in classes_list.get(int(cid), []))
            else:
                present = set(int(x) for x in classes_list[int(cid)])
            missing = sorted(list((all_classes - present) & qualified_global))
            client_missing[int(cid)] = missing
    else:
        # Fallback: compute from train_dataset.targets and split indices
        targets = getattr(train_dataset, "targets", None)
        if targets is None:
            raise AttributeError("train_dataset has no 'targets'; cannot infer client missing classes without split['classes_list'].")
        for cid in range(int(num_users)):
            idxs = _as_int_indices(user_groups[int(cid)])
            present = set(int(targets[i]) for i in idxs)
            missing = sorted(list((all_classes - present) & qualified_global))
            client_missing[int(cid)] = missing

    results: List[Dict[str, object]] = []
    start_all = time.time()

    for cid in range(num_users):
        # Build model for this client and load weights
        m = CNNCifar(args=SimpleNamespace(**{"num_classes": num_classes})).to(device)
        key = cid if cid in local_sd else str(cid)
        if key not in local_sd:
            raise KeyError(f"local_models_full_state_dicts missing client id={cid}")
        m.load_state_dict(local_sd[key], strict=True)

        # Freeze low encoder + projector; train only fc0/fc1/fc2
        for p in m.parameters():
            p.requires_grad_(False)
        for p in m.fc0.parameters():
            p.requires_grad_(True)
        for p in m.fc1.parameters():
            p.requires_grad_(True)
        for p in m.fc2.parameters():
            p.requires_grad_(True)

        _assert_freeze_setup(m)

        # DataLoaders (client train/test splits)
        train_idxs = _as_int_indices(user_groups[int(cid)])
        test_idxs = _as_int_indices(user_groups_lt[int(cid)])
        # Hard guard: CIFAR10 requires scalar integer indices
        if any(not isinstance(i, int) for i in train_idxs[:10]):
            raise TypeError(f"train_idxs contains non-int types (sample): {[type(i) for i in train_idxs[:10]]}")
        dl_train = DataLoader(Subset(train_dataset, train_idxs), batch_size=int(args.batch_size), shuffle=True, num_workers=0, drop_last=True)
        dl_test = DataLoader(Subset(test_dataset, test_idxs), batch_size=int(args.batch_size), shuffle=False, num_workers=0, drop_last=False)

        # Sanity: infer low dim from a real batch
        m.eval()
        images0, labels0 = next(iter(dl_train))
        images0 = images0.to(device)
        with torch.no_grad():
            _logits0, _lp0, _h0, low_raw0, _proj0 = m(images0)
        if int(low_raw0.shape[1]) != int(low_feature_dim):
            raise RuntimeError(f"low_feature_dim mismatch: stage2={low_feature_dim} vs model={int(low_raw0.shape[1])}")

        # Optimizer for allowed params only
        opt = torch.optim.Adam(
            [p for p in list(m.fc0.parameters()) + list(m.fc1.parameters()) + list(m.fc2.parameters()) if p.requires_grad],
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
        )
        _assert_optimizer_params(opt, allowed=list(m.fc0.parameters()) + list(m.fc1.parameters()) + list(m.fc2.parameters()))

        # Baseline eval
        acc0, loss0 = _eval_client_on_images(m, dl_test, device=device, num_classes=num_classes)

        # Train steps (iterate over dl_train as needed)
        m.train()
        step = 0
        t0 = time.time()
        it = iter(dl_train)
        last_loss = None
        best_acc = float(acc0)
        best_loss = float(loss0)
        best_step = 0
        best_state = None

        while step < int(args.steps):
            try:
                images, labels = next(it)
            except StopIteration:
                it = iter(dl_train)
                images, labels = next(it)
            images = images.to(device)
            labels = labels.to(device).long()

            # Extract real low features (raw, unnormalized)
            with torch.no_grad():
                _logits_r, _lp_r, _h_r, low_real, _proj_r = m(images)

            # Synthetic batch size
            b_real = int(low_real.shape[0])
            b_syn = int(round(float(b_real) * syn_ratio))
            b_ooc = 0
            y_syn_id = None
            x_syn_id = None
            y_ooc = None
            x_ooc = None

            if b_syn > 0:
                # Determine OOC sample count within synthetic batch
                missing = client_missing.get(int(cid), []) or []
                if float(args.ooc_ratio) > 0.0 and len(missing) > 0:
                    b_ooc = int(round(float(b_syn) * float(args.ooc_ratio)))
                    b_ooc = max(0, min(b_ooc, b_syn))
                b_id = int(b_syn - b_ooc)

                # In-distribution synthetic labels sampled from real batch labels (non-iid friendly)
                if b_id > 0:
                    idx = torch.randint(low=0, high=b_real, size=(b_id,), device=device)
                    y_syn_id = labels[idx]
                    mu_b, cov_b, _rf_b = gather_by_label(stats_low, y_syn_id)
                    with torch.no_grad():
                        x_syn_id = gen(y_syn_id, mu=mu_b, cov_diag=cov_b)["output"]

                # OOC synthetic labels sampled uniformly from client-missing classes (if any)
                if b_ooc > 0:
                    # Sample missing class ids on CPU then move to device (small overhead, deterministic w/ seed)
                    miss = torch.tensor(missing, dtype=torch.long)
                    ridx = torch.randint(low=0, high=int(miss.numel()), size=(b_ooc,))
                    y_ooc = miss[ridx].to(device)
                    mu_o, cov_o, _rf_o = gather_by_label(stats_low, y_ooc)
                    with torch.no_grad():
                        x_ooc = gen(y_ooc, mu=mu_o, cov_diag=cov_o)["output"]

            # CE loss is computed on: real + in-distribution synthetic only (NEVER on OOC labels)
            if x_syn_id is not None:
                x_low_ce = torch.cat([low_real, x_syn_id], dim=0)
                y_all = torch.cat([labels, y_syn_id], dim=0)
            else:
                x_low_ce = low_real
                y_all = labels

            # IMPORTANT: x_low must be raw low features; no extra normalize here.
            logits, _lp, _h_raw, _proj = m.forward_from_low(x_low_ce)
            logits = logits[:, 0:num_classes]
            loss_ce = F.cross_entropy(logits, y_all, reduction="mean")

            # Optional OOC regularization (fc0-only)
            loss_ooc = None
            if x_ooc is not None and y_ooc is not None and float(args.ooc_lambda) > 0.0:
                mode = str(args.ooc_mode).lower()
                if mode == "high_mean":
                    if mu_high is None:
                        raise RuntimeError("OOC high_mean requested but mu_high is None (did Stage-2 include global high stats?)")
                    loss_ooc = _ooc_high_mean_loss_fc0_only(m, x_ooc, y_ooc, mu_high=mu_high)
                elif mode == "uniform":
                    loss_ooc = _ooc_uniform_loss_fc0_only(m, x_ooc, num_classes=num_classes)
                else:
                    raise ValueError(f"Unsupported --ooc_mode: {args.ooc_mode}")

            if loss_ooc is not None:
                loss = loss_ce + float(args.ooc_lambda) * loss_ooc
            else:
                loss = loss_ce

            opt.zero_grad(set_to_none=True)
            loss.backward()
            # One-time grad sanity check early to avoid silent training drift
            if step == 0:
                _assert_grad_flow(m)
            opt.step()
            last_loss = float(loss.item())

            if (step + 1) % int(args.log_interval) == 0:
                dt = time.time() - t0
                logger.info(f"[stage4][cid={cid:02d}] step {step+1}/{int(args.steps)} loss={last_loss:.6f} sec={dt:.1f}")

            if (step + 1) % int(args.eval_interval) == 0 or (step + 1) == int(args.steps):
                acc1, loss1 = _eval_client_on_images(m, dl_test, device=device, num_classes=num_classes)
                logger.info(f"[stage4][cid={cid:02d}] eval@{step+1}: acc={acc1:.4f} loss={loss1:.4f}")
                if int(args.save_best) == 1 and float(acc1) >= float(best_acc):
                    # Tie-break: prefer later step when acc ties (often improves calibration); adjust if you prefer earlier.
                    best_acc = float(acc1)
                    best_loss = float(loss1)
                    best_step = int(step + 1)
                    best_state = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
                m.train()

            step += 1

        # Final eval + save
        accF, lossF = _eval_client_on_images(m, dl_test, device=device, num_classes=num_classes)
        # If we never evaluated (e.g., steps<eval_interval and eval not hit), optionally track final as best
        if int(args.save_best) == 1 and best_state is None:
            best_state = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
            best_acc = float(accF)
            best_loss = float(lossF)
            best_step = int(args.steps)

        use_best = (int(args.select_best_for_after) == 1) and (best_state is not None)
        acc_after = float(best_acc) if use_best else float(accF)
        loss_after = float(best_loss) if use_best else float(lossF)
        out_path = os.path.join(out_dir, f"client_{cid:02d}.pt")
        torch.save(
            {
                "meta": {
                    "stage": 4,
                    "client_id": cid,
                    "dataset": "cifar10",
                    "num_classes": num_classes,
                    "base_stage1_ckpt": args.stage1_ckpt_path,
                    "stage2_stats_path": args.stage2_stats_path,
                    "gen_path": gen_path,
                    "split_path": split_path,
                    "syn_ratio": syn_ratio,
                    "ooc_ratio": float(args.ooc_ratio),
                    "ooc_lambda": float(args.ooc_lambda),
                    "ooc_mode": str(args.ooc_mode),
                    "steps": int(args.steps),
                    "batch_size": int(args.batch_size),
                    "lr": float(args.lr),
                    "seed": int(args.seed),
                    "acc_before": float(acc0),
                    "acc_after": float(acc_after),
                    "acc_final": float(accF),
                    "acc_best": float(best_acc),
                    "best_step": int(best_step),
                },
                "state": {
                    "model_state_dict": best_state if use_best else m.state_dict(),
                    "model_state_dict_final": m.state_dict(),
                    "model_state_dict_best": best_state,
                    "components_state_dicts": m.get_component_state_dicts() if hasattr(m, "get_component_state_dicts") else None,
                },
            },
            out_path,
        )

        results.append(
            {
                "client_id": cid,
                "acc_before": float(acc0),
                "acc_after": float(acc_after),
                "acc_final": float(accF),
                "acc_best": float(best_acc),
                "best_step": int(best_step),
                "loss_before": float(loss0),
                "loss_after": float(loss_after),
                "loss_final": float(lossF),
                "loss_best": float(best_loss),
                "last_train_loss": float(last_loss) if last_loss is not None else None,
            }
        )

    # Save summary
    acc_b = [r["acc_before"] for r in results]
    acc_a = [r["acc_after"] for r in results]
    acc_best = [r.get("acc_best", r["acc_after"]) for r in results]
    mean_before = float(np.mean(acc_b)) if len(acc_b) > 0 else 0.0
    mean_after = float(np.mean(acc_a)) if len(acc_a) > 0 else 0.0
    mean_best = float(np.mean(acc_best)) if len(acc_best) > 0 else 0.0

    results_path = os.path.join(out_dir, "stage4_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "results": results,
                "sec_total": time.time() - start_all,
                "out_dir": out_dir,
                "run_tag": run_tag,
                "run_name": args.run_name,
                "auto_run_dir": int(args.auto_run_dir),
                "mean_acc_before": mean_before,
                "mean_acc_after": mean_after,
                "mean_acc_best": mean_best,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # Update meta with summary metrics (append-only for backwards compatibility)
    meta_out["summary"] = {
        "sec_total": float(time.time() - start_all),
        "mean_acc_before": mean_before,
        "mean_acc_after": mean_after,
        "mean_acc_best": mean_best,
        "select_best_for_after": int(args.select_best_for_after),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_out, f, ensure_ascii=False, indent=2)

    logger.info(f"[stage4] done. out_dir={out_dir}")
    logger.info(f"[stage4] mean acc before={mean_before:.4f} after={mean_after:.4f}")
    logger.info(f"[stage4] mean acc best={mean_best:.4f} (select_best_for_after={int(args.select_best_for_after)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



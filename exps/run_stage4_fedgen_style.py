#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage-4 (FedGen-style) for FedMPS (CIFAR10 + CNNCifar):
Client-side finetune of (fc0 + fc1 + fc2) using synthetic low-level features.

This is a *new* standalone variant of Stage-4, inspired by FedGen's idea:
  - Train on real supervised CE loss
  - Train on synthetic supervised CE loss ("teacher loss" in FedGen spirit)
  - Add a consistency / distillation loss that aligns real predictions to
    a per-label teacher distribution estimated from synthetic samples.

Notes:
  - Synthetic features here are still FedMPS-style: low_level_features_raw (flatten conv output),
    injected via CNNCifar.forward_from_low(x_low).
  - This script is intentionally separate from exps/run_stage4.py to keep the original
    behavior reproducible.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from datetime import datetime
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

# Ensure project root is on sys.path when running as a script:
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
    s = f"{float(x):.6g}"
    s = s.replace("+", "")
    return s.replace(".", "p")


def _make_run_tag(args: argparse.Namespace) -> str:
    return (
        f"fedgen_syn{_format_float_for_tag(args.syn_ratio)}"
        f"_a{_format_float_for_tag(args.syn_alpha)}"
        f"_b{_format_float_for_tag(args.syn_beta)}"
        f"_T{_format_float_for_tag(args.distill_T)}"
        f"_t2{int(args.distill_scale_T2)}"
        f"_w{int(args.syn_warmup_steps)}"
        f"_r{int(args.syn_scale_by_ratio)}"
        f"_{str(args.syn_label_sampling)}"
        f"_steps{int(args.steps)}"
        f"_bs{int(args.batch_size)}"
        f"_lr{_format_float_for_tag(args.lr)}"
        f"_wd{_format_float_for_tag(args.weight_decay)}"
        f"_seed{int(args.seed)}"
    )


def _setup_logger(out_dir: str) -> logging.Logger:
    os.makedirs(out_dir, exist_ok=True)
    logger = logging.getLogger("stage4_fedgen_style")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        for h in list(logger.handlers):
            try:
                h.flush()
                h.close()
            except Exception:
                pass
            logger.removeHandler(h)

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(os.path.join(out_dir, "stage4_fedgen_style.log"), encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def _as_int_indices(idxs) -> List[int]:
    def _flatten(v) -> List[int]:
        if v is None:
            return []

        if torch.is_tensor(v):
            v = v.detach().cpu()
            if v.numel() == 0:
                return []
            if v.numel() == 1:
                return [int(v.item())]
            return [int(x) for x in v.flatten().tolist()]

        if isinstance(v, np.ndarray):
            if v.size == 0:
                return []
            if v.ndim == 0:
                return _flatten(v.item())
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

        if isinstance(v, (list, tuple, set)):
            out: List[int] = []
            for e in v:
                out.extend(_flatten(e))
            return out

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
    if any(not isinstance(x, int) for x in out):
        bad = [type(x) for x in out if not isinstance(x, int)][:5]
        raise TypeError(f"Indices normalization failed; non-int types remain: {bad}")
    return out


@torch.no_grad()
def _eval_client_on_images(model: nn.Module, dl: DataLoader, device: str, num_classes: int) -> Tuple[float, float]:
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
    if path_like is None:
        return path_like
    p = os.path.expanduser(str(path_like))
    if not os.path.isabs(p):
        p = os.path.abspath(p)
    return p


def _resolve_generator_path(gen_path_cli: str, *, logdir: str) -> str:
    gen_path_cli = _resolve_existing_path(gen_path_cli)

    if gen_path_cli and os.path.isdir(gen_path_cli):
        cand = os.path.join(gen_path_cli, "generator.pt")
        if os.path.exists(cand):
            return cand
        raise FileNotFoundError(f"gen_path is a directory but generator.pt not found: {cand}")

    if gen_path_cli and os.path.exists(gen_path_cli):
        return gen_path_cli

    cands: List[str] = [
        os.path.join(logdir, "stage3", "gen", "generator.pt"),
        os.path.join(logdir, "stage3", "lowgen_minloop", "generator.pt"),
        os.path.join(logdir, "stage3_gen", "generator.pt"),
        os.path.join(logdir, "stage3_lowgen_minloop", "generator.pt"),
    ]

    def _scan_suffix_dirs(base_parent: str, prefix: str) -> List[str]:
        try:
            if not os.path.isdir(base_parent):
                return []
            out = []
            for name in os.listdir(base_parent):
                if not name.startswith(prefix):
                    continue
                d = os.path.join(base_parent, name)
                if not os.path.isdir(d):
                    continue
                gp = os.path.join(d, "generator.pt")
                if os.path.exists(gp):
                    out.append(gp)
            return out
        except Exception:
            return []

    cands.extend(_scan_suffix_dirs(os.path.join(logdir, "stage3"), "lowgen_minloop_"))
    cands.extend(_scan_suffix_dirs(logdir, "stage3_lowgen_minloop_"))

    existing = [p for p in cands if os.path.exists(p)]
    if len(existing) == 1:
        print(f"[stage4_fedgen_style] Warning: --gen_path not found. Fallback to: {existing[0]}")
        return existing[0]
    if len(existing) > 1:
        raise FileNotFoundError(
            "Multiple generator.pt candidates found under logdir; please pass --gen_path explicitly:\n" + "\n".join(existing)
        )

    raise FileNotFoundError(
        "generator.pt not found. Checked:\n" f"  - {gen_path_cli}\n" + "\n".join([f"  - {p}" for p in cands])
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
    if not (hasattr(model, "forward_from_low") and callable(getattr(model, "forward_from_low"))):
        raise RuntimeError("Model missing required API: forward_from_low(x_low).")
    for p in list(model.conv1.parameters()) + list(model.conv2.parameters()):
        if p.requires_grad:
            raise RuntimeError("Freeze check failed: conv1/conv2 should have requires_grad=False in Stage-4.")
    for p in model.projector.parameters():
        if p.requires_grad:
            raise RuntimeError("Freeze check failed: projector should have requires_grad=False in Stage-4.")
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


def main() -> int:
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("--stage1_ckpt_path", type=str, required=True, help="Stage-1 checkpoint path (best-wo.pt / latest.pt)")
    p.add_argument("--stage2_stats_path", type=str, required=True, help="Stage-2 global_stats.pt path")
    p.add_argument("--gen_path", type=str, required=True, help="Stage-3 generator.pt path")
    p.add_argument("--split_path", type=str, default=None, help="split.pkl path (if None, infer from ckpt meta/args)")
    p.add_argument("--out_dir", type=str, default=None, help="Output dir (default: <ckpt.meta.logdir>/stage4/finetune)")
    p.add_argument("--auto_run_dir", type=int, default=1)
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--run_tag", type=str, default=None)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=None)

    # Training hyperparams
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)

    # Synthetic controls (FedGen-style)
    p.add_argument("--syn_ratio", type=float, default=0.2, help="Synthetic batch size ratio relative to real batch (0..1)")
    p.add_argument("--syn_batch_size", type=int, default=-1, help="If >0, override synthetic batch size per step.")
    p.add_argument("--syn_alpha", type=float, default=1.0, help="Weight for synthetic CE loss.")
    p.add_argument("--syn_beta", type=float, default=1.0, help="Weight for consistency/distillation loss.")
    p.add_argument("--distill_T", type=float, default=1.0, help="Distillation temperature.")
    p.add_argument(
        "--distill_scale_T2",
        type=int,
        default=1,
        choices=[0, 1],
        help="If 1, scale KL term by T^2 (standard distillation). If 0, do not scale by T^2 (often easier to tune).",
    )
    p.add_argument(
        "--syn_scale_by_ratio",
        type=int,
        default=1,
        choices=[0, 1],
        help="If 1, scale synthetic CE and KL by (b_syn/b_real) so synthetic signal doesn't dominate when b_syn is small/large.",
    )
    p.add_argument(
        "--syn_warmup_steps",
        type=int,
        default=0,
        help="Warmup: for the first N steps of each client, disable synthetic losses (alpha=beta=0).",
    )
    p.add_argument(
        "--syn_label_sampling",
        type=str,
        default="global_qualified",
        choices=["global_qualified", "batch_labels", "client_classes"],
        help="How to sample synthetic labels y_syn. "
        "'global_qualified': uniform over global qualified classes (sample_per_class>0). "
        "'batch_labels': sample from current real batch labels. "
        "'client_classes': sample from classes present in this client (from split/classes_list or inferred).",
    )

    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--eval_interval", type=int, default=200)
    p.add_argument("--save_best", type=int, default=1)
    p.add_argument("--select_best_for_after", type=int, default=1)

    # Generator architecture (used if generator_meta.json is missing)
    p.add_argument("--gen_noise_dim", type=int, default=64)
    p.add_argument("--gen_y_emb_dim", type=int, default=32)
    p.add_argument("--gen_stat_emb_dim", type=int, default=128)
    p.add_argument("--gen_hidden_dim", type=int, default=256)
    p.add_argument("--gen_n_hidden_layers", type=int, default=2)
    p.add_argument("--gen_relu_output", type=int, default=1)
    p.add_argument("--gen_use_cov_diag", type=int, default=1)

    args = p.parse_args()

    if args.seed is None:
        args.seed = int(time.time_ns() % (2**31 - 1))

    device = _resolve_device(args.gpu)
    _seed_all(int(args.seed))

    ckpt = load_checkpoint(args.stage1_ckpt_path, map_location="cpu")
    ckpt_meta = (ckpt.get("meta", {}) or {}) if isinstance(ckpt, dict) else {}
    ckpt_state = (ckpt.get("state", {}) or {}) if isinstance(ckpt, dict) else {}
    ckpt_args = (ckpt.get("args", {}) or {}) if isinstance(ckpt, dict) else {}

    dataset = str(ckpt_meta.get("dataset", ckpt_args.get("dataset", "cifar10")))
    if dataset != "cifar10":
        raise ValueError(f"Stage-4 currently supports cifar10 only. Got dataset={dataset}")
    num_users = int(ckpt_meta.get("num_users", ckpt_args.get("num_users", 20)))
    num_classes = int(ckpt_meta.get("num_classes", ckpt_args.get("num_classes", 10)))

    split_path = args.split_path or ckpt_meta.get("split_path", None) or ckpt_args.get("split_path", None)
    if not split_path or not os.path.exists(split_path):
        raise FileNotFoundError(f"split_path not found: {split_path}")

    logdir = ckpt_meta.get("logdir", None) or ckpt_args.get("log_dir", None) or os.path.dirname(os.path.abspath(args.stage1_ckpt_path))
    base_out_dir = args.out_dir or os.path.join(logdir, "stage4", "finetune_fedgen_style")
    run_tag = str(args.run_tag).strip() if args.run_tag is not None else _make_run_tag(args)
    if args.run_name:
        run_tag = f"{run_tag}_{str(args.run_name).strip()}"

    if int(args.auto_run_dir) == 1:
        out_dir = os.path.join(base_out_dir, run_tag)
        if os.path.exists(out_dir) and os.listdir(out_dir):
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            out_dir = f"{out_dir}_{ts}"
    else:
        out_dir = base_out_dir

    os.makedirs(out_dir, exist_ok=True)
    logger = _setup_logger(out_dir)
    logger.info(f"[stage4_fedgen_style] out_dir={out_dir}")
    logger.info(f"[stage4_fedgen_style] run_tag={run_tag} auto_run_dir={int(args.auto_run_dir)} run_name={args.run_name}")
    logger.info(f"[stage4_fedgen_style] resolved_seed={args.seed}")
    logger.info(f"[stage4_fedgen_style] cli_args={vars(args)}")

    gen_path = _resolve_generator_path(args.gen_path, logdir=logdir)
    logger.info(
        f"[stage4_fedgen_style] resolved_paths: stage1={args.stage1_ckpt_path} stage2={args.stage2_stats_path} gen={gen_path} split={split_path}"
    )

    split = load_split(split_path)
    n_list = split["n_list"]
    k_list = split["k_list"]
    user_groups: Dict[int, List[int]] = split["user_groups"]
    user_groups_lt: Dict[int, List[int]] = split["user_groups_lt"]

    base = dict(ckpt_args)
    base["dataset"] = "cifar10"
    base["num_classes"] = num_classes
    base["num_users"] = num_users
    base["gpu"] = int(args.gpu)
    base["device"] = device
    args_ds = SimpleNamespace(**base)
    train_dataset, test_dataset, _, _, _, _ = get_dataset(args_ds, n_list, k_list)

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

    qualified = torch.nonzero(stats_low.sample_per_class > 0, as_tuple=False).view(-1).to(device)
    if qualified.numel() == 0:
        raise ValueError("No qualified classes with sample_per_class > 0.")
    qualified_set = set(qualified.detach().cpu().tolist())

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

    local_sd = ckpt_state.get("local_models_full_state_dicts", None)
    if not isinstance(local_sd, dict):
        raise ValueError("Stage-1 checkpoint missing state['local_models_full_state_dicts']")

    meta_path = os.path.join(out_dir, "stage4_fedgen_style_meta.json")
    meta_out = {
        "stage": 4,
        "variant": "fedgen_style",
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
            "syn_batch_size": int(args.syn_batch_size),
            "syn_alpha": float(args.syn_alpha),
            "syn_beta": float(args.syn_beta),
            "distill_T": float(args.distill_T),
            "distill_scale_T2": int(args.distill_scale_T2),
            "syn_scale_by_ratio": int(args.syn_scale_by_ratio),
            "syn_warmup_steps": int(args.syn_warmup_steps),
            "seed": int(args.seed),
            "syn_label_sampling": str(args.syn_label_sampling),
        },
        "gen_hparams": gen_h,
    }
    # Pre-compute per-client present classes (for syn_label_sampling='client_classes')
    client_present: Dict[int, List[int]] = {}
    classes_list = split.get("classes_list", None)
    if classes_list is not None:
        for cid in range(int(num_users)):
            if isinstance(classes_list, dict):
                present = set(int(x) for x in classes_list.get(int(cid), []))
            else:
                present = set(int(x) for x in classes_list[int(cid)])
            # Only keep globally qualified classes to avoid meaningless conditioning
            present = sorted(list(present & qualified_set))
            client_present[int(cid)] = present
    else:
        # Fallback: infer from train_dataset.targets and split indices
        targets = getattr(train_dataset, "targets", None)
        if targets is None:
            raise AttributeError(
                "train_dataset has no 'targets'; cannot infer client classes without split['classes_list']."
            )
        for cid in range(int(num_users)):
            idxs = _as_int_indices(user_groups[int(cid)])
            present = set(int(targets[i]) for i in idxs)
            present = sorted(list(present & qualified_set))
            client_present[int(cid)] = present

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_out, f, ensure_ascii=False, indent=2)

    results: List[Dict[str, object]] = []
    start_all = time.time()

    for cid in range(num_users):
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

        train_idxs = _as_int_indices(user_groups[int(cid)])
        test_idxs = _as_int_indices(user_groups_lt[int(cid)])
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

        opt = torch.optim.Adam(
            [p for p in list(m.fc0.parameters()) + list(m.fc1.parameters()) + list(m.fc2.parameters()) if p.requires_grad],
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
        )
        _assert_optimizer_params(opt, allowed=list(m.fc0.parameters()) + list(m.fc1.parameters()) + list(m.fc2.parameters()))

        acc0, loss0 = _eval_client_on_images(m, dl_test, device=device, num_classes=num_classes)

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

            # Extract real low features (raw, unnormalized) without grad.
            with torch.no_grad():
                _logits_r, _lp_r, _h_r, low_real, _proj_r = m(images)

            b_real = int(low_real.shape[0])
            b_syn = int(round(float(b_real) * float(args.syn_ratio)))
            if int(args.syn_batch_size) > 0:
                b_syn = int(args.syn_batch_size)

            # -------------------------
            # FedGen-style synthetic training core:
            #   - real_CE: supervised loss on real samples (via forward_from_low)
            #   - syn_teacher_CE: supervised loss on synthetic samples (labels from global qualified classes)
            #   - consistency_KL: align real predictions to per-label teacher distribution estimated from synthetic batch
            # -------------------------
            # Real forward (trainable head only)
            logits_real, _lp_real, _h_real, _proj_real = m.forward_from_low(low_real)
            logits_real = logits_real[:, 0:num_classes]
            real_ce = F.cross_entropy(logits_real, labels, reduction="mean")

            # Synthetic sampling: y_syn ~ Uniform(qualified_global)
            syn_ce = torch.tensor(0.0, device=device)
            cons_kl = torch.tensor(0.0, device=device)
            cover = torch.tensor(0.0, device=device)
            if b_syn > 0:
                sampling = str(args.syn_label_sampling).lower()
                if sampling == "global_qualified":
                    idx = torch.randint(low=0, high=int(qualified.numel()), size=(b_syn,), device=device)
                    y_syn = qualified[idx].long()
                elif sampling == "batch_labels":
                    idx = torch.randint(low=0, high=b_real, size=(b_syn,), device=device)
                    y_syn = labels[idx].long()
                elif sampling == "client_classes":
                    present = client_present.get(int(cid), []) or []
                    if len(present) == 0:
                        # Fallback to batch_labels to avoid hard failure
                        idx = torch.randint(low=0, high=b_real, size=(b_syn,), device=device)
                        y_syn = labels[idx].long()
                    else:
                        present_t = torch.tensor(present, dtype=torch.long, device=device)
                        ridx = torch.randint(low=0, high=int(present_t.numel()), size=(b_syn,), device=device)
                        y_syn = present_t[ridx].long()
                else:
                    raise ValueError(f"Unsupported --syn_label_sampling: {args.syn_label_sampling}")

                mu_b, cov_b, _rf_b = gather_by_label(stats_low, y_syn)
                with torch.no_grad():
                    x_syn = gen(y_syn, mu=mu_b, cov_diag=cov_b)["output"]

                # Synthetic forward (trainable head only)
                logits_syn, _lp_syn, _h_syn, _proj_syn = m.forward_from_low(x_syn)
                logits_syn = logits_syn[:, 0:num_classes]
                syn_ce = F.cross_entropy(logits_syn, y_syn, reduction="mean")

                # Consistency/distillation:
                # teacher_prob[c] = mean softmax(logits_syn/T) over syn samples of class c
                T = float(args.distill_T)
                if T <= 0:
                    raise ValueError("--distill_T must be > 0")
                syn_prob = F.softmax(logits_syn / T, dim=1)  # (B_syn, C)

                teacher_sum = torch.zeros((num_classes, num_classes), device=device, dtype=syn_prob.dtype)
                teacher_cnt = torch.zeros((num_classes,), device=device, dtype=syn_prob.dtype)
                teacher_sum.index_add_(0, y_syn, syn_prob)
                teacher_cnt.index_add_(0, y_syn, torch.ones_like(y_syn, dtype=syn_prob.dtype))
                teacher_avg = teacher_sum / (teacher_cnt.clamp_min(1.0).unsqueeze(1))
                # FedGen-style: detach teacher distribution so KL only updates the real branch.
                teacher_avg = teacher_avg.detach()

                mask = (teacher_cnt[labels] > 0)
                n_mask = int(mask.sum().item())
                if n_mask > 0:
                    logp_real_T = F.log_softmax(logits_real / T, dim=1)
                    teacher_for_real = teacher_avg[labels]  # (B_real, C)
                    logp_sel = logp_real_T[mask]
                    teacher_sel = teacher_for_real[mask]
                    cons_kl = F.kl_div(logp_sel, teacher_sel, reduction="batchmean")
                    if int(args.distill_scale_T2) == 1:
                        cons_kl = cons_kl * (T * T)
                    cover = torch.tensor(float(n_mask) / float(b_real), device=device)

            # Optional scaling so synthetic doesn't dominate
            ratio = (float(b_syn) / float(b_real)) if (b_real > 0) else 0.0
            if int(args.syn_scale_by_ratio) == 1:
                syn_ce_eff = syn_ce * ratio
                cons_kl_eff = cons_kl * ratio
            else:
                syn_ce_eff = syn_ce
                cons_kl_eff = cons_kl

            # Warmup: disable synthetic loss early per client
            if int(args.syn_warmup_steps) > 0 and int(step) < int(args.syn_warmup_steps):
                alpha = 0.0
                beta = 0.0
            else:
                alpha = float(args.syn_alpha)
                beta = float(args.syn_beta)

            loss = real_ce + alpha * syn_ce_eff + beta * cons_kl_eff

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if step == 0:
                _assert_grad_flow(m)
            opt.step()
            last_loss = float(loss.item())

            if (step + 1) % int(args.log_interval) == 0:
                dt = time.time() - t0
                logger.info(
                    f"[stage4_fedgen_style][cid={cid:02d}] step {step+1}/{int(args.steps)} "
                    f"loss={last_loss:.6f} real_ce={float(real_ce.item()):.6f} syn_ce={float(syn_ce.item()):.6f} "
                    f"kl={float(cons_kl.item()):.6f} cover={float(cover.item()):.3f} "
                    f"ratio={ratio:.3f} alpha={alpha:.4g} beta={beta:.4g} sec={dt:.1f}"
                )

            if (step + 1) % int(args.eval_interval) == 0 or (step + 1) == int(args.steps):
                acc1, loss1 = _eval_client_on_images(m, dl_test, device=device, num_classes=num_classes)
                logger.info(f"[stage4_fedgen_style][cid={cid:02d}] eval@{step+1}: acc={acc1:.4f} loss={loss1:.4f}")
                if int(args.save_best) == 1 and float(acc1) >= float(best_acc):
                    best_acc = float(acc1)
                    best_loss = float(loss1)
                    best_step = int(step + 1)
                    best_state = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
                m.train()

            step += 1

        accF, lossF = _eval_client_on_images(m, dl_test, device=device, num_classes=num_classes)
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
                    "variant": "fedgen_style",
                    "client_id": cid,
                    "dataset": "cifar10",
                    "num_classes": num_classes,
                    "base_stage1_ckpt": args.stage1_ckpt_path,
                    "stage2_stats_path": args.stage2_stats_path,
                    "gen_path": gen_path,
                    "split_path": split_path,
                    "steps": int(args.steps),
                    "batch_size": int(args.batch_size),
                    "lr": float(args.lr),
                    "seed": int(args.seed),
                    "syn_ratio": float(args.syn_ratio),
                    "syn_batch_size": int(args.syn_batch_size),
                    "syn_alpha": float(args.syn_alpha),
                    "syn_beta": float(args.syn_beta),
                    "distill_T": float(args.distill_T),
                    "distill_scale_T2": int(args.distill_scale_T2),
                    "syn_scale_by_ratio": int(args.syn_scale_by_ratio),
                    "syn_warmup_steps": int(args.syn_warmup_steps),
                    "syn_label_sampling": str(args.syn_label_sampling),
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

    acc_b = [r["acc_before"] for r in results]
    acc_a = [r["acc_after"] for r in results]
    acc_best_arr = [r.get("acc_best", r["acc_after"]) for r in results]
    mean_before = float(np.mean(acc_b)) if len(acc_b) > 0 else 0.0
    mean_after = float(np.mean(acc_a)) if len(acc_a) > 0 else 0.0
    mean_best = float(np.mean(acc_best_arr)) if len(acc_best_arr) > 0 else 0.0

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

    meta_out["summary"] = {
        "sec_total": float(time.time() - start_all),
        "mean_acc_before": mean_before,
        "mean_acc_after": mean_after,
        "mean_acc_best": mean_best,
        "select_best_for_after": int(args.select_best_for_after),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_out, f, ensure_ascii=False, indent=2)

    logger.info(f"[stage4_fedgen_style] done. out_dir={out_dir}")
    logger.info(f"[stage4_fedgen_style] mean acc before={mean_before:.4f} after={mean_after:.4f}")
    logger.info(f"[stage4_fedgen_style] mean acc best={mean_best:.4f} (select_best_for_after={int(args.select_best_for_after)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



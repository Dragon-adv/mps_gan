#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage-4 (cVAE shared feature pool) for FedMPS (CIFAR10 + CNNCifar):

Server-side:
  - Load Stage-3 cVAE (ConditionalFeatureVAE) and build a per-class shared pool
    of synthetic low-level features (low_level_features_raw).

Client-side:
  - Freeze low encoder (conv1/conv2) and projector; finetune only (fc0, fc1, fc2).
  - Train with real supervised CE + synthetic supervised CE (CE-only variant):
      loss = real_ce + alpha * syn_ce

This script is intentionally separate from exps/run_stage4_fedgen_style.py.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import math
import sys
import time
from datetime import datetime
from collections import Counter, defaultdict
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

# Ensure project root is on sys.path when running as a script:
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from lib.checkpoint import load_checkpoint
from lib.cvae_feature_gen import CVAEConfig, ConditionalFeatureVAE, generate_features
from lib.models.models import CNNCifar
from lib.split_manager import load_split
from lib.utils import get_dataset


def _maybe_win_long_path(path: str) -> str:
    """
    Windows 经典 MAX_PATH 限制下，超长路径有时会以 FileNotFoundError 的形式报错。

    这里在路径较长时自动转换为绝对路径并添加 long-path 前缀（\\\\?\\），
    以确保 open/torch.save 等 IO 在 Windows 上稳定工作。

    - 非 Windows：原样返回
    - Windows：长度较短时返回绝对路径；超长时返回带前缀的绝对路径
    """
    if os.name != "nt":
        return path

    p = os.path.abspath(path)
    # 经验阈值：接近 260 时就可能触发问题（包含后续拼接/内部展开）
    if len(p) < 240:
        return p

    if p.startswith("\\\\?\\"):
        return p
    # UNC: \\server\share\... -> \\?\UNC\server\share\...
    if p.startswith("\\\\"):
        return "\\\\?\\UNC\\" + p.lstrip("\\")
    return "\\\\?\\" + p


def _resolve_device(gpu: int) -> str:
    if torch.cuda.is_available() and gpu is not None and int(gpu) >= 0:
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
    # Use timestamp for directory name uniqueness instead of long parameter string
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"cvae_pool_{timestamp}"


def _setup_logger(out_dir: str) -> logging.Logger:
    os.makedirs(out_dir, exist_ok=True)
    logger = logging.getLogger("stage4_cvae_pool")
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
    fh = logging.FileHandler(_maybe_win_long_path(os.path.join(out_dir, "stage4_cvae_pool.log")), encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def _infer_logdir_from_ckpt(stage1_ckpt_path: str, ckpt_meta: dict) -> str:
    meta_logdir = (ckpt_meta or {}).get("logdir", None)
    if meta_logdir:
        return meta_logdir
    # typical (legacy): <logdir>/stage1_ckpts/best-wo.pt
    # typical (new):    <logdir>/stage1/ckpts/best-wo.pt
    ckpt_abs = os.path.abspath(stage1_ckpt_path)
    ckpt_dir = os.path.dirname(ckpt_abs)
    if os.path.basename(ckpt_dir) == "stage1_ckpts":
        return os.path.dirname(ckpt_dir)
    if os.path.basename(ckpt_dir) == "ckpts" and os.path.basename(os.path.dirname(ckpt_dir)) == "stage1":
        return os.path.dirname(os.path.dirname(ckpt_dir))
    return ckpt_dir


def _resolve_existing_path(path_like: str) -> str:
    if path_like is None:
        return path_like
    p = os.path.expanduser(str(path_like))
    if not os.path.isabs(p):
        p = os.path.abspath(p)
    return p


def _resolve_cvae_path(cvae_path_cli: str) -> str:
    cvae_path_cli = _resolve_existing_path(cvae_path_cli)
    if cvae_path_cli and os.path.isdir(cvae_path_cli):
        cand = os.path.join(cvae_path_cli, "generator.pt")
        if os.path.exists(cand):
            return cand
        raise FileNotFoundError(f"--cvae_path is a directory but generator.pt not found: {cand}")
    if cvae_path_cli and os.path.exists(cvae_path_cli):
        return cvae_path_cli
    raise FileNotFoundError(f"cVAE generator.pt not found: {cvae_path_cli}")


def _load_cvae_meta(gen_path: str) -> dict:
    d = os.path.dirname(os.path.abspath(gen_path))
    meta_path = os.path.join(d, "generator_meta.json")
    if not os.path.exists(_maybe_win_long_path(meta_path)):
        raise FileNotFoundError(f"generator_meta.json not found next to cVAE generator: {meta_path}")
    with open(_maybe_win_long_path(meta_path), "r", encoding="utf-8") as f:
        return json.load(f)


def _load_cvae(gen_path: str, *, device: str, y_emb_dim: int = 32) -> Tuple[ConditionalFeatureVAE, dict]:
    meta = _load_cvae_meta(gen_path)
    if str(meta.get("variant", "")).lower() != "cvae":
        # still allow if missing, but warn via meta later
        pass
    c = meta.get("cvae_cfg", {}) or {}
    feature_dim = int(meta.get("feature_dim"))
    num_classes = int(meta.get("num_classes"))
    cfg = CVAEConfig(
        feature_dim=feature_dim,
        num_classes=num_classes,
        latent_dim=int(c.get("latent_dim", 64)),
        hidden_dim=int(c.get("hidden_dim", 256)),
        n_hidden=int(c.get("n_hidden", 2)),
        y_emb_dim=int(c.get("y_emb_dim", y_emb_dim)),
        recon_loss=str(c.get("recon_loss", "l2")),
        beta=float(c.get("beta", 1.0)),
        kl_anneal_steps=int(c.get("kl_anneal_steps", 0)),
    )
    model = ConditionalFeatureVAE(cfg).to(device)
    sd = torch.load(_maybe_win_long_path(gen_path), map_location="cpu")
    model.load_state_dict(sd, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, meta


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


def _compute_global_class_counts(train_dataset, user_groups: Dict[int, List[int]], num_classes: int) -> torch.Tensor:
    targets = getattr(train_dataset, "targets", None)
    if targets is None:
        raise AttributeError("train_dataset has no 'targets'; cannot compute global class counts.")
    if torch.is_tensor(targets):
        targets_t = targets.detach().cpu().long()
    else:
        targets_t = torch.tensor(list(targets), dtype=torch.long)

    counts = torch.zeros((num_classes,), dtype=torch.long)
    for cid, idxs in user_groups.items():
        ii = _as_int_indices(idxs)
        if len(ii) == 0:
            continue
        y = targets_t[ii]
        counts += torch.bincount(y, minlength=num_classes)
    return counts


def _build_or_load_pool(
    *,
    pool_path: str,
    rebuild_pool: int,
    cvae: ConditionalFeatureVAE,
    qualified_classes: List[int],
    global_counts: torch.Tensor,
    pool_ratio: float,
    pool_min: int,
    pool_max: int,
    temperature: float,
    device: str,
    logger: logging.Logger,
) -> Dict[int, torch.Tensor]:
    pool_path_abs = os.path.abspath(pool_path)
    pool_path_io = _maybe_win_long_path(pool_path_abs)

    if int(rebuild_pool) != 1 and os.path.exists(pool_path_io):
        logger.info(f"[pool] loading existing pool: {pool_path_abs}")
        obj = torch.load(pool_path_io, map_location="cpu")
        pools = obj.get("pools", None) if isinstance(obj, dict) else None
        if not isinstance(pools, dict):
            raise ValueError("Invalid pool file: missing dict key 'pools'")
        out = {int(k): v.float() for k, v in pools.items()}
        return out

    os.makedirs(_maybe_win_long_path(os.path.dirname(pool_path_abs)), exist_ok=True)
    logger.info(f"[pool] building pool (rebuild={int(rebuild_pool)}): {pool_path_abs}")

    num_classes = int(cvae.cfg.num_classes)
    pools: Dict[int, torch.Tensor] = {}
    for c in range(num_classes):
        n = int(global_counts[c].item())
        if c not in set(qualified_classes):
            pools[int(c)] = torch.empty((0, int(cvae.cfg.feature_dim)), dtype=torch.float32)
            continue
        m = int(round(float(pool_ratio) * float(n)))
        m = max(int(pool_min), min(int(pool_max), m))
        labels = torch.full((m,), int(c), dtype=torch.long)
        feats = generate_features(cvae, labels, num_per_label=1, temperature=float(temperature), device=device)
        pools[int(c)] = feats.float().contiguous()
        if (c + 1) % 2 == 0:
            logger.info(f"[pool] class {c}: n_global={n} -> M_c={m} shape={tuple(pools[int(c)].shape)}")

    payload = {
        "meta": {
            "stage": 4,
            "variant": "cvae_pool",
            "num_classes": int(cvae.cfg.num_classes),
            "feature_dim": int(cvae.cfg.feature_dim),
            "latent_dim": int(cvae.cfg.latent_dim),
            "pool_ratio": float(pool_ratio),
            "pool_min": int(pool_min),
            "pool_max": int(pool_max),
            "temperature": float(temperature),
            "qualified_classes": [int(x) for x in qualified_classes],
            "global_counts": global_counts.tolist(),
        },
        "pools": pools,
    }
    torch.save(payload, pool_path_io)
    logger.info("[pool] saved.")
    return pools


def _sample_from_pool(
    pools: Dict[int, torch.Tensor],
    y: torch.Tensor,
    *,
    device: str,
    return_indices: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    pools[c]: (M_c, D) on CPU; y: (B,) on device.
    returns x: (B, D) on device.
    """
    y_cpu = y.detach().cpu().long()
    out_chunks: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for c in torch.unique(y_cpu).tolist():
        c = int(c)
        mask = (y_cpu == c)
        idxs = torch.nonzero(mask, as_tuple=False).view(-1)
        pool_c = pools.get(c, None)
        if pool_c is None or pool_c.numel() == 0:
            raise ValueError(f"Pool for class {c} is empty; adjust label sampling or rebuild pool.")
        m = int(pool_c.shape[0])
        ridx = torch.randint(low=0, high=m, size=(int(idxs.numel()),), dtype=torch.long)
        x_sel = pool_c[ridx].to(device)
        out_chunks.append((idxs, x_sel, ridx))

    # stitch back in original order
    d = int(next(iter(pools.values())).shape[1])
    x = torch.empty((int(y_cpu.numel()), d), device=device, dtype=torch.float32)
    sel_indices = torch.empty((int(y_cpu.numel()),), dtype=torch.long) if return_indices else None
    for idxs, x_sel, ridx in out_chunks:
        x[idxs.to(device)] = x_sel
        if return_indices and sel_indices is not None:
            sel_indices[idxs] = ridx
    if return_indices:
        return x, sel_indices
    return x


def _compute_class_prototypes(
    model: nn.Module,
    dataset,
    idxs: List[int],
    *,
    batch_size: int,
    device: str,
) -> Dict[int, torch.Tensor]:
    """
    计算给定样本索引集合的各类别低层特征原型 (均值)。
    返回: {class_id: prototype (1,D) on CPU}
    """
    loader = DataLoader(
        Subset(dataset, idxs),
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    sums: Dict[int, torch.Tensor] = {}
    counts: Dict[int, int] = {}
    feature_dim: Optional[int] = None
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).long()
            _logits, _lp, _h, low_raw, _proj = model(images)
            if feature_dim is None:
                feature_dim = int(low_raw.shape[1])
            for cls in torch.unique(labels).tolist():
                cls = int(cls)
                mask = labels == cls
                if not mask.any():
                    continue
                feats = low_raw[mask]
                if cls not in sums:
                    sums[cls] = torch.zeros((int(feature_dim)), device=device, dtype=torch.float32)
                    counts[cls] = 0
                sums[cls] += feats.sum(dim=0)
                counts[cls] += int(mask.sum().item())
    out: Dict[int, torch.Tensor] = {}
    for cls, s in sums.items():
        cnt = counts.get(cls, 0)
        if cnt > 0:
            out[int(cls)] = (s / float(cnt)).detach().cpu()
    return out


@torch.no_grad()
def _compute_class_moments(
    model: nn.Module,
    dataset,
    idxs: List[int],
    *,
    num_classes: int,
    n_per_class: int,
    batch_size: int,
    device: str,
    seed: int,
) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor], Dict[int, int]]:
    """
    计算 low_raw 的 per-class 均值/方差（逐维 diag），用于 moment matching。

    returns:
      - means: {c: (D,) on CPU}
      - vars:  {c: (D,) on CPU}
      - counts:{c: n_used}
    """
    targets = getattr(dataset, "targets", None)
    if targets is None:
        return {}, {}, {}
    if torch.is_tensor(targets):
        targets_t = targets.detach().cpu().long()
    else:
        targets_t = torch.tensor(list(targets), dtype=torch.long)

    by_class: Dict[int, List[int]] = {int(c): [] for c in range(int(num_classes))}
    for i in idxs:
        y = int(targets_t[int(i)].item())
        if 0 <= y < int(num_classes):
            by_class[int(y)].append(int(i))

    rng = np.random.default_rng(int(seed))
    chosen: List[int] = []
    for c in range(int(num_classes)):
        pool = by_class[int(c)]
        if not pool:
            continue
        k = min(int(n_per_class), len(pool))
        sel = rng.choice(np.asarray(pool, dtype=np.int64), size=int(k), replace=False).tolist()
        chosen.extend(int(x) for x in sel)

    if not chosen:
        return {}, {}, {}

    loader = DataLoader(
        Subset(dataset, chosen),
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    sums: Dict[int, torch.Tensor] = {}
    sumsq: Dict[int, torch.Tensor] = {}
    counts: Dict[int, int] = {}
    feature_dim: Optional[int] = None
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).long()
            _logits, _lp, _h, low_raw, _proj = model(images)
            if feature_dim is None:
                feature_dim = int(low_raw.shape[1])
            for cls in torch.unique(labels).tolist():
                cls = int(cls)
                mask = labels == cls
                if not mask.any():
                    continue
                feats = low_raw[mask].float()
                if cls not in sums:
                    sums[cls] = torch.zeros((int(feature_dim)), device=device, dtype=torch.float32)
                    sumsq[cls] = torch.zeros((int(feature_dim)), device=device, dtype=torch.float32)
                    counts[cls] = 0
                sums[cls] += feats.sum(dim=0)
                sumsq[cls] += (feats * feats).sum(dim=0)
                counts[cls] += int(mask.sum().item())

    means: Dict[int, torch.Tensor] = {}
    vars_: Dict[int, torch.Tensor] = {}
    for cls, s in sums.items():
        n = int(counts.get(int(cls), 0))
        if n <= 0:
            continue
        mu = (s / float(n)).detach().cpu()
        ex2 = (sumsq[int(cls)] / float(n)).detach().cpu()
        var = (ex2 - mu * mu).clamp_min(0.0)
        means[int(cls)] = mu
        vars_[int(cls)] = var
    return means, vars_, {int(k): int(v) for k, v in counts.items()}


def _compute_pool_moments(pools: Dict[int, torch.Tensor]) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    """
    直接在 pool 上计算每类均值/方差（逐维 diag）。pools 在 CPU。
    returns:
      - mu_syn: {c: (D,) on CPU}
      - var_syn:{c: (D,) on CPU}
    """
    mu_syn: Dict[int, torch.Tensor] = {}
    var_syn: Dict[int, torch.Tensor] = {}
    for cls, feats in pools.items():
        if feats is None or feats.numel() == 0:
            continue
        x = feats.float()
        mu = x.mean(dim=0)
        ex2 = (x * x).mean(dim=0)
        var = (ex2 - mu * mu).clamp_min(0.0)
        mu_syn[int(cls)] = mu.detach().cpu()
        var_syn[int(cls)] = var.detach().cpu()
    return mu_syn, var_syn


def _build_moment_match_tables(
    *,
    num_classes: int,
    feature_dim: int,
    mu_real: Dict[int, torch.Tensor],
    var_real: Dict[int, torch.Tensor],
    mu_syn: Dict[int, torch.Tensor],
    var_syn: Dict[int, torch.Tensor],
    eps: float,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    构建用于批量 moment matching 的查表张量（device 上）：
      - mu_s_table: (C,D)
      - mu_r_table: (C,D)
      - scale_table: (C,D) 其中 scale = sqrt((var_r+eps)/(var_s+eps))
    缺失类默认 identity: mu=0, scale=1。
    """
    c = int(num_classes)
    d = int(feature_dim)
    mu_s = torch.zeros((c, d), dtype=torch.float32, device=device)
    mu_r = torch.zeros((c, d), dtype=torch.float32, device=device)
    scale = torch.ones((c, d), dtype=torch.float32, device=device)

    e = float(eps)
    for k, v in mu_syn.items():
        kk = int(k)
        if 0 <= kk < c:
            mu_s[kk] = v.float().to(device)
    for k, v in mu_real.items():
        kk = int(k)
        if 0 <= kk < c:
            mu_r[kk] = v.float().to(device)
    for k, v_s in var_syn.items():
        kk = int(k)
        if kk not in var_real:
            continue
        if 0 <= kk < c:
            vv_s = v_s.float().to(device).clamp_min(0.0)
            vv_r = var_real[kk].float().to(device).clamp_min(0.0)
            scale[kk] = torch.sqrt((vv_r + e) / (vv_s + e))

    return mu_s, mu_r, scale


def _apply_moment_match(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    mu_s_table: torch.Tensor,
    mu_r_table: torch.Tensor,
    scale_table: torch.Tensor,
) -> torch.Tensor:
    y = y.long()
    return (x - mu_s_table[y]) * scale_table[y] + mu_r_table[y]


def _shrink_moments(
    mu_local: Dict[int, torch.Tensor],
    var_local: Dict[int, torch.Tensor],
    *,
    mu_global: Dict[int, torch.Tensor],
    var_global: Dict[int, torch.Tensor],
    lam: float,
) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    """
    将本地 per-class moments 向全局 moments 做 shrinkage：
      mu_t = (1-lam)*mu_local + lam*mu_global
      var_t = (1-lam)*var_local + lam*var_global

    仅对本地已有的类做 shrink；若某类缺失 global moments，则保持本地值不变。
    """
    l = float(lam)
    if l <= 0.0:
        return mu_local, var_local

    mu_t: Dict[int, torch.Tensor] = {}
    var_t: Dict[int, torch.Tensor] = {}
    for cls, mu_l in mu_local.items():
        c = int(cls)
        mu_g = mu_global.get(c, None)
        var_l = var_local.get(c, None)
        var_g = var_global.get(c, None)
        if var_l is None:
            continue
        if mu_g is None or var_g is None:
            mu_t[c] = mu_l
            var_t[c] = var_l
        else:
            mu_t[c] = (1.0 - l) * mu_l + l * mu_g
            var_t[c] = (1.0 - l) * var_l + l * var_g
    return mu_t, var_t


def _filter_pools_by_prototypes(
    pools: Dict[int, torch.Tensor],
    prototypes: Dict[int, torch.Tensor],
    *,
    proportion: float,
    logger: Optional[logging.Logger] = None,
) -> Dict[int, torch.Tensor]:
    """
    按照与本地类原型的余弦相似度，筛选池中前 proportion 的特征。
    返回: 仅包含被过滤替换的类别，保持在 CPU。
    """
    if proportion is None or proportion <= 0:
        return {}

    filtered: Dict[int, torch.Tensor] = {}
    for cls, proto in prototypes.items():
        pool_c = pools.get(int(cls), None)
        if pool_c is None or pool_c.numel() == 0:
            continue
        total = int(pool_c.shape[0])
        k = max(1, int(math.ceil(total * float(proportion))))
        k = min(k, total)

        proto_n = F.normalize(proto.view(1, -1), dim=1)
        pool_n = F.normalize(pool_c, dim=1)
        sims = torch.matmul(pool_n, proto_n.t()).view(-1)
        topk = torch.topk(sims, k=k, largest=True, sorted=False)
        filtered[int(cls)] = pool_c[topk.indices].contiguous()

        if logger is not None:
            logger.info(
                f"[filter_pool] class={cls} kept={int(filtered[int(cls)].shape[0])}/{total} proportion={proportion:.3f}"
            )
    return filtered


def _merge_filtered_pools(
    base_pools: Dict[int, torch.Tensor],
    filtered: Dict[int, torch.Tensor],
) -> Dict[int, torch.Tensor]:
    """
    用过滤后的池覆盖对应类别，其余类别保持原样。
    返回新的 dict，不修改原始 base_pools。
    """
    out: Dict[int, torch.Tensor] = {}
    for cls, pool in base_pools.items():
        if int(cls) in filtered:
            out[int(cls)] = filtered[int(cls)]
        else:
            out[int(cls)] = pool
    return out


def main() -> int:
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("--stage1_ckpt_path", type=str, required=True, help="Stage-1 checkpoint path (best-wo.pt / latest.pt)")
    p.add_argument("--cvae_path", type=str, required=True, help="Stage-3 cVAE generator.pt path (or its directory)")
    p.add_argument("--split_path", type=str, default=None, help="split.pkl path (if None, infer from ckpt meta/args)")
    p.add_argument("--out_dir", type=str, default=None, help="Output dir (default: <ckpt.meta.logdir>/stage4/finetune_cvae_pool)")
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

    # Synthetic controls (CE-only)
    p.add_argument("--syn_ratio", type=float, default=0.2, help="Synthetic batch size ratio relative to real batch (0..1)")
    p.add_argument("--syn_batch_size", type=int, default=-1, help="If >0, override synthetic batch size per step.")
    p.add_argument("--syn_alpha", type=float, default=1.0, help="Weight for synthetic CE loss.")
    p.add_argument(
        "--syn_scale_by_ratio",
        type=int,
        default=1,
        choices=[0, 1],
        help="If 1, scale synthetic CE by (b_syn/b_real) so synthetic signal doesn't dominate when b_syn is small/large.",
    )
    p.add_argument(
        "--syn_label_sampling",
        type=str,
        default="global_uniform",
        choices=["global_uniform", "batch_labels", "client_classes"],
        help="How to sample synthetic labels y_syn.",
    )
    p.add_argument(
        "--syn_filter_proportion",
        type=float,
        default=None,
        help="When using client_classes sampling, keep top proportion (0,1] synthetic features most similar to class prototype (source set by --syn_filter_target). None or <=0 disables filtering.",
    )
    p.add_argument(
        "--syn_filter_target",
        type=str,
        default="local",
        choices=["local", "global"],
        help="Prototype source for filtering synthetic pool when syn_filter_proportion>0. 'local' uses per-client real data; 'global' uses global prototypes computed once.",
    )
    p.add_argument(
        "--syn_moment_match",
        type=int,
        default=0,
        choices=[0, 1],
        help="Apply per-class moment matching to synthetic features before forward_from_low (Probe A).",
    )
    p.add_argument(
        "--syn_moment_eps",
        type=float,
        default=1e-6,
        help="Epsilon for moment matching scale: sqrt((var_real+eps)/(var_syn+eps)).",
    )
    p.add_argument(
        "--syn_moment_n_real_per_class",
        type=int,
        default=200,
        help="Max real samples per class (per client) to estimate low_raw moments for moment matching.",
    )
    p.add_argument(
        "--syn_moment_shrink",
        type=float,
        default=0.0,
        help=(
            "Shrink per-class real moments towards global real moments (computed using the same client model "
            "but over all clients' train indices). 0 disables (default). "
            "Target moments: mu_t=(1-lam)*mu_local+lam*mu_global, var_t=(1-lam)*var_local+lam*var_global."
        ),
    )
    p.add_argument(
        "--syn_moment_n_global_per_class",
        type=int,
        default=200,
        help="Max real samples per class (per client model) to estimate global moments for shrinkage.",
    )

    # Pool building (auto)
    p.add_argument("--pool_ratio", type=float, default=1.0)
    p.add_argument("--pool_min", type=int, default=200)
    p.add_argument("--pool_max", type=int, default=5000)
    p.add_argument("--pool_temperature", type=float, default=1.0)
    p.add_argument("--pool_path", type=str, default=None, help="Path to cache the feature pool (default: <out_dir>/feature_pool.pt)")
    p.add_argument("--rebuild_pool", type=int, default=0, choices=[0, 1])
    p.add_argument("--build_pool_only", type=int, default=0, choices=[0, 1])

    # cVAE rebuild knobs (only used if meta missing some fields)
    p.add_argument("--cvae_y_emb_dim", type=int, default=32)

    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--eval_interval", type=int, default=200)
    p.add_argument("--save_best", type=int, default=1)
    p.add_argument("--select_best_for_after", type=int, default=1)

    args = p.parse_args()
    if args.seed is None:
        args.seed = int(time.time_ns() % (2**31 - 1))
    if args.syn_filter_proportion is not None:
        if float(args.syn_filter_proportion) <= 0 or float(args.syn_filter_proportion) > 1:
            raise ValueError("--syn_filter_proportion must be in (0,1]; got {}".format(args.syn_filter_proportion))
    if float(args.syn_moment_shrink) < 0.0 or float(args.syn_moment_shrink) > 1.0:
        raise ValueError("--syn_moment_shrink must be in [0,1]; got {}".format(args.syn_moment_shrink))

    device = _resolve_device(args.gpu)
    _seed_all(int(args.seed))

    ckpt = load_checkpoint(args.stage1_ckpt_path, map_location="cpu")
    ckpt_meta = (ckpt.get("meta", {}) or {}) if isinstance(ckpt, dict) else {}
    ckpt_state = (ckpt.get("state", {}) or {}) if isinstance(ckpt, dict) else {}
    ckpt_args = (ckpt.get("args", {}) or {}) if isinstance(ckpt, dict) else {}

    dataset = str(ckpt_meta.get("dataset", ckpt_args.get("dataset", "cifar10")))
    if dataset != "cifar10":
        raise ValueError(f"Stage-4(cvae_pool) currently supports cifar10 only. Got dataset={dataset}")
    num_users = int(ckpt_meta.get("num_users", ckpt_args.get("num_users", 20)))
    num_classes = int(ckpt_meta.get("num_classes", ckpt_args.get("num_classes", 10)))

    split_path = args.split_path or ckpt_meta.get("split_path", None) or ckpt_args.get("split_path", None)
    if not split_path or not os.path.exists(split_path):
        raise FileNotFoundError(f"split_path not found: {split_path}")

    logdir = _infer_logdir_from_ckpt(args.stage1_ckpt_path, ckpt_meta)
    base_out_dir = args.out_dir or os.path.join(logdir, "stage4", "finetune_cvae_pool")
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

    out_dir = os.path.abspath(out_dir)
    os.makedirs(_maybe_win_long_path(out_dir), exist_ok=True)
    logger = _setup_logger(out_dir)
    logger.info(f"[stage4_cvae_pool] out_dir={out_dir}")
    logger.info(f"[stage4_cvae_pool] run_tag={run_tag} auto_run_dir={int(args.auto_run_dir)} run_name={args.run_name}")
    logger.info(f"[stage4_cvae_pool] resolved_seed={args.seed}")
    logger.info(f"[stage4_cvae_pool] cli_args={vars(args)}")
    tb_dir = os.path.join(out_dir, "tensorboard")
    os.makedirs(_maybe_win_long_path(tb_dir), exist_ok=True)
    writer = SummaryWriter(log_dir=_maybe_win_long_path(tb_dir))
    logger.info(f"[stage4_cvae_pool] tensorboard_dir={tb_dir}")

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

    # Load cVAE
    cvae_path = _resolve_cvae_path(args.cvae_path)
    cvae, cvae_meta = _load_cvae(cvae_path, device=device, y_emb_dim=int(args.cvae_y_emb_dim))
    if int(cvae.cfg.num_classes) != int(num_classes):
        raise RuntimeError(f"num_classes mismatch: stage1={num_classes} vs cvae={int(cvae.cfg.num_classes)}")
    logger.info(
        f"[stage4_cvae_pool] loaded cVAE: path={cvae_path} feature_dim={int(cvae.cfg.feature_dim)} latent_dim={int(cvae.cfg.latent_dim)}"
    )

    # Compute global counts and qualified classes
    global_counts = _compute_global_class_counts(train_dataset, user_groups=user_groups, num_classes=num_classes)
    qualified = [int(i) for i in torch.nonzero(global_counts > 0, as_tuple=False).view(-1).tolist()]
    if len(qualified) == 0:
        raise ValueError("No qualified classes found (global_counts all zero).")
    logger.info(f"[stage4_cvae_pool] qualified_classes={qualified}")
    logger.info(f"[stage4_cvae_pool] global_counts={global_counts.tolist()}")

    local_sd = ckpt_state.get("local_models_full_state_dicts", None)
    if not isinstance(local_sd, dict):
        raise ValueError("Stage-1 checkpoint missing state['local_models_full_state_dicts']")

    # Cache global train indices once (used by moment shrinkage; also useful elsewhere)
    all_train_indices = _as_int_indices([user_groups[int(cid)] for cid in range(int(num_users))])

    # 可选：预先计算全局原型（一次）
    global_prototypes: Optional[Dict[int, torch.Tensor]] = None
    if (
        args.syn_filter_proportion is not None
        and float(args.syn_filter_proportion) > 0
        and str(args.syn_filter_target).lower() == "global"
    ):
        if len(all_train_indices) == 0:
            logger.info("[global_proto] no train indices found; skip global prototype computation.")
        else:
            ref_key = next(iter(local_sd.keys()))
            ref_model = CNNCifar(args=SimpleNamespace(**{"num_classes": num_classes})).to(device)
            ref_model.load_state_dict(local_sd[ref_key], strict=True)
            logger.info(
                f"[global_proto] computing prototypes using ref client={ref_key} samples={len(all_train_indices)} "
                f"batch_size={int(args.batch_size)}"
            )
            global_prototypes = _compute_class_prototypes(
                ref_model,
                train_dataset,
                all_train_indices,
                batch_size=int(args.batch_size),
                device=device,
            )
            logger.info(
                f"[global_proto] done. classes_with_proto={sorted(list(global_prototypes.keys())) if global_prototypes else []}"
            )
            ref_model.cpu()
            del ref_model

    # Build/load pool
    pool_path = args.pool_path or os.path.join(out_dir, "feature_pool.pt")
    pools = _build_or_load_pool(
        pool_path=pool_path,
        rebuild_pool=int(args.rebuild_pool),
        cvae=cvae,
        qualified_classes=qualified,
        global_counts=global_counts,
        pool_ratio=float(args.pool_ratio),
        pool_min=int(args.pool_min),
        pool_max=int(args.pool_max),
        temperature=float(args.pool_temperature),
        device=device,
        logger=logger,
    )

    if int(args.build_pool_only) == 1:
        logger.info("[stage4_cvae_pool] build_pool_only=1 -> done.")
        return 0

    # Pre-compute per-client present classes (for syn_label_sampling='client_classes')
    client_present: Dict[int, List[int]] = {}
    classes_list = split.get("classes_list", None)
    qualified_set = set(qualified)
    if classes_list is not None:
        for cid in range(int(num_users)):
            if isinstance(classes_list, dict):
                present = set(int(x) for x in classes_list.get(int(cid), []))
            else:
                present = set(int(x) for x in classes_list[int(cid)])
            present = sorted(list(present & qualified_set))
            client_present[int(cid)] = present
    else:
        targets = getattr(train_dataset, "targets", None)
        if targets is None:
            raise AttributeError("train_dataset has no 'targets'; cannot infer client classes without split['classes_list'].")
        for cid in range(int(num_users)):
            idxs = _as_int_indices(user_groups[int(cid)])
            present = set(int(targets[i]) for i in idxs)
            present = sorted(list(present & qualified_set))
            client_present[int(cid)] = present

    meta_path = os.path.join(out_dir, "stage4_cvae_pool_meta.json")
    os.makedirs(_maybe_win_long_path(os.path.dirname(meta_path)), exist_ok=True)
    with open(_maybe_win_long_path(meta_path), "w", encoding="utf-8") as f:
        json.dump(
            {
                "stage": 4,
                "variant": "cvae_pool",
                "dataset": "cifar10",
                "num_users": num_users,
                "num_classes": num_classes,
                "stage1_ckpt_path": args.stage1_ckpt_path,
                "cvae_path": cvae_path,
                "cvae_meta_path": os.path.join(os.path.dirname(os.path.abspath(cvae_path)), "generator_meta.json"),
                "split_path": split_path,
                "out_dir": out_dir,
                "run_tag": run_tag,
                "run_name": args.run_name,
                "auto_run_dir": int(args.auto_run_dir),
                "pool_path": pool_path,
                "pool_cfg": {
                    "pool_ratio": float(args.pool_ratio),
                    "pool_min": int(args.pool_min),
                    "pool_max": int(args.pool_max),
                    "pool_temperature": float(args.pool_temperature),
                    "rebuild_pool": int(args.rebuild_pool),
                    "qualified_classes": qualified,
                    "global_counts": global_counts.tolist(),
                },
                "hparams": {
                    "steps": int(args.steps),
                    "batch_size": int(args.batch_size),
                    "lr": float(args.lr),
                    "weight_decay": float(args.weight_decay),
                    "syn_ratio": float(args.syn_ratio),
                    "syn_batch_size": int(args.syn_batch_size),
                    "syn_alpha": float(args.syn_alpha),
                    "syn_scale_by_ratio": int(args.syn_scale_by_ratio),
                    "syn_label_sampling": str(args.syn_label_sampling),
                    "syn_filter_proportion": float(args.syn_filter_proportion) if args.syn_filter_proportion is not None else None,
                    "syn_filter_target": str(args.syn_filter_target),
                    "syn_moment_match": int(args.syn_moment_match),
                    "syn_moment_eps": float(args.syn_moment_eps),
                    "syn_moment_n_real_per_class": int(args.syn_moment_n_real_per_class),
                    "syn_moment_shrink": float(args.syn_moment_shrink),
                    "syn_moment_n_global_per_class": int(args.syn_moment_n_global_per_class),
                    "seed": int(args.seed),
                },
                "cvae_cfg": {
                    "feature_dim": int(cvae.cfg.feature_dim),
                    "num_classes": int(cvae.cfg.num_classes),
                    "latent_dim": int(cvae.cfg.latent_dim),
                    "hidden_dim": int(cvae.cfg.hidden_dim),
                    "n_hidden": int(cvae.cfg.n_hidden),
                    "y_emb_dim": int(cvae.cfg.y_emb_dim),
                },
                "cvae_meta": cvae_meta,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    results: List[Dict[str, object]] = []
    start_all = time.time()

    for cid in range(num_users):
        m = CNNCifar(args=SimpleNamespace(**{"num_classes": num_classes})).to(device)
        key = cid if cid in local_sd else str(cid)
        if key not in local_sd:
            raise KeyError(f"local_models_full_state_dicts missing client id={cid}")
        m.load_state_dict(local_sd[key], strict=True)

        # Freeze low encoder + projector; train only fc0/fc1/fc2
        for pp in m.parameters():
            pp.requires_grad_(False)
        for pp in m.fc0.parameters():
            pp.requires_grad_(True)
        for pp in m.fc1.parameters():
            pp.requires_grad_(True)
        for pp in m.fc2.parameters():
            pp.requires_grad_(True)
        _assert_freeze_setup(m)

        train_idxs = _as_int_indices(user_groups[int(cid)])
        test_idxs = _as_int_indices(user_groups_lt[int(cid)])
        dl_train = DataLoader(
            Subset(train_dataset, train_idxs),
            batch_size=int(args.batch_size),
            shuffle=True,
            num_workers=0,
            drop_last=True,
        )
        dl_test = DataLoader(
            Subset(test_dataset, test_idxs),
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )

        # 可选：基于本地类原型过滤合成特征池
        use_filter = (
            args.syn_filter_proportion is not None
            and float(args.syn_filter_proportion) > 0
            and str(args.syn_label_sampling).lower() == "client_classes"
        )
        client_pools = pools
        filter_kept: Dict[int, Dict[str, int]] = {}
        if use_filter:
            if str(args.syn_filter_target).lower() == "global":
                prototypes = global_prototypes or {}
            else:
                prototypes = _compute_class_prototypes(
                    m,
                    train_dataset,
                    train_idxs,
                    batch_size=int(args.batch_size),
                    device=device,
                )
            if len(prototypes) == 0:
                logger.info(f"[filter_pool] cid={cid:02d} no prototypes found (target={args.syn_filter_target}), fallback to global pool.")
            else:
                filtered = _filter_pools_by_prototypes(
                    pools,
                    prototypes,
                    proportion=float(args.syn_filter_proportion),
                    logger=logger,
                )
                if len(filtered) > 0:
                    client_pools = _merge_filtered_pools(pools, filtered)
                    filter_kept = {
                        int(c): {
                            "kept": int(client_pools[int(c)].shape[0]),
                            "base": int(pools[int(c)].shape[0]),
                        }
                        for c in filtered.keys()
                    }
                    logger.info(
                        f"[filter_pool] cid={cid:02d} syn_filter_proportion={float(args.syn_filter_proportion):.3f} "
                        f"filtered_classes={sorted(list(filtered.keys()))}"
                    )
                else:
                    logger.info(f"[filter_pool] cid={cid:02d} prototypes ok but no filtered pools produced, fallback.")

        # Synthetic moment matching tables (per client; depends on client_pools & client model)
        mm_tables = None
        if int(args.syn_moment_match) == 1:
            # (1) real moments from local real data (low_real from this client's frozen convs)
            mu_real, var_real, cnt_real = _compute_class_moments(
                m,
                train_dataset,
                train_idxs,
                num_classes=int(num_classes),
                n_per_class=int(args.syn_moment_n_real_per_class),
                batch_size=int(args.batch_size),
                device=device,
                seed=int(args.seed) + 10000 + int(cid),
            )
            # Optional: shrink local moments towards "global" moments (same client model, all clients' train indices)
            lam = float(args.syn_moment_shrink)
            if lam > 0.0:
                if len(all_train_indices) == 0:
                    logger.info(f"[syn_mm][shrink] cid={cid:02d} all_train_indices empty; skip shrinkage.")
                else:
                    mu_g, var_g, cnt_g = _compute_class_moments(
                        m,
                        train_dataset,
                        all_train_indices,
                        num_classes=int(num_classes),
                        n_per_class=int(args.syn_moment_n_global_per_class),
                        batch_size=int(args.batch_size),
                        device=device,
                        seed=int(args.seed) + 20000 + int(cid),
                    )
                    mu_real, var_real = _shrink_moments(mu_real, var_real, mu_global=mu_g, var_global=var_g, lam=lam)
            # (2) synthetic moments directly from (possibly filtered) pool
            mu_syn, var_syn = _compute_pool_moments(client_pools)
            if len(mu_real) == 0 or len(var_real) == 0 or len(mu_syn) == 0 or len(var_syn) == 0:
                logger.info(f"[syn_mm] cid={cid:02d} insufficient moments; disable moment matching for this client.")
            else:
                feat_dim = int(cvae.cfg.feature_dim)
                mu_s_table, mu_r_table, scale_table = _build_moment_match_tables(
                    num_classes=int(num_classes),
                    feature_dim=int(feat_dim),
                    mu_real=mu_real,
                    var_real=var_real,
                    mu_syn=mu_syn,
                    var_syn=var_syn,
                    eps=float(args.syn_moment_eps),
                    device=device,
                )
                mm_tables = {"mu_s_table": mu_s_table, "mu_r_table": mu_r_table, "scale_table": scale_table}
                logger.info(
                    f"[syn_mm] cid={cid:02d} enabled. classes_real={len(mu_real)} classes_pool={len(mu_syn)} "
                    f"n_real_per_class={int(args.syn_moment_n_real_per_class)}"
                )

        # Sanity: infer low dim from a real batch
        m.eval()
        images0, labels0 = next(iter(dl_train))
        images0 = images0.to(device)
        with torch.no_grad():
            _logits0, _lp0, _h0, low_raw0, _proj0 = m(images0)
        if int(low_raw0.shape[1]) != int(cvae.cfg.feature_dim):
            raise RuntimeError(
                f"low_feature_dim mismatch: cvae.feature_dim={int(cvae.cfg.feature_dim)} vs model.low_dim={int(low_raw0.shape[1])}"
            )

        opt = torch.optim.Adam(
            [pp for pp in list(m.fc0.parameters()) + list(m.fc1.parameters()) + list(m.fc2.parameters()) if pp.requires_grad],
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
        )
        _assert_optimizer_params(opt, allowed=list(m.fc0.parameters()) + list(m.fc1.parameters()) + list(m.fc2.parameters()))

        acc0, loss0 = _eval_client_on_images(m, dl_test, device=device, num_classes=num_classes)
        writer.add_scalar(f"client_{cid:02d}/eval/acc_before", float(acc0), 0)
        writer.add_scalar(f"client_{cid:02d}/eval/loss_before", float(loss0), 0)

        m.train()
        step = 0
        t0 = time.time()
        it = iter(dl_train)
        last_loss = None
        # 改为初始化为 -1，记录微调过程中的最佳准确率（而非以 acc_before 为门槛）
        best_acc = -1.0
        best_loss = float('inf')
        best_step = 0
        best_state = None

        # 区间累积器：只在 eval 间隔写一次
        label_counts: Counter = Counter()
        index_counts: Dict[int, Counter] = defaultdict(Counter)
        total_syn = 0
        interval_start = 1

        selection_log_path = os.path.join(out_dir, f"client_{cid:02d}_syn_selection.jsonl")
        os.makedirs(_maybe_win_long_path(os.path.dirname(selection_log_path)), exist_ok=True)
        with open(_maybe_win_long_path(selection_log_path), "w", encoding="utf-8") as sel_log:
            sel_log.write(
                json.dumps(
                    {
                        "client_id": int(cid),
                        "type": "filter_meta",
                        "syn_filter_proportion": float(args.syn_filter_proportion)
                        if args.syn_filter_proportion is not None
                        else None,
                        "syn_filter_target": str(args.syn_filter_target),
                        "syn_moment_match": int(args.syn_moment_match),
                        "syn_moment_eps": float(args.syn_moment_eps),
                        "syn_moment_n_real_per_class": int(args.syn_moment_n_real_per_class),
                        "syn_moment_shrink": float(args.syn_moment_shrink),
                        "syn_moment_n_global_per_class": int(args.syn_moment_n_global_per_class),
                        "filtered_classes": sorted(list(filter_kept.keys())) if filter_kept else [],
                        "filter_kept": filter_kept,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            while step < int(args.steps):
                try:
                    images, labels = next(it)
                except StopIteration:
                    it = iter(dl_train)
                    images, labels = next(it)
                images = images.to(device)
                labels = labels.to(device).long()

                with torch.no_grad():
                    _logits_r, _lp_r, _h_r, low_real, _proj_r = m(images)

                b_real = int(low_real.shape[0])
                b_syn = int(round(float(b_real) * float(args.syn_ratio)))
                if int(args.syn_batch_size) > 0:
                    b_syn = int(args.syn_batch_size)

                logits_real, _lp_real, _h_real, _proj_real = m.forward_from_low(low_real)
                logits_real = logits_real[:, 0:num_classes]
                real_ce = F.cross_entropy(logits_real, labels, reduction="mean")

                syn_ce = torch.tensor(0.0, device=device)
                ratio = (float(b_syn) / float(b_real)) if (b_real > 0) else 0.0

                if b_syn > 0:
                    sampling = str(args.syn_label_sampling).lower()
                    if sampling == "global_uniform":
                        q = torch.tensor(qualified, dtype=torch.long, device=device)
                        ridx = torch.randint(low=0, high=int(q.numel()), size=(b_syn,), device=device)
                        y_syn = q[ridx].long()
                    elif sampling == "batch_labels":
                        ridx = torch.randint(low=0, high=b_real, size=(b_syn,), device=device)
                        y_syn = labels[ridx].long()
                    elif sampling == "client_classes":
                        present = client_present.get(int(cid), []) or []
                        if len(present) == 0:
                            ridx = torch.randint(low=0, high=b_real, size=(b_syn,), device=device)
                            y_syn = labels[ridx].long()
                        else:
                            present_t = torch.tensor(present, dtype=torch.long, device=device)
                            ridx = torch.randint(low=0, high=int(present_t.numel()), size=(b_syn,), device=device)
                            y_syn = present_t[ridx].long()
                    else:
                        raise ValueError(f"Unsupported --syn_label_sampling: {args.syn_label_sampling}")

                    x_syn, sel_idx = _sample_from_pool(client_pools, y_syn, device=device, return_indices=True)
                    if mm_tables is not None:
                        x_syn = _apply_moment_match(
                            x_syn,
                            y_syn,
                            mu_s_table=mm_tables["mu_s_table"],
                            mu_r_table=mm_tables["mu_r_table"],
                            scale_table=mm_tables["scale_table"],
                        )
                    logits_syn, _lp_syn, _h_syn, _proj_syn = m.forward_from_low(x_syn)
                    logits_syn = logits_syn[:, 0:num_classes]
                    syn_ce = F.cross_entropy(logits_syn, y_syn, reduction="mean")

                    # 累积本区间的标签与池索引频次
                    ys_cpu = y_syn.detach().cpu().tolist()
                    idx_cpu = sel_idx.detach().cpu().tolist()
                    label_counts.update(int(y) for y in ys_cpu)
                    for yv, iv in zip(ys_cpu, idx_cpu):
                        index_counts[int(yv)][int(iv)] += 1
                    total_syn += int(b_syn)

                if int(args.syn_scale_by_ratio) == 1:
                    syn_ce_eff = syn_ce * ratio
                else:
                    syn_ce_eff = syn_ce

                alpha = float(args.syn_alpha)
                loss = real_ce + alpha * syn_ce_eff

                opt.zero_grad(set_to_none=True)
                loss.backward()
                if step == 0:
                    _assert_grad_flow(m)
                opt.step()
                last_loss = float(loss.item())

                if (step + 1) % int(args.log_interval) == 0:
                    dt = time.time() - t0
                    logger.info(
                        f"[stage4_cvae_pool][cid={cid:02d}] step {step+1}/{int(args.steps)} "
                        f"loss={last_loss:.6f} real_ce={float(real_ce.item()):.6f} syn_ce={float(syn_ce.item()):.6f} "
                        f"ratio={ratio:.3f} alpha={alpha:.4g} sec={dt:.1f}"
                    )
                    global_step = step + 1
                    writer.add_scalar(f"client_{cid:02d}/train/loss", float(last_loss), global_step)
                    writer.add_scalar(f"client_{cid:02d}/train/real_ce", float(real_ce.item()), global_step)
                    writer.add_scalar(f"client_{cid:02d}/train/syn_ce", float(syn_ce.item()), global_step)
                    writer.add_scalar(f"client_{cid:02d}/train/syn_ratio", float(ratio), global_step)

                if (step + 1) % int(args.eval_interval) == 0 or (step + 1) == int(args.steps):
                    acc1, loss1 = _eval_client_on_images(m, dl_test, device=device, num_classes=num_classes)
                    logger.info(f"[stage4_cvae_pool][cid={cid:02d}] eval@{step+1}: acc={acc1:.4f} loss={loss1:.4f}")
                    if int(args.save_best) == 1 and float(acc1) >= float(best_acc):
                        best_acc = float(acc1)
                        best_loss = float(loss1)
                        best_step = int(step + 1)
                        best_state = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
                        writer.add_scalar(f"client_{cid:02d}/eval/best_acc", float(best_acc), best_step)
                        writer.add_scalar(f"client_{cid:02d}/eval/best_step", float(best_step), best_step)
                    m.train()
                    eval_step = step + 1
                    writer.add_scalar(f"client_{cid:02d}/eval/acc", float(acc1), eval_step)
                    writer.add_scalar(f"client_{cid:02d}/eval/loss", float(loss1), eval_step)
                    writer.flush()

                    # 在两次 eval 之间写一条汇总记录
                    rec = {
                        "client_id": int(cid),
                        "eval_step": int(step + 1),
                        "step_start": int(interval_start),
                        "step_end": int(step + 1),
                        "total_syn": int(total_syn),
                        "label_counts": {int(k): int(v) for k, v in label_counts.items()},
                        "index_counts": {
                            int(lbl): {int(idx): int(cnt) for idx, cnt in cnts.items()}
                            for lbl, cnts in index_counts.items()
                        },
                    }
                    sel_log.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    sel_log.flush()

                    # 重置区间累积器
                    label_counts.clear()
                    index_counts = defaultdict(Counter)
                    total_syn = 0
                    interval_start = int(step + 2)

                step += 1

        accF, lossF = _eval_client_on_images(m, dl_test, device=device, num_classes=num_classes)
        # 如果开启 save_best 但 best_state 仍为 None（理论上不会出现，因为初始 best_acc=-1），则兜底用最终模型
        if int(args.save_best) == 1 and best_state is None:
            best_state = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
            best_acc = float(accF)
            best_loss = float(lossF)
            best_step = int(args.steps)

        use_best = (int(args.select_best_for_after) == 1) and (best_state is not None)
        acc_after = float(best_acc) if use_best else float(accF)
        loss_after = float(best_loss) if use_best else float(lossF)

        out_path = os.path.join(out_dir, f"client_{cid:02d}.pt")
        os.makedirs(_maybe_win_long_path(os.path.dirname(out_path)), exist_ok=True)
        torch.save(
            {
                "meta": {
                    "stage": 4,
                    "variant": "cvae_pool",
                    "client_id": cid,
                    "dataset": "cifar10",
                    "num_classes": num_classes,
                    "base_stage1_ckpt": args.stage1_ckpt_path,
                    "cvae_path": cvae_path,
                    "split_path": split_path,
                    "pool_path": pool_path,
                    "steps": int(args.steps),
                    "batch_size": int(args.batch_size),
                    "lr": float(args.lr),
                    "seed": int(args.seed),
                    "syn_ratio": float(args.syn_ratio),
                    "syn_batch_size": int(args.syn_batch_size),
                    "syn_alpha": float(args.syn_alpha),
                    "syn_scale_by_ratio": int(args.syn_scale_by_ratio),
                    "syn_label_sampling": str(args.syn_label_sampling),
                    "syn_filter_proportion": float(args.syn_filter_proportion) if args.syn_filter_proportion is not None else None,
                    "syn_filter_target": str(args.syn_filter_target),
                    "syn_moment_match": int(args.syn_moment_match),
                    "syn_moment_eps": float(args.syn_moment_eps),
                    "syn_moment_n_real_per_class": int(args.syn_moment_n_real_per_class),
                    "syn_filter_kept": {int(k): {kk: int(vv) for kk, vv in v.items()} for k, v in filter_kept.items()},
                    "acc_before": float(acc0),
                    "loss_before": float(loss0),
                    "acc_after": float(acc_after),
                    "loss_after": float(loss_after),
                    "best_step": int(best_step),
                    "used_best_for_after": bool(use_best),
                },
                "state_dict": (best_state if use_best else m.state_dict()),
            },
            _maybe_win_long_path(out_path),
        )

        results.append(
            {
                "client_id": int(cid),
                "acc_before": float(acc0),
                "loss_before": float(loss0),
                "acc_after": float(acc_after),
                "loss_after": float(loss_after),
                "best_step": int(best_step),
            }
        )
        logger.info(
            f"[stage4_cvae_pool][cid={cid:02d}] done: acc {float(acc0):.4f} -> {float(acc_after):.4f} "
            f"(best_step={int(best_step)}) saved={out_path}"
        )
        writer.add_scalar(
            f"client_{cid:02d}/eval/acc_after",
            float(acc_after),
            int(best_step if use_best else args.steps),
        )
        writer.add_scalar(
            f"client_{cid:02d}/eval/loss_after",
            float(loss_after),
            int(best_step if use_best else args.steps),
        )
        writer.flush()

    dt_all = time.time() - start_all
    avg_before = float(np.mean([r["acc_before"] for r in results])) if results else 0.0
    avg_after = float(np.mean([r["acc_after"] for r in results])) if results else 0.0
    logger.info(f"[stage4_cvae_pool] finished: avg_acc {avg_before:.4f} -> {avg_after:.4f} time={dt_all:.1f}s")

    with open(_maybe_win_long_path(os.path.join(out_dir, "results.json")), "w", encoding="utf-8") as f:
        json.dump({"results": results, "avg_before": avg_before, "avg_after": avg_after}, f, ensure_ascii=False, indent=2)

    writer.add_scalar("global/avg_acc_before", avg_before, 0)
    writer.add_scalar("global/avg_acc_after", avg_after, int(args.steps))
    writer.flush()
    writer.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


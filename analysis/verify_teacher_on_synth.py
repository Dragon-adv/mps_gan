#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
验证：Stage-3 cVAE 生成的“合成低级特征”到底能不能用？

核心观察点（Teacher Accuracy）：
  - 对每个客户端的 Stage-1 local_model(teacher)，在合成数据 (x_syn, y_syn) 上计算 top-1 accuracy。
  - 如果合成特征是“四不像”，teacher 对它的分类准确率会很低。

结论解读（经验阈值）：
  - 若 teacher 对合成数据的分类准确率长期低于 0.80 ~ 0.90，
    说明 Stage-3 的生成需要加“语义约束”（否则生成的特征不对齐 label）。

运行示例（Windows PowerShell）：
  py analysis/verify_teacher_on_synth.py `
    --stage1_ckpt_path "D:\path\to\stage1\ckpts\best-wo.pt" `
    --cvae_path "D:\path\to\stage3\cvae\...\generator.pt" `
    --label_sampling both `
    --n_syn 10000 `
    --gpu 0

输出：
  - 在 out_dir 下写 summary.json（含每客户端、两种采样策略的 acc/conf/entropy 等统计）。
  - （可选诊断）在 out_dir/diagnostics 下写：
      - 每 client、每采样策略：y(合成标签) vs yhat(teacher预测) 的混淆矩阵 + yhat 边际分布
      - 每 client：真实 low_raw vs 合成特征（按类）均值/方差、以及合成特征与真实原型的余弦相似度分布

注意：
  - 该脚本不跑 Stage-4 训练，只做“teacher 在合成数据上的可用性评估”。
  - pool 构建/采样逻辑与 exps/run_stage4_cvae_pool.py 对齐。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import Counter
from datetime import datetime
from statistics import mean, median
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

from lib.checkpoint import load_checkpoint  # noqa: E402
from lib.cvae_feature_gen import CVAEConfig, ConditionalFeatureVAE, generate_features  # noqa: E402
from lib.models.models import CNNCifar  # noqa: E402
from lib.split_manager import load_split  # noqa: E402
from lib.utils import get_dataset  # noqa: E402


def _try_import_matplotlib():
    try:
        import matplotlib.pyplot as plt  # type: ignore

        return plt
    except Exception:
        return None


def _maybe_win_long_path(path: str) -> str:
    if os.name != "nt":
        return path
    p = os.path.abspath(path)
    if len(p) < 240:
        return p
    if p.startswith("\\\\?\\"):
        return p
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
        if os.path.exists(_maybe_win_long_path(cand)):
            return cand
        raise FileNotFoundError(f"--cvae_path is a directory but generator.pt not found: {cand}")
    if cvae_path_cli and os.path.exists(_maybe_win_long_path(cvae_path_cli)):
        return cvae_path_cli
    raise FileNotFoundError(f"cVAE generator.pt not found: {cvae_path_cli}")


def _load_cvae_meta(gen_path: str) -> dict:
    d = os.path.dirname(os.path.abspath(gen_path))
    meta_path = os.path.join(d, "generator_meta.json")
    meta_path_io = _maybe_win_long_path(meta_path)
    if not os.path.exists(meta_path_io):
        raise FileNotFoundError(f"generator_meta.json not found next to cVAE generator: {meta_path}")
    with open(meta_path_io, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_cvae(gen_path: str, *, device: str, y_emb_dim: int = 32) -> Tuple[ConditionalFeatureVAE, dict]:
    meta = _load_cvae_meta(gen_path)
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


def _bincount_confusion(y_true: torch.Tensor, y_pred: torch.Tensor, *, num_classes: int) -> torch.Tensor:
    """
    混淆矩阵 cm (C,C), cm[i,j] 表示 true=i, pred=j 的计数。
    输入可在任意 device；返回在 CPU。
    """
    # torch.bincount 在不同版本/平台上对 CUDA 的支持不一致；这里统一转 CPU 保守处理
    y_true = y_true.detach().view(-1).long().cpu()
    y_pred = y_pred.detach().view(-1).long().cpu()
    c = int(num_classes)
    idx = (y_true.clamp_min(0).clamp_max(c - 1) * c + y_pred.clamp_min(0).clamp_max(c - 1)).long()
    cm = torch.bincount(idx, minlength=c * c).view(c, c).cpu()
    return cm


def _summarize_array(x: np.ndarray) -> dict:
    x = np.asarray(x).reshape(-1)
    if x.size == 0:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "p05": float(np.quantile(x, 0.05)),
        "p25": float(np.quantile(x, 0.25)),
        "p75": float(np.quantile(x, 0.75)),
        "p95": float(np.quantile(x, 0.95)),
    }


@torch.no_grad()
def _compute_real_low_stats_per_class(
    *,
    teacher: CNNCifar,
    dataset,
    indices: List[int],
    num_classes: int,
    n_real_per_class: int,
    batch_size: int,
    device: str,
    seed: int,
) -> Tuple[Dict[int, dict], Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    """
    从真实图像抽样，计算每类 low_raw 的均值/方差（逐维 diag），并返回每类原型（均值向量）。

    returns:
      - stats: {c: {n, mean_l2, var_mean, var_l2}}
      - prototypes: {c: proto (D,) on CPU}
      - variances: {c: var (D,) on CPU}
    """
    targets = getattr(dataset, "targets", None)
    if targets is None:
        raise AttributeError("train_dataset has no 'targets'; cannot sample per-class real data.")
    if torch.is_tensor(targets):
        targets_t = targets.detach().cpu().long()
    else:
        targets_t = torch.tensor(list(targets), dtype=torch.long)

    by_class: Dict[int, List[int]] = {int(c): [] for c in range(int(num_classes))}
    for i in indices:
        y = int(targets_t[int(i)].item())
        if 0 <= y < int(num_classes):
            by_class[int(y)].append(int(i))

    rng = np.random.default_rng(int(seed))
    chosen: List[int] = []
    for c in range(int(num_classes)):
        pool = by_class[int(c)]
        if not pool:
            continue
        k = min(int(n_real_per_class), len(pool))
        sel = rng.choice(np.asarray(pool, dtype=np.int64), size=int(k), replace=False).tolist()
        chosen.extend(int(x) for x in sel)

    if not chosen:
        return {}, {}

    dl = DataLoader(
        Subset(dataset, chosen),
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    teacher.eval()
    sums: Dict[int, torch.Tensor] = {}
    sumsq: Dict[int, torch.Tensor] = {}
    counts: Dict[int, int] = {}
    feature_dim: Optional[int] = None

    for images, labels in dl:
        images = images.to(device)
        labels = labels.to(device).long()
        _logits, _lp, _h_raw, low_raw, _proj = teacher(images)
        if feature_dim is None:
            feature_dim = int(low_raw.shape[1])
        for c in torch.unique(labels).tolist():
            c = int(c)
            mask = labels == c
            if not mask.any():
                continue
            feats = low_raw[mask].float()
            if c not in sums:
                sums[c] = torch.zeros((int(feature_dim),), device=device, dtype=torch.float32)
                sumsq[c] = torch.zeros((int(feature_dim),), device=device, dtype=torch.float32)
                counts[c] = 0
            sums[c] += feats.sum(dim=0)
            sumsq[c] += (feats * feats).sum(dim=0)
            counts[c] += int(mask.sum().item())

    prototypes: Dict[int, torch.Tensor] = {}
    variances: Dict[int, torch.Tensor] = {}
    stats: Dict[int, dict] = {}
    for c, n in counts.items():
        n = int(n)
        if n <= 0:
            continue
        mu = (sums[int(c)] / float(n)).detach().cpu()
        ex2 = (sumsq[int(c)] / float(n)).detach().cpu()
        var = (ex2 - mu * mu).clamp_min(0.0)
        prototypes[int(c)] = mu
        variances[int(c)] = var
        stats[int(c)] = {
            "n": int(n),
            "mean_l2": float(mu.norm(p=2).item()),
            "var_mean": float(var.mean().item()),
            "var_l2": float(var.norm(p=2).item()),
        }
    return stats, prototypes, variances


@torch.no_grad()
def _compute_synth_moments_per_class(
    *,
    pools: Dict[int, torch.Tensor],
    classes: List[int],
    n_syn_per_class: int,
    device: str,
) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    """
    从合成池按类抽样，计算每类的均值/方差（逐维 diag）。
    returns:
      - mu_syn: {c: (D,) on CPU}
      - var_syn: {c: (D,) on CPU}
    """
    mu_syn: Dict[int, torch.Tensor] = {}
    var_syn: Dict[int, torch.Tensor] = {}
    for c in classes:
        c = int(c)
        pool_c = pools.get(c, None)
        if pool_c is None or pool_c.numel() == 0:
            continue
        n = int(n_syn_per_class)
        y = torch.full((n,), c, dtype=torch.long, device=device)
        x = _sample_from_pool(pools, y, device=device).float()
        mu = x.mean(dim=0)
        ex2 = (x * x).mean(dim=0)
        var = (ex2 - mu * mu).clamp_min(0.0)
        mu_syn[c] = mu.detach().cpu()
        var_syn[c] = var.detach().cpu()
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
    构建用于批量 moment matching 的查表张量（都在 device 上）：
      - mu_s_table: (C,D)
      - mu_r_table: (C,D)
      - scale_table: (C,D) 其中 scale = sqrt((var_r+eps)/(var_s+eps))

    对于缺失类：默认 mu=0, scale=1，使变换退化为 identity。
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
            v_r = var_real[kk]
            # clamp 防止极小方差导致爆炸
            vv_s = v_s.float().to(device).clamp_min(0.0)
            vv_r = v_r.float().to(device).clamp_min(0.0)
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
    """
    x: (B,D) on device
    y: (B,) int64 on device
    """
    y = y.long()
    mu_s = mu_s_table[y]
    mu_r = mu_r_table[y]
    sc = scale_table[y]
    return (x - mu_s) * sc + mu_r


@torch.no_grad()
def _compare_synth_to_real_prototypes(
    *,
    pools: Dict[int, torch.Tensor],
    prototypes: Dict[int, torch.Tensor],
    num_classes: int,
    n_syn_per_class: int,
    device: str,
) -> dict:
    """
    对每类从池中抽样合成特征，计算：
      - 合成按类均值/方差（逐维 diag）
      - 合成特征与真实原型（同类）的余弦相似度分布
      - “最近原型”分类准确率（用真实原型当类中心）
    """
    classes = sorted([int(c) for c in prototypes.keys() if int(c) in pools and pools[int(c)].numel() > 0])
    if not classes:
        return {"classes_used": [], "per_class": {}, "proto_cosine_all": {}, "proto_nn_acc": 0.0}

    protos = torch.stack([prototypes[int(c)] for c in classes], dim=0).float().to(device)  # (K,D)
    protos_n = F.normalize(protos, dim=1)
    class_to_k = {int(c): int(i) for i, c in enumerate(classes)}

    per_class: Dict[int, dict] = {}
    cos_all: List[float] = []
    correct_nn = 0
    total_nn = 0

    for c in classes:
        n = int(n_syn_per_class)
        y = torch.full((n,), int(c), dtype=torch.long, device=device)
        x = _sample_from_pool(pools, y, device=device).float()

        mu = x.mean(dim=0)
        ex2 = (x * x).mean(dim=0)
        var = (ex2 - mu * mu).clamp_min(0.0)

        k = class_to_k[int(c)]
        proto_y_n = protos_n[k : k + 1]  # (1,D)
        x_n = F.normalize(x, dim=1)
        cos = torch.matmul(x_n, proto_y_n.t()).view(-1)
        cos_cpu = cos.detach().cpu().numpy()
        cos_all.extend(float(v) for v in cos_cpu.tolist())

        sims = torch.matmul(x_n, protos_n.t())  # (n,K)
        pred_k = sims.argmax(dim=1)
        correct_nn += int((pred_k == int(k)).sum().item())
        total_nn += int(pred_k.numel())

        per_class[int(c)] = {
            "n": int(n),
            "mean_l2": float(mu.norm(p=2).item()),
            "var_mean": float(var.mean().item()),
            "var_l2": float(var.norm(p=2).item()),
            "cos_to_real_proto": _summarize_array(cos_cpu),
        }

    return {
        "classes_used": [int(c) for c in classes],
        "per_class": per_class,
        "proto_cosine_all": _summarize_array(np.asarray(cos_all, dtype=np.float32)),
        "proto_nn_acc": float(correct_nn / total_nn) if total_nn > 0 else 0.0,
    }


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
) -> Dict[int, torch.Tensor]:
    pool_path_abs = os.path.abspath(pool_path)
    pool_path_io = _maybe_win_long_path(pool_path_abs)

    if int(rebuild_pool) != 1 and os.path.exists(pool_path_io):
        obj = torch.load(pool_path_io, map_location="cpu")
        pools = obj.get("pools", None) if isinstance(obj, dict) else None
        if not isinstance(pools, dict):
            raise ValueError("Invalid pool file: missing dict key 'pools'")
        return {int(k): v.float() for k, v in pools.items()}

    os.makedirs(_maybe_win_long_path(os.path.dirname(pool_path_abs)), exist_ok=True)

    num_classes = int(cvae.cfg.num_classes)
    pools: Dict[int, torch.Tensor] = {}
    qset = set(int(x) for x in qualified_classes)
    for c in range(num_classes):
        n = int(global_counts[c].item())
        if c not in qset:
            pools[int(c)] = torch.empty((0, int(cvae.cfg.feature_dim)), dtype=torch.float32)
            continue
        m = int(round(float(pool_ratio) * float(n)))
        m = max(int(pool_min), min(int(pool_max), m))
        labels = torch.full((m,), int(c), dtype=torch.long)
        feats = generate_features(cvae, labels, num_per_label=1, temperature=float(temperature), device=device)
        pools[int(c)] = feats.float().contiguous()

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
    return pools


def _sample_from_pool(pools: Dict[int, torch.Tensor], y: torch.Tensor, *, device: str) -> torch.Tensor:
    """
    pools[c]: (M_c, D) on CPU; y: (B,) on device.
    returns x: (B, D) on device.
    """
    y_cpu = y.detach().cpu().long()
    out_chunks: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for c in torch.unique(y_cpu).tolist():
        c = int(c)
        mask = y_cpu == c
        idxs = torch.nonzero(mask, as_tuple=False).view(-1)
        pool_c = pools.get(c, None)
        if pool_c is None or pool_c.numel() == 0:
            raise ValueError(f"Pool for class {c} is empty; adjust label sampling or rebuild pool.")
        m = int(pool_c.shape[0])
        ridx = torch.randint(low=0, high=m, size=(int(idxs.numel()),), dtype=torch.long)
        x_sel = pool_c[ridx].to(device)
        out_chunks.append((idxs, x_sel))

    d = int(next(iter(pools.values())).shape[1])
    x = torch.empty((int(y_cpu.numel()), d), device=device, dtype=torch.float32)
    for idxs, x_sel in out_chunks:
        x[idxs.to(device)] = x_sel
    return x


@torch.no_grad()
def _eval_teacher_on_synth(
    *,
    teacher: CNNCifar,
    pools: Dict[int, torch.Tensor],
    qualified_classes: List[int],
    present_classes: List[int],
    label_sampling: str,
    n_syn: int,
    batch_size: int,
    device: str,
    num_classes: int,
    collect_confusion: bool = True,
    moment_match: Optional[dict] = None,
) -> dict:
    teacher.eval()
    if int(n_syn) <= 0:
        out = {"acc": 0.0, "mean_conf": 0.0, "mean_entropy": 0.0, "per_class_acc": {}, "n_syn": int(n_syn)}
        if collect_confusion:
            out.update(
                {
                    "confusion_matrix": [[0 for _ in range(int(num_classes))] for _ in range(int(num_classes))],
                    "pred_marginal": [0 for _ in range(int(num_classes))],
                    "true_marginal": [0 for _ in range(int(num_classes))],
                }
            )
        return out

    sampling = str(label_sampling).lower()
    q = torch.tensor([int(x) for x in qualified_classes], dtype=torch.long, device=device)

    if sampling == "client_classes":
        if present_classes:
            present_t = torch.tensor([int(x) for x in present_classes], dtype=torch.long, device=device)
        else:
            present_t = q
        label_space = present_t
    elif sampling == "global_uniform":
        label_space = q
    else:
        raise ValueError(f"Unsupported label_sampling: {label_sampling}")

    correct = 0
    total = 0
    conf_sum = 0.0
    ent_sum = 0.0
    per_total: Counter = Counter()
    per_correct: Counter = Counter()
    cm = torch.zeros((int(num_classes), int(num_classes)), dtype=torch.long) if collect_confusion else None
    pred_counts = torch.zeros((int(num_classes),), dtype=torch.long) if collect_confusion else None
    true_counts = torch.zeros((int(num_classes),), dtype=torch.long) if collect_confusion else None

    remaining = int(n_syn)
    eps = 1e-12
    while remaining > 0:
        b = min(int(batch_size), remaining)
        remaining -= b

        ridx = torch.randint(low=0, high=int(label_space.numel()), size=(b,), device=device)
        y_syn = label_space[ridx].long()
        x_syn = _sample_from_pool(pools, y_syn, device=device)
        if moment_match is not None:
            x_syn = _apply_moment_match(
                x_syn,
                y_syn,
                mu_s_table=moment_match["mu_s_table"],
                mu_r_table=moment_match["mu_r_table"],
                scale_table=moment_match["scale_table"],
            )

        logits, _lp, _h_raw, _proj = teacher.forward_from_low(x_syn)
        logits = logits[:, 0:num_classes]
        probs = torch.softmax(logits, dim=1)
        pred = probs.argmax(dim=1)

        correct_b = int((pred == y_syn).sum().item())
        correct += correct_b
        total += int(y_syn.numel())

        maxp = probs.max(dim=1).values
        conf_sum += float(maxp.sum().item())

        ent = -(probs * torch.log(probs.clamp_min(eps))).sum(dim=1)
        ent_sum += float(ent.sum().item())

        ys = y_syn.detach().cpu().tolist()
        ps = pred.detach().cpu().tolist()
        for yt, pt in zip(ys, ps):
            per_total[int(yt)] += 1
            if int(yt) == int(pt):
                per_correct[int(yt)] += 1
        if collect_confusion and cm is not None and pred_counts is not None and true_counts is not None:
            cm += _bincount_confusion(y_syn, pred, num_classes=int(num_classes))
            pred_counts += torch.bincount(pred.detach().cpu().long(), minlength=int(num_classes))
            true_counts += torch.bincount(y_syn.detach().cpu().long(), minlength=int(num_classes))

    acc = (correct / total) if total > 0 else 0.0
    mean_conf = (conf_sum / total) if total > 0 else 0.0
    mean_entropy = (ent_sum / total) if total > 0 else 0.0
    per_class_acc = {int(c): (per_correct[int(c)] / per_total[int(c)]) for c in per_total.keys()}

    out = {
        "acc": float(acc),
        "mean_conf": float(mean_conf),
        "mean_entropy": float(mean_entropy),
        "per_class_acc": {int(k): float(v) for k, v in per_class_acc.items()},
        "n_syn": int(n_syn),
    }
    if collect_confusion and cm is not None and pred_counts is not None and true_counts is not None:
        out.update(
            {
                "confusion_matrix": cm.tolist(),
                "pred_marginal": pred_counts.tolist(),
                "true_marginal": true_counts.tolist(),
            }
        )
    return out


def _make_default_out_dir() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(_ROOT, "analysis", "teacher_on_synth", ts)


def _summarize(values: List[float]) -> dict:
    if not values:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(mean(values)),
        "median": float(median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def main() -> int:
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("--stage1_ckpt_path", type=str, required=True, help="Stage-1 checkpoint path (best-wo.pt / latest.pt)")
    p.add_argument("--cvae_path", type=str, required=True, help="Stage-3 cVAE generator.pt path (or its directory)")
    p.add_argument("--split_path", type=str, default=None, help="split.pkl path (if None, infer from ckpt meta/args)")

    p.add_argument("--out_dir", type=str, default=None, help="Output dir (default: analysis/teacher_on_synth/<timestamp>)")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=None)

    p.add_argument("--label_sampling", type=str, default="both", choices=["both", "global_uniform", "client_classes"])
    p.add_argument("--n_syn", type=int, default=10000, help="Number of synthetic samples per client per sampling strategy")
    p.add_argument("--batch_size", type=int, default=1024, help="Eval batch size on synthetic features")

    # Diagnostics (先定位：到底坏在哪)
    p.add_argument("--diag_confusion", type=int, default=1, choices=[0, 1], help="Collect y vs yhat confusion & yhat marginal")
    p.add_argument("--diag_real_compare", type=int, default=1, choices=[0, 1], help="Compare real low_raw vs synth by class")
    p.add_argument("--diag_save_plots", type=int, default=1, choices=[0, 1], help="If matplotlib is available, save png plots")
    p.add_argument("--diag_n_real_per_class", type=int, default=200, help="Real samples per class per client for low_raw stats")
    p.add_argument("--diag_n_syn_per_class", type=int, default=1000, help="Synthetic samples per class per client for stats")
    p.add_argument("--diag_moment_match", type=int, default=0, choices=[0, 1], help="Probe A: per-class moment matching before teacher eval")
    p.add_argument("--diag_moment_eps", type=float, default=1e-6, help="Epsilon for moment matching scale stability")

    # Pool building knobs (aligned with stage4)
    p.add_argument("--pool_ratio", type=float, default=1.0)
    p.add_argument("--pool_min", type=int, default=200)
    p.add_argument("--pool_max", type=int, default=5000)
    p.add_argument("--pool_temperature", type=float, default=1.0)
    p.add_argument("--pool_path", type=str, default=None, help="Path to cache the feature pool (default: <out_dir>/feature_pool.pt)")
    p.add_argument("--rebuild_pool", type=int, default=0, choices=[0, 1])
    p.add_argument("--cvae_y_emb_dim", type=int, default=32)

    # Threshold for conclusion printing
    p.add_argument("--warn_thr_low", type=float, default=0.8)
    p.add_argument("--warn_thr_high", type=float, default=0.9)

    args = p.parse_args()
    if args.seed is None:
        args.seed = int(time.time_ns() % (2**31 - 1))

    device = _resolve_device(args.gpu)
    _seed_all(int(args.seed))

    out_dir = os.path.abspath(args.out_dir or _make_default_out_dir())
    os.makedirs(_maybe_win_long_path(out_dir), exist_ok=True)

    ckpt = load_checkpoint(args.stage1_ckpt_path, map_location="cpu")
    ckpt_meta = (ckpt.get("meta", {}) or {}) if isinstance(ckpt, dict) else {}
    ckpt_state = (ckpt.get("state", {}) or {}) if isinstance(ckpt, dict) else {}
    ckpt_args = (ckpt.get("args", {}) or {}) if isinstance(ckpt, dict) else {}

    dataset = str(ckpt_meta.get("dataset", ckpt_args.get("dataset", "cifar10"))).lower()
    if dataset != "cifar10":
        raise ValueError(f"This verifier currently supports cifar10 only. Got dataset={dataset}")
    num_users = int(ckpt_meta.get("num_users", ckpt_args.get("num_users", 20)))
    num_classes = int(ckpt_meta.get("num_classes", ckpt_args.get("num_classes", 10)))

    split_path = args.split_path or ckpt_meta.get("split_path", None) or ckpt_args.get("split_path", None)
    if not split_path or not os.path.exists(_maybe_win_long_path(split_path)):
        raise FileNotFoundError(f"split_path not found: {split_path}")
    split = load_split(split_path)
    n_list = split["n_list"]
    k_list = split["k_list"]
    user_groups: Dict[int, List[int]] = split["user_groups"]

    # Build args namespace for get_dataset (reuse ckpt args as much as possible)
    base = dict(ckpt_args)
    base["dataset"] = "cifar10"
    base["num_classes"] = num_classes
    base["num_users"] = num_users
    base["gpu"] = int(args.gpu)
    base["device"] = device
    args_ds = SimpleNamespace(**base)
    train_dataset, _test_dataset, _, _, _, _ = get_dataset(args_ds, n_list, k_list)

    # Load cVAE
    cvae_path = _resolve_cvae_path(args.cvae_path)
    cvae, cvae_meta = _load_cvae(cvae_path, device=device, y_emb_dim=int(args.cvae_y_emb_dim))
    if int(cvae.cfg.num_classes) != int(num_classes):
        raise RuntimeError(f"num_classes mismatch: stage1={num_classes} vs cvae={int(cvae.cfg.num_classes)}")

    # Compute global counts and qualified classes
    global_counts = _compute_global_class_counts(train_dataset, user_groups=user_groups, num_classes=num_classes)
    qualified = [int(i) for i in torch.nonzero(global_counts > 0, as_tuple=False).view(-1).tolist()]
    if len(qualified) == 0:
        raise ValueError("No qualified classes found (global_counts all zero).")

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
    )

    # Per-client present classes (for client_classes sampling)
    client_present: Dict[int, List[int]] = {}
    classes_list = split.get("classes_list", None)
    qualified_set = set(int(x) for x in qualified)
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

    local_sd = ckpt_state.get("local_models_full_state_dicts", None)
    if not isinstance(local_sd, dict):
        raise ValueError("Stage-1 checkpoint missing state['local_models_full_state_dicts']")

    modes: List[str]
    if args.label_sampling == "both":
        modes = ["global_uniform", "client_classes"]
    else:
        modes = [str(args.label_sampling)]

    results: Dict[str, List[dict]] = {m: [] for m in modes}
    real_vs_synth_all: List[dict] = []
    print(f"[info] out_dir={out_dir}")
    print(f"[info] device={device} seed={args.seed} n_users={num_users} n_syn={int(args.n_syn)} modes={modes}")
    print(f"[info] cvae_path={cvae_path}")
    print(f"[info] pool_path={pool_path} rebuild_pool={int(args.rebuild_pool)}")
    diag_dir = os.path.join(out_dir, "diagnostics")
    os.makedirs(_maybe_win_long_path(diag_dir), exist_ok=True)
    plt = _try_import_matplotlib() if int(args.diag_save_plots) == 1 else None

    t0 = time.time()
    for cid in range(int(num_users)):
        teacher = CNNCifar(args=SimpleNamespace(**{"num_classes": num_classes})).to(device)
        key = cid if cid in local_sd else str(cid)
        if key not in local_sd:
            raise KeyError(f"local_models_full_state_dicts missing client id={cid}")
        teacher.load_state_dict(local_sd[key], strict=True)

        per_client = {"client_id": int(cid), "present_classes": client_present.get(int(cid), [])}

        # 诊断 2：真实 low_raw vs 合成特征对齐（每 client 做一次）
        mm_tables = None
        if int(args.diag_real_compare) == 1 or int(args.diag_moment_match) == 1:
            train_idxs = _as_int_indices(user_groups[int(cid)])
            real_stats, prototypes, real_vars = _compute_real_low_stats_per_class(
                teacher=teacher,
                dataset=train_dataset,
                indices=train_idxs,
                num_classes=int(num_classes),
                n_real_per_class=int(args.diag_n_real_per_class),
                batch_size=min(int(args.batch_size), 512),
                device=device,
                seed=int(args.seed) + 1000 + int(cid),
            )
            # 仅在 diag_real_compare=1 时写出对比统计
            if int(args.diag_real_compare) == 1:
                synth_vs_real = _compare_synth_to_real_prototypes(
                    pools=pools,
                    prototypes=prototypes,
                    num_classes=int(num_classes),
                    n_syn_per_class=int(args.diag_n_syn_per_class),
                    device=device,
                )
                per_client["real_low_stats"] = real_stats
                per_client["synth_vs_real"] = synth_vs_real
                real_vs_synth_all.append(
                    {
                        "client_id": int(cid),
                        "present_classes": client_present.get(int(cid), []),
                        "real_low_stats": real_stats,
                        "synth_vs_real": synth_vs_real,
                    }
                )
                out_cmp = os.path.join(diag_dir, f"client_{cid:02d}_real_vs_synth.json")
                with open(_maybe_win_long_path(out_cmp), "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "client_id": int(cid),
                            "present_classes": client_present.get(int(cid), []),
                            "real_low_stats": real_stats,
                            "synth_vs_real": synth_vs_real,
                        },
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )

            # 探针 A：moment matching（按类均值/方差对齐），用于判断 shift 是否为主因
            if int(args.diag_moment_match) == 1 and len(prototypes) > 0 and len(real_vars) > 0:
                mm_classes = sorted([int(c) for c in prototypes.keys() if int(c) in real_vars])
                mu_syn, var_syn = _compute_synth_moments_per_class(
                    pools=pools,
                    classes=mm_classes,
                    n_syn_per_class=int(args.diag_n_syn_per_class),
                    device=device,
                )
                if len(mu_syn) > 0 and len(var_syn) > 0:
                    feat_dim = int(next(iter(prototypes.values())).numel())
                    mu_s_table, mu_r_table, scale_table = _build_moment_match_tables(
                        num_classes=int(num_classes),
                        feature_dim=int(feat_dim),
                        mu_real=prototypes,
                        var_real=real_vars,
                        mu_syn=mu_syn,
                        var_syn=var_syn,
                        eps=float(args.diag_moment_eps),
                        device=device,
                    )
                    mm_tables = {"mu_s_table": mu_s_table, "mu_r_table": mu_r_table, "scale_table": scale_table}

        for m in modes:
            stats = _eval_teacher_on_synth(
                teacher=teacher,
                pools=pools,
                qualified_classes=qualified,
                present_classes=client_present.get(int(cid), []),
                label_sampling=m,
                n_syn=int(args.n_syn),
                batch_size=int(args.batch_size),
                device=device,
                num_classes=num_classes,
                collect_confusion=(int(args.diag_confusion) == 1),
            )
            # 同一套采样下再跑一遍 moment-matched 的合成特征
            if int(args.diag_moment_match) == 1 and mm_tables is not None:
                stats_mm = _eval_teacher_on_synth(
                    teacher=teacher,
                    pools=pools,
                    qualified_classes=qualified,
                    present_classes=client_present.get(int(cid), []),
                    label_sampling=m,
                    n_syn=int(args.n_syn),
                    batch_size=int(args.batch_size),
                    device=device,
                    num_classes=num_classes,
                    collect_confusion=(int(args.diag_confusion) == 1),
                    moment_match=mm_tables,
                )
                stats["moment_match"] = stats_mm
            per_client[m] = stats
            results[m].append({"client_id": int(cid), **stats})

            # 诊断 1：y vs yhat 混淆矩阵 + yhat 边际分布
            if int(args.diag_confusion) == 1:
                cm = stats.get("confusion_matrix", None)
                pm = stats.get("pred_marginal", None)
                if isinstance(cm, list) and isinstance(pm, list):
                    out_cm = os.path.join(diag_dir, f"client_{cid:02d}_{m}_confusion.json")
                    with open(_maybe_win_long_path(out_cm), "w", encoding="utf-8") as f:
                        json.dump(
                            {
                                "client_id": int(cid),
                                "mode": str(m),
                                "present_classes": client_present.get(int(cid), []),
                                "confusion_matrix": cm,
                                "pred_marginal": pm,
                                "true_marginal": stats.get("true_marginal", None),
                            },
                            f,
                            ensure_ascii=False,
                            indent=2,
                        )

                    if plt is not None:
                        try:
                            fig = plt.figure(figsize=(6, 5))
                            ax = fig.add_subplot(111)
                            arr = np.asarray(cm, dtype=np.float32)
                            im = ax.imshow(arr, interpolation="nearest", cmap="Blues")
                            ax.set_title(f"client {cid:02d} | {m} | y vs yhat")
                            ax.set_xlabel("yhat")
                            ax.set_ylabel("y")
                            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                            fig.tight_layout()
                            fig.savefig(
                                _maybe_win_long_path(os.path.join(diag_dir, f"client_{cid:02d}_{m}_confusion.png")),
                                dpi=160,
                            )
                            plt.close(fig)

                            fig2 = plt.figure(figsize=(7, 3))
                            ax2 = fig2.add_subplot(111)
                            x = np.arange(int(num_classes))
                            ax2.bar(x, np.asarray(pm, dtype=np.int64))
                            ax2.set_title(f"client {cid:02d} | {m} | yhat marginal")
                            ax2.set_xlabel("class")
                            ax2.set_ylabel("count")
                            fig2.tight_layout()
                            fig2.savefig(
                                _maybe_win_long_path(os.path.join(diag_dir, f"client_{cid:02d}_{m}_pred_marginal.png")),
                                dpi=160,
                            )
                            plt.close(fig2)
                        except Exception:
                            pass

                # moment-match 的混淆矩阵也单独落盘（如存在）
                mm = stats.get("moment_match", None)
                if isinstance(mm, dict):
                    cm2 = mm.get("confusion_matrix", None)
                    pm2 = mm.get("pred_marginal", None)
                    if isinstance(cm2, list) and isinstance(pm2, list):
                        out_cm2 = os.path.join(diag_dir, f"client_{cid:02d}_{m}_moment_match_confusion.json")
                        with open(_maybe_win_long_path(out_cm2), "w", encoding="utf-8") as f:
                            json.dump(
                                {
                                    "client_id": int(cid),
                                    "mode": str(m),
                                    "variant": "moment_match",
                                    "present_classes": client_present.get(int(cid), []),
                                    "confusion_matrix": cm2,
                                    "pred_marginal": pm2,
                                    "true_marginal": mm.get("true_marginal", None),
                                },
                                f,
                                ensure_ascii=False,
                                indent=2,
                            )

        # compact progress line
        msg_parts = [f"[cid={cid:02d}]"]
        for m in modes:
            msg_parts.append(f"{m}: acc={per_client[m]['acc']:.4f} conf={per_client[m]['mean_conf']:.4f} ent={per_client[m]['mean_entropy']:.4f}")
        print(" ".join(msg_parts))

        teacher.cpu()
        del teacher

    dt = time.time() - t0

    summary: dict = {
        "meta": {
            "script": "analysis/verify_teacher_on_synth.py",
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "out_dir": out_dir,
            "stage1_ckpt_path": os.path.abspath(args.stage1_ckpt_path),
            "split_path": os.path.abspath(split_path),
            "cvae_path": os.path.abspath(cvae_path),
            "cvae_meta_path": os.path.join(os.path.dirname(os.path.abspath(cvae_path)), "generator_meta.json"),
            "cvae_meta": cvae_meta,
            "pool_cfg": {
                "pool_ratio": float(args.pool_ratio),
                "pool_min": int(args.pool_min),
                "pool_max": int(args.pool_max),
                "pool_temperature": float(args.pool_temperature),
                "rebuild_pool": int(args.rebuild_pool),
            },
            "eval_cfg": {
                "label_sampling": str(args.label_sampling),
                "modes": modes,
                "n_syn": int(args.n_syn),
                "batch_size": int(args.batch_size),
                "seed": int(args.seed),
                "device": device,
            },
            "dataset": dataset,
            "num_users": int(num_users),
            "num_classes": int(num_classes),
            "qualified_classes": [int(x) for x in qualified],
            "global_counts": global_counts.tolist(),
            "elapsed_sec": float(dt),
        },
        "by_mode": {},
        "diagnostics_cfg": {
            "diag_confusion": int(args.diag_confusion),
            "diag_real_compare": int(args.diag_real_compare),
            "diag_save_plots": int(args.diag_save_plots),
            "diag_n_real_per_class": int(args.diag_n_real_per_class),
            "diag_n_syn_per_class": int(args.diag_n_syn_per_class),
            "diag_moment_match": int(args.diag_moment_match),
            "diag_moment_eps": float(args.diag_moment_eps),
        },
        "diagnostics": {
            "real_vs_synth": real_vs_synth_all,
        },
    }

    for m in modes:
        accs = [float(r["acc"]) for r in results[m]]
        confs = [float(r["mean_conf"]) for r in results[m]]
        ents = [float(r["mean_entropy"]) for r in results[m]]
        mm_accs = [float(r.get("moment_match", {}).get("acc")) for r in results[m] if isinstance(r.get("moment_match", None), dict)]
        mm_confs = [float(r.get("moment_match", {}).get("mean_conf")) for r in results[m] if isinstance(r.get("moment_match", None), dict)]
        mm_ents = [float(r.get("moment_match", {}).get("mean_entropy")) for r in results[m] if isinstance(r.get("moment_match", None), dict)]
        low_thr = float(args.warn_thr_low)
        high_thr = float(args.warn_thr_high)
        below_low = [i for i, a in enumerate(accs) if a < low_thr]
        below_high = [i for i, a in enumerate(accs) if a < high_thr]
        summary["by_mode"][m] = {
            "acc": _summarize(accs),
            "mean_conf": _summarize(confs),
            "mean_entropy": _summarize(ents),
            "moment_match": {
                "acc": _summarize(mm_accs),
                "mean_conf": _summarize(mm_confs),
                "mean_entropy": _summarize(mm_ents),
            }
            if mm_accs
            else None,
            "clients_below_thr": {
                "thr_low": low_thr,
                "thr_high": high_thr,
                "below_low": [int(x) for x in below_low],
                "below_high": [int(x) for x in below_high],
                "frac_below_low": float(len(below_low) / len(accs)) if accs else 0.0,
                "frac_below_high": float(len(below_high) / len(accs)) if accs else 0.0,
            },
            "per_client": results[m],
        }

    out_json = os.path.join(out_dir, "summary.json")
    with open(_maybe_win_long_path(out_json), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[done] wrote: {out_json}")
    for m in modes:
        s = summary["by_mode"][m]
        acc_mean = s["acc"]["mean"]
        frac_low = s["clients_below_thr"]["frac_below_low"]
        frac_high = s["clients_below_thr"]["frac_below_high"]
        print(f"[summary][{m}] mean_acc={acc_mean:.4f} frac(acc<{args.warn_thr_low})={frac_low:.2%} frac(acc<{args.warn_thr_high})={frac_high:.2%}")

    # conclusion hint
    for m in modes:
        s = summary["by_mode"][m]
        if float(s["acc"]["mean"]) < float(args.warn_thr_high):
            print(
                f"[hint][{m}] mean teacher-acc {float(s['acc']['mean']):.4f} < {float(args.warn_thr_high):.2f}; "
                "若大量客户端/类别也低于 0.80~0.90，Stage-3 需要语义约束。"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


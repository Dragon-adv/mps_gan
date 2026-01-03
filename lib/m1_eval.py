#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6+

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def set_all_seeds(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def per_class_correct_total(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    num_classes: int,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    返回 (correct, total)，shape 都是 (num_classes,)。
    """
    model.eval()
    correct = torch.zeros(num_classes, dtype=torch.long)
    total = torch.zeros(num_classes, dtype=torch.long)

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device).long()

        # model returns: logits, log_probs, high_raw, low_raw, projected
        out = model(images)
        if isinstance(out, (list, tuple)) and len(out) >= 2:
            log_probs = out[1]
        else:
            raise ValueError("Unexpected model output; expect (logits, log_probs, ...)")

        preds = torch.argmax(log_probs, dim=1)

        total += torch.bincount(labels, minlength=num_classes).cpu()
        hit = preds.eq(labels)
        if hit.any():
            correct += torch.bincount(labels[hit], minlength=num_classes).cpu()

    return correct, total


@torch.no_grad()
def extract_low_by_class(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    num_classes: int,
    device: str,
    max_per_class: Optional[int] = None,
) -> List[torch.Tensor]:
    """
    从 full forward 中提取 low_level_features_raw，按类返回 list[tensor]。
    """
    model.eval()
    buckets: List[List[torch.Tensor]] = [[] for _ in range(num_classes)]
    counts = [0 for _ in range(num_classes)]

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device).long()
        logits, log_probs, high_raw, low_raw, projected = model(images)
        low_raw = low_raw.detach().cpu()
        labels_cpu = labels.detach().cpu()

        for i in range(low_raw.shape[0]):
            c = int(labels_cpu[i].item())
            if max_per_class is not None and counts[c] >= max_per_class:
                continue
            buckets[c].append(low_raw[i].view(1, -1))
            counts[c] += 1

        if max_per_class is not None and all(ct >= max_per_class for ct in counts):
            break

    out: List[torch.Tensor] = []
    for c in range(num_classes):
        if len(buckets[c]) == 0:
            out.append(torch.empty(0, 0))
        else:
            out.append(torch.cat(buckets[c], dim=0))
    return out


@torch.no_grad()
def extract_high_by_class(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    num_classes: int,
    device: str,
    max_per_class: Optional[int] = None,
    use_norm: bool = True,
) -> List[torch.Tensor]:
    """
    从 full forward 中提取 high_level_features_raw（或其 normalize），按类返回 list[tensor]。
    """
    model.eval()
    buckets: List[List[torch.Tensor]] = [[] for _ in range(num_classes)]
    counts = [0 for _ in range(num_classes)]

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device).long()
        logits, log_probs, high_raw, low_raw, projected = model(images)
        h = high_raw
        if use_norm:
            h = F.normalize(h, dim=1)
        h = h.detach().cpu()
        labels_cpu = labels.detach().cpu()

        for i in range(h.shape[0]):
            c = int(labels_cpu[i].item())
            if max_per_class is not None and counts[c] >= max_per_class:
                continue
            buckets[c].append(h[i].view(1, -1))
            counts[c] += 1

        if max_per_class is not None and all(ct >= max_per_class for ct in counts):
            break

    out: List[torch.Tensor] = []
    for c in range(num_classes):
        if len(buckets[c]) == 0:
            out.append(torch.empty(0, 0))
        else:
            out.append(torch.cat(buckets[c], dim=0))
    return out


def build_teacher_map(
    correct_mat: torch.Tensor,
    total_mat: torch.Tensor,
    k_teachers: int,
    n_real_min: int,
) -> Dict[int, List[int]]:
    """
    correct_mat/total_mat: shape (num_clients, num_classes)
    """
    num_clients, num_classes = total_mat.shape
    teacher_map: Dict[int, List[int]] = {}
    for c in range(num_classes):
        totals = total_mat[:, c]
        mask = totals >= n_real_min
        if mask.sum().item() == 0:
            # fallback: allow all clients
            mask = totals >= 0
        acc = torch.zeros(num_clients, dtype=torch.float32)
        acc[mask] = (correct_mat[mask, c].float() / (totals[mask].float().clamp_min(1.0)))
        # sort desc
        order = torch.argsort(acc, descending=True)
        picked = []
        for idx in order.tolist():
            if mask[idx].item():
                picked.append(int(idx))
            if len(picked) >= k_teachers:
                break
        teacher_map[c] = picked
    return teacher_map


def compute_n_syn(
    n_real: int,
    n_syn_min: int,
    n_syn_max: int,
    r: float,
) -> int:
    base = int(round(float(r) * float(n_real)))
    return int(min(n_syn_max, max(n_syn_min, base)))


@dataclass
class GaussianDiag:
    mean: torch.Tensor  # (D,)
    std: torch.Tensor   # (D,)

    def sample(self, n: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        eps = torch.randn((n, self.mean.numel()), generator=generator, dtype=self.mean.dtype)
        return self.mean.view(1, -1) + eps * self.std.view(1, -1)


def fit_gaussian_diag(x: torch.Tensor, eps: float = 1e-6) -> GaussianDiag:
    """
    x: (N, D)
    """
    if x.ndim != 2:
        raise ValueError(f"Expected 2D tensor, got {tuple(x.shape)}")
    mean = x.mean(dim=0)
    # std could be zero for some dims; add eps floor
    std = x.std(dim=0, unbiased=False).clamp_min(eps)
    return GaussianDiag(mean=mean, std=std)


@torch.no_grad()
def forward_from_low_cnncifar(
    model: torch.nn.Module,
    low_flat: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    对 CNNCifar：low_flat (B, 400) -> logits (B, C), high_norm (B, 120)。
    仅用于离线评估/合成，不走 conv。
    """
    # CNNCifar: high_raw = relu(fc0(low)); high_norm = normalize(high_raw)
    high_raw = F.relu(model.fc0(low_flat))
    high_norm = F.normalize(high_raw, dim=1)
    feat = F.relu(model.fc1(high_norm))
    logits = model.fc2(feat)
    return logits, high_norm


def save_json(path: str, obj: object) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)



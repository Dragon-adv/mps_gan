"""
Stats-conditioned low-level feature generator for FedMPS.

Goal:
  Given a label y and global low-level statistics (mu, cov, rf_mean) from Stage-2,
  generate synthetic low-level features (vector) for downstream client fine-tuning.

This is inspired by FedGen's conditional generator, but adapted to FedMPS:
  - output is low-level feature vector (low_level_features_raw)
  - conditioning includes global statistics (mu/diag(cov) and optionally RFF mean as loss target)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiversityLoss(nn.Module):
    """
    Diversity loss to discourage mode collapse.
    Ported from FedGen's DiversityLoss with minimal changes.
    """

    def __init__(self, metric: str = "l1"):
        super().__init__()
        self.metric = metric
        self.cosine = nn.CosineSimilarity(dim=2)

    def _compute_distance(self, tensor1: torch.Tensor, tensor2: torch.Tensor, metric: str) -> torch.Tensor:
        if metric == "l1":
            return torch.abs(tensor1 - tensor2).mean(dim=2)
        if metric == "l2":
            return torch.pow(tensor1 - tensor2, 2).mean(dim=2)
        if metric == "cosine":
            return 1 - self.cosine(tensor1, tensor2)
        raise ValueError(metric)

    def _pairwise_distance(self, tensor: torch.Tensor, how: str) -> torch.Tensor:
        n = tensor.size(0)
        t1 = tensor.expand((n, n, tensor.size(1)))
        t2 = tensor.unsqueeze(dim=1)
        return self._compute_distance(t1, t2, how)

    def forward(self, noises: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        if features.dim() > 2:
            features = features.view(features.size(0), -1)
        layer_dist = self._pairwise_distance(features, how=self.metric)
        noise_dist = self._pairwise_distance(noises, how="l2")
        return torch.exp(torch.mean(-noise_dist * layer_dist))


@dataclass(frozen=True)
class LowStatsTensors:
    """
    Stacked tensors for Stage-2 low-level global statistics.
    Shapes:
      - mu: (C, d)
      - cov_diag: (C, d)
      - rf_mean: (C, D_rf)
      - sample_per_class: (C,)
    """

    mu: torch.Tensor
    cov_diag: torch.Tensor
    rf_mean: torch.Tensor
    sample_per_class: torch.Tensor


def stack_low_global_stats(global_stats: Dict) -> LowStatsTensors:
    """
    Convert Stage-2 payload['state']['global_stats'] (low) into stacked tensors.
    """
    if "low" not in global_stats:
        raise KeyError("global_stats missing 'low' level.")
    low = global_stats["low"]

    mu_list = low["class_means"]
    cov_list = low["class_covs"]
    rf_list = low["class_rf_means"]
    spc = global_stats["sample_per_class"]

    mu = torch.stack([m.detach().clone() for m in mu_list], dim=0)
    cov_diag = torch.stack([c.detach().clone().diagonal() for c in cov_list], dim=0)
    rf_mean = torch.stack([r.detach().clone() for r in rf_list], dim=0)
    sample_per_class = spc.detach().clone()
    return LowStatsTensors(mu=mu, cov_diag=cov_diag, rf_mean=rf_mean, sample_per_class=sample_per_class)


class StatsConditionedFeatureGenerator(nn.Module):
    """
    Conditional generator:
      x_low ~ G(z, y, stats(y))
    where stats(y) includes at least mu_y and diag(cov_y).
    """

    def __init__(
        self,
        *,
        num_classes: int,
        feature_dim: int,
        noise_dim: int = 64,
        y_emb_dim: int = 32,
        stat_emb_dim: int = 128,
        hidden_dim: int = 256,
        n_hidden_layers: int = 2,
        relu_output: bool = True,
        use_cov_diag: bool = True,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.feature_dim = int(feature_dim)
        self.noise_dim = int(noise_dim)
        self.y_emb_dim = int(y_emb_dim)
        self.stat_emb_dim = int(stat_emb_dim)
        self.hidden_dim = int(hidden_dim)
        self.n_hidden_layers = int(n_hidden_layers)
        self.relu_output = bool(relu_output)
        self.use_cov_diag = bool(use_cov_diag)

        self.y_emb = nn.Embedding(self.num_classes, self.y_emb_dim)

        stat_in_dim = self.feature_dim + (self.feature_dim if self.use_cov_diag else 0)
        self.stat_mlp = nn.Sequential(
            nn.Linear(stat_in_dim, self.stat_emb_dim),
            nn.ReLU(),
            nn.Linear(self.stat_emb_dim, self.stat_emb_dim),
            nn.ReLU(),
        )

        in_dim = self.noise_dim + self.y_emb_dim + self.stat_emb_dim
        layers = []
        d = in_dim
        for _ in range(max(0, self.n_hidden_layers)):
            layers.extend([nn.Linear(d, self.hidden_dim), nn.ReLU()])
            d = self.hidden_dim
        layers.append(nn.Linear(d, self.feature_dim))
        self.gen_mlp = nn.Sequential(*layers)

    def forward(
        self,
        y: torch.Tensor,
        *,
        mu: torch.Tensor,
        cov_diag: Optional[torch.Tensor] = None,
        z: Optional[torch.Tensor] = None,
        verbose: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
          y: (B,) int64 labels
          mu: (B, d) class mean for each sample label
          cov_diag: (B, d) diag(cov) for each sample label (optional if use_cov_diag=False)
          z: (B, noise_dim) optional noise
        Returns:
          dict with keys: 'output' (B,d) and optionally 'eps'
        """
        y = y.long()
        bsz = y.shape[0]
        device = y.device

        if z is None:
            z = torch.randn((bsz, self.noise_dim), device=device)

        yv = self.y_emb(y)  # (B, y_emb_dim)

        if self.use_cov_diag:
            if cov_diag is None:
                raise ValueError("cov_diag is required when use_cov_diag=True")
            stat_in = torch.cat([mu, cov_diag], dim=1)
        else:
            stat_in = mu
        sv = self.stat_mlp(stat_in)  # (B, stat_emb_dim)

        h = torch.cat([z, yv, sv], dim=1)
        out = self.gen_mlp(h)
        if self.relu_output:
            out = F.relu(out)

        res = {"output": out}
        if verbose:
            res["eps"] = z
        return res


def gather_by_label(stats: LowStatsTensors, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Gather per-sample stats tensors by label.
    Returns (mu, cov_diag, rf_mean) each of shape (B, *)
    """
    y = y.long()
    mu = stats.mu.to(y.device)[y]
    cov_diag = stats.cov_diag.to(y.device)[y]
    rf_mean = stats.rf_mean.to(y.device)[y]
    return mu, cov_diag, rf_mean



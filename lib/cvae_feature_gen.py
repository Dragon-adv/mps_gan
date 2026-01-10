"""
Conditional VAE for low-level feature synthesis.

Use cases:
  1) 集中式训练：汇聚各客户端上传的低级特征+标签，训练条件VAE。
  2) 生成：给定标签批量采样虚拟低级特征，供 Stage-4 或其他下游使用。
  3) 特征来源：
     - 直接加载已缓存特征文件（npz/pt）
     - 或通过已训练好的低级编码器/完整模型实时提取

接口概览:
  - extract_low_features(model, dataloader, device, save_path=None)
  - load_cached_features(path)
  - TensorFeatureDataset / make_feature_loader(...)
  - ConditionalFeatureVAE (cVAE 模型)
  - cvae_loss(...)
  - train_cvae(...)
  - generate_features(model, labels, num_per_label, temperature)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Iterable, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset


# -----------------------------------------------------------------------------
# 数据加载与提取
# -----------------------------------------------------------------------------


@torch.no_grad()
def extract_low_features(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device | str,
    *,
    save_path: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    使用已训练好的模型提取低级特征（raw，未归一化）。
    适配 CNNCifar/CNNMnist 等模型，要求 forward 返回 (logits, log_probs, high_raw, low_raw, proj)。
    """
    model.eval()
    feats = []
    labels = []
    for images, y in dataloader:
        images = images.to(device)
        y = y.to(device)
        _logits, _log_probs, _high_raw, low_raw, _proj = model(images)
        feats.append(low_raw.detach().cpu())
        labels.append(y.detach().cpu())

    features = torch.cat(feats, dim=0)
    ys = torch.cat(labels, dim=0)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({"features": features, "labels": ys}, save_path)

    return features, ys


def load_cached_features(path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    支持 .pt / .pth (torch.save) 和 .npz (numpy)。
    期望键：
      - torch: dict with keys ["features", "labels"]
      - npz: arrays named ["features", "labels"]
    """
    if path.endswith((".pt", ".pth")):
        obj = torch.load(path, map_location="cpu")
        return obj["features"], obj["labels"]
    if path.endswith(".npz"):
        obj = np.load(path)
        feats = torch.tensor(obj["features"])
        labels = torch.tensor(obj["labels"])
        return feats, labels
    raise ValueError(f"Unsupported cache format: {path}")


class TensorFeatureDataset(Dataset):
    """
    以张量形式存储 (features, labels) 的简单 Dataset。
    可选 transform_fn 对特征做增强/归一化。
    """

    def __init__(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        transform_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        if features.shape[0] != labels.shape[0]:
            raise ValueError("features and labels must have same length.")
        self.features = features
        self.labels = labels.long()
        self.transform_fn = transform_fn

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.features[idx]
        if self.transform_fn:
            x = self.transform_fn(x)
        return x, self.labels[idx]


class SameClassMixupFeatureDataset(Dataset):
    """
    同类特征 mixup（特征级线性插值）：
      x' = lam * x_i + (1-lam) * x_j, where y_i == y_j

    - label 保持为硬标签（仍返回 y_i）。
    - 若某类样本数 < 2，则该类样本不会被 mix（退化为原样本）。
    - 该 Dataset 需要 labels 做同类采样，因此无法通过 transform_fn 实现。
    """

    def __init__(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        *,
        lam: Optional[float] = None,
        lam_mode: Literal["fixed", "beta"] = "fixed",
        beta_alpha: float = 0.4,
        beta_beta: Optional[float] = None,
        p: float = 1.0,
        seed: Optional[int] = None,
    ):
        if features.shape[0] != labels.shape[0]:
            raise ValueError("features and labels must have same length.")
        lam_mode = str(lam_mode).lower().strip()
        if lam_mode not in ("fixed", "beta"):
            raise ValueError(f"mixup lam_mode must be one of ['fixed','beta'], got {lam_mode}")
        if lam_mode == "fixed":
            if lam is None:
                raise ValueError("mixup lam must be provided when lam_mode='fixed'")
            if not (0.0 <= float(lam) <= 1.0):
                raise ValueError(f"mixup lam must be in [0,1], got {lam}")
        else:
            a = float(beta_alpha)
            b = float(beta_beta) if beta_beta is not None else float(beta_alpha)
            if not (a > 0.0 and b > 0.0):
                raise ValueError(
                    f"mixup beta_alpha/beta_beta must be > 0, got alpha={beta_alpha}, beta={beta_beta}"
                )
        if not (0.0 <= float(p) <= 1.0):
            raise ValueError(f"mixup p must be in [0,1], got {p}")

        self.features = features
        self.labels = labels.long()
        self.lam_mode: Literal["fixed", "beta"] = lam_mode  # type: ignore[assignment]
        self.lam = (float(lam) if lam is not None else None)
        self.beta_alpha = float(beta_alpha)
        self.beta_beta = (float(beta_beta) if beta_beta is not None else None)
        self.p = float(p)

        # 预先构建每个类别到索引列表的映射
        self._class_to_indices: dict[int, np.ndarray] = {}
        for c in torch.unique(self.labels).tolist():
            idxs = torch.nonzero(self.labels == int(c), as_tuple=False).view(-1).detach().cpu().numpy()
            self._class_to_indices[int(c)] = idxs.astype(np.int64, copy=False)

        # numpy RNG：用于 beta/lambda 采样与配对采样，保证 seed 可复现（默认 DataLoader num_workers=0）
        self._np_rng = np.random.default_rng(None if seed is None else int(seed))

    def __len__(self) -> int:
        return self.features.shape[0]

    def _sample_same_class_index(self, cls: int, avoid_idx: int) -> int:
        idxs = self._class_to_indices.get(int(cls))
        if idxs is None or idxs.size <= 1:
            return int(avoid_idx)

        pos = int(self._np_rng.integers(low=0, high=int(idxs.size)))
        j = int(idxs[pos])
        if j == int(avoid_idx):
            j = int(idxs[(pos + 1) % int(idxs.size)])
        return j

    def _sample_lam(self) -> float:
        """
        返回单个样本的 mixup lambda。
        - fixed: 直接返回 self.lam
        - beta:  从 Beta(alpha, beta) 采样
        """
        if self.lam_mode == "fixed":
            # lam 在 __init__ 已校验非 None
            return float(self.lam)  # type: ignore[arg-type]

        a = float(self.beta_alpha)
        b = float(self.beta_beta) if self.beta_beta is not None else float(self.beta_alpha)
        lam = float(self._np_rng.beta(a, b))
        # 数值安全夹紧
        if lam < 0.0:
            lam = 0.0
        elif lam > 1.0:
            lam = 1.0
        return lam

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x_i = self.features[idx]
        y = self.labels[idx]

        if self.p > 0.0:
            do_mix = float(self._np_rng.random()) < self.p
        else:
            do_mix = False

        lam = self._sample_lam()
        if do_mix and lam not in (0.0, 1.0):
            j = self._sample_same_class_index(int(y.item()), int(idx))
            x_j = self.features[j]
            x = lam * x_i + (1.0 - lam) * x_j
        else:
            x = x_i

        return x, y


def make_feature_loader(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    batch_size: int = 256,
    shuffle: bool = True,
    transform_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    mixup_lam: Optional[float] = None,
    mixup_lam_mode: Literal["fixed", "beta"] = "fixed",
    mixup_beta_alpha: float = 0.4,
    mixup_beta_beta: Optional[float] = None,
    mixup_p: float = 1.0,
    mixup_seed: Optional[int] = None,
) -> DataLoader:
    # 兼容旧行为：
    # - fixed + mixup_lam is None => 不启用 mixup
    # - beta  => 即使 mixup_lam is None，也启用 mixup（lambda 来自 Beta 分布）
    if mixup_lam is None and str(mixup_lam_mode).lower().strip() == "fixed":
        ds: Dataset = TensorFeatureDataset(features, labels, transform_fn=transform_fn)
    else:
        if transform_fn is not None:
            raise ValueError("transform_fn cannot be used together with same-class mixup.")
        ds = SameClassMixupFeatureDataset(
            features,
            labels,
            lam=(float(mixup_lam) if mixup_lam is not None else None),
            lam_mode=str(mixup_lam_mode).lower().strip(),  # type: ignore[arg-type]
            beta_alpha=float(mixup_beta_alpha),
            beta_beta=(float(mixup_beta_beta) if mixup_beta_beta is not None else None),
            p=float(mixup_p),
            seed=mixup_seed,
        )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


# -----------------------------------------------------------------------------
# 条件 VAE 模型
# -----------------------------------------------------------------------------


@dataclass
class CVAEConfig:
    feature_dim: int
    num_classes: int
    latent_dim: int = 64
    hidden_dim: int = 256
    n_hidden: int = 2
    y_emb_dim: int = 32
    recon_loss: Literal["l1", "l2"] = "l2"
    beta: float = 1.0  # KL 系数
    kl_anneal_steps: int = 0  # >0 时线性退火到 beta


class ConditionalFeatureVAE(nn.Module):
    """
    cVAE: q(z|x,y), p(x|z,y) 生成低级特征。
    - 编码器输入: x + y_emb
    - 解码器输入: z + y_emb
    """

    def __init__(self, cfg: CVAEConfig):
        super().__init__()
        self.cfg = cfg
        fdim = cfg.feature_dim
        ydim = cfg.y_emb_dim
        zdim = cfg.latent_dim
        hdim = cfg.hidden_dim
        n_hidden = max(0, cfg.n_hidden)

        self.y_emb = nn.Embedding(cfg.num_classes, ydim)

        enc_layers = []
        in_dim = fdim + ydim
        d = in_dim
        for _ in range(n_hidden):
            enc_layers.extend([nn.Linear(d, hdim), nn.ReLU()])
            d = hdim
        self.encoder = nn.Sequential(*enc_layers)
        self.mu_head = nn.Linear(d, zdim)
        self.logvar_head = nn.Linear(d, zdim)

        dec_layers = []
        d = zdim + ydim
        for _ in range(n_hidden):
            dec_layers.extend([nn.Linear(d, hdim), nn.ReLU()])
            d = hdim
        dec_layers.append(nn.Linear(d, fdim))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        yv = self.y_emb(y.long())
        h = torch.cat([x, yv], dim=1)
        h = self.encoder(h) if len(self.encoder) > 0 else h
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        yv = self.y_emb(y.long())
        h = torch.cat([z, yv], dim=1)
        return self.decoder(h)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, y)
        return recon, mu, logvar


def cvae_loss(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    *,
    beta: float = 1.0,
    recon_loss: Literal["l1", "l2"] = "l2",
) -> Tuple[torch.Tensor, dict]:
    if recon_loss == "l1":
        rec = F.l1_loss(recon_x, x, reduction="mean")
    elif recon_loss == "l2":
        rec = F.mse_loss(recon_x, x, reduction="mean")
    else:
        raise ValueError(recon_loss)
    # KL divergence to N(0, I)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    loss = rec + beta * kl
    return loss, {"rec": rec.item(), "kl": kl.item(), "beta": beta}


# -----------------------------------------------------------------------------
# 训练与生成
# -----------------------------------------------------------------------------


def _anneal_beta(step: int, cfg: CVAEConfig) -> float:
    if cfg.kl_anneal_steps <= 0:
        return cfg.beta
    t = min(1.0, step / float(cfg.kl_anneal_steps))
    return float(cfg.beta) * t


def train_cvae(
    model: ConditionalFeatureVAE,
    loader: DataLoader,
    *,
    epochs: int,
    device: torch.device | str,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    max_grad_norm: Optional[float] = None,
    log_every: int = 50,
) -> dict:
    """
    简单的单机训练循环。返回日志 dict。
    """
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    logs = {"loss": [], "rec": [], "kl": []}
    global_step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        for batch_idx, (x, y) in enumerate(loader, start=1):
            global_step += 1
            x = x.to(device).float()
            y = y.to(device)
            beta = _anneal_beta(global_step, model.cfg)

            recon, mu, logvar = model(x, y)
            loss, stats = cvae_loss(
                recon, x, mu, logvar, beta=beta, recon_loss=model.cfg.recon_loss
            )

            opt.zero_grad()
            loss.backward()
            if max_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            opt.step()

            logs["loss"].append(float(loss.item()))
            logs["rec"].append(stats["rec"])
            logs["kl"].append(stats["kl"])

            if log_every and batch_idx % log_every == 0:
                print(
                    f"[epoch {epoch}/{epochs}] step {batch_idx} "
                    f"loss={loss.item():.4f} rec={stats['rec']:.4f} kl={stats['kl']:.4f} beta={beta:.3f}"
                )
    return logs


@torch.no_grad()
def generate_features(
    model: ConditionalFeatureVAE,
    labels: torch.Tensor,
    *,
    num_per_label: int = 1,
    temperature: float = 1.0,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """
    给定标签批量采样特征。
    Args:
      labels: (B,) int64
      num_per_label: 每个标签采样多少个样本；>1 时自动重复标签。
      temperature: 采样温度，缩放标准差。
    Returns:
      Tensor shape (B * num_per_label, feature_dim)
    """
    model.eval()
    device = torch.device(device)
    model = model.to(device)

    labels = labels.long().to(device)
    if num_per_label > 1:
        labels = labels.repeat_interleave(num_per_label)

    z = torch.randn((labels.shape[0], model.cfg.latent_dim), device=device) * float(
        temperature
    )
    feats = model.decode(z, labels)
    return feats.detach().cpu()


# -----------------------------------------------------------------------------
# 便捷加载/保存
# -----------------------------------------------------------------------------


def save_cvae(model: ConditionalFeatureVAE, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "cfg": model.cfg}, path)


def load_cvae(path: str, device: torch.device | str = "cpu") -> ConditionalFeatureVAE:
    obj = torch.load(path, map_location=device)
    cfg = obj["cfg"]
    model = ConditionalFeatureVAE(cfg)
    model.load_state_dict(obj["state_dict"])
    model.to(device)
    return model


# -----------------------------------------------------------------------------
# 示例：从缓存或编码器构建 DataLoader
# -----------------------------------------------------------------------------


def build_loader_from_cache(
    cache_path: str,
    *,
    batch_size: int = 256,
    shuffle: bool = True,
    transform_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    mixup_lam: Optional[float] = None,
    mixup_lam_mode: Literal["fixed", "beta"] = "fixed",
    mixup_beta_alpha: float = 0.4,
    mixup_beta_beta: Optional[float] = None,
    mixup_p: float = 1.0,
    mixup_seed: Optional[int] = None,
) -> Tuple[DataLoader, int, int]:
    feats, labels = load_cached_features(cache_path)
    feature_dim = feats.shape[1]
    num_classes = int(labels.max().item() + 1)
    loader = make_feature_loader(
        feats,
        labels,
        batch_size=batch_size,
        shuffle=shuffle,
        transform_fn=transform_fn,
        mixup_lam=mixup_lam,
        mixup_lam_mode=mixup_lam_mode,
        mixup_beta_alpha=mixup_beta_alpha,
        mixup_beta_beta=mixup_beta_beta,
        mixup_p=mixup_p,
        mixup_seed=mixup_seed,
    )
    return loader, feature_dim, num_classes


def build_loader_from_encoder(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device | str,
    *,
    cache_path: Optional[str] = None,
    batch_size: int = 256,
    shuffle: bool = True,
    transform_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    mixup_lam: Optional[float] = None,
    mixup_lam_mode: Literal["fixed", "beta"] = "fixed",
    mixup_beta_alpha: float = 0.4,
    mixup_beta_beta: Optional[float] = None,
    mixup_p: float = 1.0,
    mixup_seed: Optional[int] = None,
) -> Tuple[DataLoader, int, int]:
    feats, labels = extract_low_features(model, dataloader, device, save_path=cache_path)
    feature_dim = feats.shape[1]
    num_classes = int(labels.max().item() + 1)
    loader = make_feature_loader(
        feats,
        labels,
        batch_size=batch_size,
        shuffle=shuffle,
        transform_fn=transform_fn,
        mixup_lam=mixup_lam,
        mixup_lam_mode=mixup_lam_mode,
        mixup_beta_alpha=mixup_beta_alpha,
        mixup_beta_beta=mixup_beta_beta,
        mixup_p=mixup_p,
        mixup_seed=mixup_seed,
    )
    return loader, feature_dim, num_classes


#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage-3 (集中式) 训练脚本：使用 cVAE 合成低级特征

功能：
  1) 直接用 Stage-1 已训练好的模型对全量训练集提取 low-level 特征。
  2) 基于提取的特征和标签训练条件 VAE (lib/cvae_feature_gen.py)。
  3) 支持将提取的特征缓存到磁盘，重复实验时可直接加载缓存。
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from types import SimpleNamespace
from typing import Tuple
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 确保项目根目录在 sys.path 中
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from lib.checkpoint import load_checkpoint  # noqa: E402
from lib.cvae_feature_gen import (  # noqa: E402
    CVAEConfig,
    ConditionalFeatureVAE,
    build_loader_from_cache,
    build_loader_from_encoder,
    train_cvae,
)
from lib.models.models import CNNCifar  # noqa: E402


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_dataset(dataset: str, data_dir: str) -> Tuple[torch.utils.data.Dataset, int]:
    dataset = dataset.lower()
    if dataset == "cifar10":
        transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        ds = datasets.CIFAR10(root=os.path.join(data_dir, dataset), train=True, download=True, transform=transform)
        num_classes = 10
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return ds, num_classes


def _with_timestamp_dir(path: str) -> str:
    """
    保持路径不变，只确保父目录存在。
    若文件已存在，仅提示，不做改名；调用方可自行决定是否覆盖。
    """
    if not path:
        return path
    base_dir = os.path.dirname(path)
    fname = os.path.basename(path)
    if base_dir:
        os.makedirs(base_dir, exist_ok=True)
    if os.path.exists(path):
        print(f"[warn] {path} 已存在，可能会被覆盖。")
    else:
        print(f"[info] 输出将写入: {path}")
    return path


def _infer_log_dir(stage1_ckpt_path: str) -> str:
    """
    尝试根据 stage1 ckpt 路径推断 log_dir:
      <LOGDIR>/stage1/ckpts/best-wo.pt -> <LOGDIR>
    若未找到 "stage1" 目录，则回退到 ckpt 上一级目录的上一级。
    """
    p = Path(stage1_ckpt_path).resolve()
    parts = p.parts
    if "stage1" in parts:
        idx = parts.index("stage1")
        if idx > 0:
            return str(Path(*parts[:idx]))
    # fallback: parent of parent
    return str(p.parent.parent)


def _make_run_tag(args) -> str:
    return f"seed{args.seed}_bs{args.batch_size}_ep{args.epochs}_lr{args.lr}"


def _make_time_tag() -> str:
    # 例如：20260110_143012（日期+时间）
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _load_stage1_payload(path: str, map_location: str | torch.device):
    ckpt = load_checkpoint(path, map_location=map_location)
    if not isinstance(ckpt, dict):
        raise ValueError(f"Invalid checkpoint payload type: {type(ckpt)}")

    ckpt_meta = ckpt.get("meta", {}) or {}
    ckpt_args = ckpt.get("args", {}) or {}
    ckpt_state = ckpt.get("state", {}) or {}

    dataset = str(ckpt_meta.get("dataset", ckpt_args.get("dataset", "cifar10"))).lower()
    num_classes = int(ckpt_meta.get("num_classes", ckpt_args.get("num_classes", 10)))

    return ckpt, ckpt_meta, ckpt_args, ckpt_state, dataset, num_classes


def _load_stage1_state_dict(model: torch.nn.Module, ckpt: dict, ckpt_state: dict) -> str:
    """
    尝试从 Stage-1 checkpoint 中加载权重，优先级：
      1) state['global_model_state_dict']
      2) state['local_models_full_state_dicts'][0]（或第一个 key）
      3) state['state_dict']
      4) ckpt['state_dict']
      5) 兜底：ckpt 本身像是 state_dict（值都是 Tensor）
    返回实际使用的字段名称，便于提示。
    """
    # 1) global model
    if isinstance(ckpt_state, dict):
        sd_global = ckpt_state.get("global_model_state_dict", None)
        if isinstance(sd_global, dict):
            model.load_state_dict(sd_global, strict=True)
            return "state.global_model_state_dict"

        # 2) first local model
        local_sd = ckpt_state.get("local_models_full_state_dicts", None)
        if isinstance(local_sd, dict) and len(local_sd) > 0:
            first_key = 0 if 0 in local_sd else sorted(local_sd.keys(), key=lambda k: int(k) if str(k).isdigit() else str(k))[0]
            model.load_state_dict(local_sd[first_key], strict=True)
            return f"state.local_models_full_state_dicts[{first_key}]"

        # 3) generic state_dict inside state
        sd_generic = ckpt_state.get("state_dict", None)
        if isinstance(sd_generic, dict):
            model.load_state_dict(sd_generic, strict=True)
            return "state.state_dict"

    # 4) top-level state_dict
    if isinstance(ckpt, dict):
        sd_top = ckpt.get("state_dict", None)
        if isinstance(sd_top, dict):
            model.load_state_dict(sd_top, strict=True)
            return "ckpt.state_dict"

        # 5) ckpt resembles a plain state_dict (all values are tensors)
        if len(ckpt) > 0 and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            missing_unexp = model.load_state_dict(ckpt, strict=False)
            if getattr(missing_unexp, "missing_keys", []) or getattr(missing_unexp, "unexpected_keys", []):
                print(f"[warn] Loaded with missing/unexpected keys: {missing_unexp}")
            return "ckpt_as_state_dict"

    raise ValueError("Cannot find a valid state_dict in the provided Stage-1 checkpoint.")


def main() -> int:
    p = argparse.ArgumentParser(description="Stage-3 cVAE (集中式) 训练脚本")
    # 数据 / 模型
    p.add_argument("--dataset", type=str, default="cifar10", help="目前支持 cifar10")
    p.add_argument("--data_dir", type=str, default="../data/", help="数据根目录（内部会拼上数据集名）")
    p.add_argument("--stage1_ckpt_path", type=str, required=True, help="Stage-1 训练好的模型权重路径")
    # 训练超参
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--latent_dim", type=int, default=64)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--n_hidden", type=int, default=2)
    p.add_argument("--beta", type=float, default=1.0, help="KL 系数")
    p.add_argument("--kl_anneal_steps", type=int, default=0, help=">0 时线性退火到 beta")
    p.add_argument("--recon_loss", type=str, default="l2", choices=["l1", "l2"])
    p.add_argument("--max_grad_norm", type=float, default=None, help="可选梯度裁剪")
    # 特征增强（默认关闭，保持原行为）
    p.add_argument(
        "--mixup_enable",
        action="store_true",
        help="启用同类特征 mixup（在训练 cVAE 的 feature_loader 内进行；缓存文件不被修改）",
    )
    p.add_argument(
        "--mixup_lam",
        type=float,
        default=0.7,
        help="mixup 系数 lambda：x' = lam*x_i + (1-lam)*x_j（同类）",
    )
    p.add_argument(
        "--mixup_p",
        type=float,
        default=1.0,
        help="对每个样本执行 mixup 的概率（0~1）",
    )
    p.add_argument(
        "--mixup_seed",
        type=int,
        default=None,
        help="mixup 采样随机种子（不设则使用随机种子）",
    )
    # 设备 / 其他
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=1234)
    # 缓存与保存
    p.add_argument("--out_dir", type=str, default=None, help="输出目录，若不指定则根据 stage1 路径自动推断")
    p.add_argument("--cache_path", type=str, default=None, help="提取的低级特征缓存路径（.pt/.npz），默认放在 out_dir/cache")
    p.add_argument("--save_path", type=str, default=None, help="训练好的 cVAE 保存路径，默认 out_dir/generator.pt")

    args = p.parse_args()

    # 0) 预加载 Stage-1 ckpt 元信息，以确定数据集与类别数
    ckpt, ckpt_meta, ckpt_args, ckpt_state, ckpt_dataset, ckpt_num_classes = _load_stage1_payload(
        args.stage1_ckpt_path, map_location="cpu"
    )

    # 0) 推断输出目录 / 缓存 / 保存路径
    log_dir = _infer_log_dir(args.stage1_ckpt_path)
    run_tag = _make_run_tag(args)
    # 默认用“日期+时间”命名，避免重复运行覆盖旧结果
    time_tag = _make_time_tag()
    out_dir = args.out_dir or os.path.join(log_dir, "stage3", "cvae", time_tag)
    save_path = args.save_path or os.path.join(out_dir, "generator.pt")
    cache_path = args.cache_path or os.path.join(out_dir, "cache", "low_feats.pt")

    # 数据集名称以 ckpt 为准（若 CLI 未显式覆盖）
    dataset_name = (args.dataset or ckpt_dataset).lower()
    if dataset_name != ckpt_dataset:
        print(f"[warn] CLI dataset={args.dataset} 与 ckpt.dataset={ckpt_dataset} 不一致，已使用 ckpt 配置。")
        dataset_name = ckpt_dataset

    num_classes = ckpt_num_classes

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.set_device(int(args.gpu))

    _seed_all(args.seed)

    # 1) 构建数据集
    train_ds, num_classes_ds = _build_dataset(dataset_name, args.data_dir)
    if num_classes_ds != num_classes:
        # 以 ckpt 中的 num_classes 为主，但提示潜在不一致
        print(f"[warn] num_classes: ckpt={num_classes} dataset_builder={num_classes_ds}. 使用 ckpt 值。")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 2) 加载 Stage-1 模型并提取特征 (或直接从缓存加载)
    model_args = SimpleNamespace(num_classes=num_classes)
    stage1_model = CNNCifar(model_args).to(device)
    loaded_from = _load_stage1_state_dict(stage1_model, ckpt, ckpt_state)
    print(f"[info] Loaded Stage-1 weights from {loaded_from}")

    if cache_path and os.path.exists(cache_path):
        print(f"[info] Loading cached features from {cache_path}")
        feature_loader, feat_dim, num_classes = build_loader_from_cache(
            cache_path,
            batch_size=args.batch_size,
            shuffle=True,
            mixup_lam=(args.mixup_lam if args.mixup_enable else None),
            mixup_p=args.mixup_p,
            mixup_seed=args.mixup_seed,
        )
    else:
        if cache_path:
            cache_path = _with_timestamp_dir(cache_path)
        print("[info] Extracting low-level features from Stage-1 model ...")
        feature_loader, feat_dim, num_classes = build_loader_from_encoder(
            model=stage1_model,
            dataloader=train_loader,
            device=device,
            cache_path=cache_path,
            batch_size=args.batch_size,
            shuffle=True,
            mixup_lam=(args.mixup_lam if args.mixup_enable else None),
            mixup_p=args.mixup_p,
            mixup_seed=args.mixup_seed,
        )

    # 3) 配置并训练 cVAE
    cfg = CVAEConfig(
        feature_dim=feat_dim,
        num_classes=num_classes,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        n_hidden=args.n_hidden,
        recon_loss=args.recon_loss,
        beta=args.beta,
        kl_anneal_steps=args.kl_anneal_steps,
    )
    cvae = ConditionalFeatureVAE(cfg)

    print(
        f"[info] Start training cVAE: epochs={args.epochs}, "
        f"batch_size={args.batch_size}, lr={args.lr}, beta={args.beta}"
    )
    train_cvae(
        cvae,
        feature_loader,
        epochs=args.epochs,
        device=device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        log_every=50,
    )

    # 4) 保存模型
    save_path = _with_timestamp_dir(save_path)
    out_dir = os.path.dirname(save_path)
    os.makedirs(out_dir, exist_ok=True)
    torch.save(cvae.state_dict(), save_path)

    meta = {
        "stage": 3,
        "variant": "cvae",
        "time_tag": time_tag,
        "run_tag": run_tag,
        "dataset": args.dataset,
        "num_classes": num_classes,
        "feature_dim": feat_dim,
        "stage1_ckpt_path": args.stage1_ckpt_path,
        "cache_path": cache_path,
        "seed": args.seed,
        "mixup": {
            "enable": bool(args.mixup_enable),
            "lam": float(args.mixup_lam),
            "p": float(args.mixup_p),
            "seed": (int(args.mixup_seed) if args.mixup_seed is not None else None),
            "note": "same-class feature mixup applied in feature_loader; cache file is unchanged",
        },
        "train": {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
        },
        "cvae_cfg": {
            "latent_dim": int(args.latent_dim),
            "hidden_dim": int(args.hidden_dim),
            "n_hidden": int(args.n_hidden),
            "beta": float(args.beta),
            "kl_anneal_steps": int(args.kl_anneal_steps),
            "recon_loss": str(args.recon_loss),
        },
        "save_path": save_path,
    }
    meta_path = os.path.join(out_dir, "generator_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[done] Saved cVAE state_dict to {save_path}")
    print(f"[done] Saved meta to {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


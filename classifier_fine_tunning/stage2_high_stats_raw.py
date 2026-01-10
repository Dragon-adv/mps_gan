from __future__ import annotations

"""
Stage-2 (raw-high): aggregate global statistics in raw high-level feature space.

We compute, for each class c:
  - mean:      mu_g^c
  - cov:       Sigma_g^c  (via E[zz^T] - mu mu^T)
  - rff_mean:  phi_g^c    where RFF is applied to raw-high features (NOT normalized)
  - sample_per_class

Input:
  - Stage-1 checkpoint (.pt) saved by exps/federated_main.py
  - split.pkl used by Stage-1 (for consistent train split)
Output:
  - global_high_stats_raw.pt (and .pkl) containing global_stats + rf_model_state + meta
"""

import argparse
import json
import os
import pickle
import sys
import time
from dataclasses import asdict, dataclass
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils import (
    collect_env_info,
    ensure_repo_root_on_path,
    infer_logdir_from_stage1_ckpt,
    make_isolated_run_dir,
    normalize_split_indices,
    resolve_device,
    seed_all,
)


ensure_repo_root_on_path()

from lib.checkpoint import load_checkpoint
from lib.models.models import (
    CNNCifar,
    CNNFemnist,
    CNNFashion_Mnist,
    CNNMnist,
    ModelCT,
    ResNetWithFeatures,
)
from lib.sfd_utils import RFF, aggregate_global_statistics
from lib.split_manager import load_split
from lib.utils import get_dataset


@dataclass
class RFArgs:
    rf_dim: int
    rbf_gamma: float
    rf_type: str
    rf_seed: int


def _build_client_model(args: SimpleNamespace, client_id: int) -> torch.nn.Module:
    """
    Mirror exps/federated_main.py Stage-2 model reconstruction.
    """
    if args.dataset == "mnist":
        if getattr(args, "mode", None) == "model_heter":
            if client_id < 7:
                args.out_channels = 18
            elif client_id < 14:
                args.out_channels = 20
            else:
                args.out_channels = 22
        else:
            args.out_channels = 20
        return CNNMnist(args=args)

    if args.dataset == "femnist":
        if getattr(args, "mode", None) == "model_heter":
            if client_id < 7:
                args.out_channels = 18
            elif client_id < 14:
                args.out_channels = 20
            else:
                args.out_channels = 22
        else:
            args.out_channels = 20
        return CNNFemnist(args=args)

    if args.dataset in {"cifar10", "cifar100", "flowers", "defungi", "realwaste"}:
        return CNNCifar(args=args)

    if args.dataset == "tinyimagenet":
        args.num_classes = 200
        return ModelCT(out_dim=256, n_classes=args.num_classes)

    if args.dataset == "fashion":
        return CNNFashion_Mnist(args=args)

    if args.dataset == "imagenet":
        return ResNetWithFeatures(base="resnet18", num_classes=args.num_classes)

    raise ValueError(f"Unsupported dataset for Stage-2(raw-high): {args.dataset}")


@torch.no_grad()
def _infer_high_dim(model: torch.nn.Module, dataset: str, device: str) -> int:
    model = model.to(device).eval()
    if dataset in {"mnist", "femnist", "fashion"}:
        x = torch.randn(1, 1, 28, 28, device=device)
    elif dataset == "tinyimagenet":
        x = torch.randn(1, 3, 64, 64, device=device)
    elif dataset == "imagenet":
        x = torch.randn(1, 3, 224, 224, device=device)
    else:
        x = torch.randn(1, 3, 32, 32, device=device)
    out = model(x)
    if not (isinstance(out, tuple) and len(out) >= 3):
        raise ValueError("Model forward output unexpected; expected tuple with high features at index 2.")
    high = out[2]
    if high.dim() != 2:
        raise ValueError(f"High feature expected 2D (B,d), got shape={tuple(high.shape)}")
    return int(high.shape[1])


@torch.no_grad()
def compute_local_high_stats_raw(
    *,
    model: torch.nn.Module,
    trainloader: DataLoader,
    rf_model: RFF,
    num_classes: int,
    device: str,
) -> Dict[str, Any]:
    """
    Compute per-client, per-class stats on raw high features (no per-sample normalization):
      - class_means[c]      = mean(z)
      - class_outers[c]     = mean(zz^T)
      - class_rf_means[c]   = mean(RFF(z))
      - sample_per_class[c] = count
    """
    model = model.to(device).eval()
    rf_model = rf_model.to(device).eval()

    # Accumulators in float64 for numerical stability
    sum_z = [torch.zeros(rf_model.d, dtype=torch.float64, device=device) for _ in range(num_classes)]
    sum_outer = [torch.zeros((rf_model.d, rf_model.d), dtype=torch.float64, device=device) for _ in range(num_classes)]
    sum_rf = [torch.zeros(rf_model.D, dtype=torch.float64, device=device) for _ in range(num_classes)]
    n = torch.zeros(num_classes, dtype=torch.long, device=device)

    for images, labels in trainloader:
        images = images.to(device)
        labels = labels.to(device).long()
        out = model(images)
        if not (isinstance(out, tuple) and len(out) >= 3):
            raise ValueError("Model forward output unexpected.")
        high_raw = out[2]  # (B, d), raw-high

        # Compute RFF on raw-high (no normalization)
        rf = rf_model(high_raw)  # (B, D)

        # Accumulate per class using boolean masks
        for c in torch.unique(labels).tolist():
            c = int(c)
            mask = labels == c
            if not torch.any(mask):
                continue
            zc = high_raw[mask]  # (n_c, d)
            rfc = rf[mask]       # (n_c, D)
            nc = int(mask.sum().item())
            n[c] += nc
            sum_z[c] += zc.to(torch.float64).sum(dim=0)
            # sum of zz^T
            zc64 = zc.to(torch.float64)
            sum_outer[c] += zc64.t().matmul(zc64)
            sum_rf[c] += rfc.to(torch.float64).sum(dim=0)

    # Convert to means/outers per class on CPU
    class_means: list[torch.Tensor] = []
    class_outers: list[torch.Tensor] = []
    class_rf_means: list[torch.Tensor] = []
    for c in range(num_classes):
        nc = int(n[c].item())
        if nc <= 0:
            class_means.append(torch.zeros(rf_model.d, dtype=torch.float32))
            class_outers.append(torch.zeros((rf_model.d, rf_model.d), dtype=torch.float32))
            class_rf_means.append(torch.zeros(rf_model.D, dtype=torch.float32))
        else:
            mu = (sum_z[c] / float(nc)).to(torch.float32).detach().cpu()
            outer = (sum_outer[c] / float(nc)).to(torch.float32).detach().cpu()
            phi = (sum_rf[c] / float(nc)).to(torch.float32).detach().cpu()
            class_means.append(mu)
            class_outers.append(outer)
            class_rf_means.append(phi)

    return {
        "high": {
            "class_means": class_means,
            "class_outers": class_outers,
            "class_rf_means": class_rf_means,
        },
        "sample_per_class": n.detach().cpu(),
    }


def main() -> int:
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("--stage1_ckpt_path", type=str, required=True, help="Stage-1 checkpoint path (best-wo.pt / latest.pt)")
    p.add_argument("--split_path", type=str, default=None, help="split.pkl path (if None, infer from ckpt meta/args)")
    p.add_argument("--out_dir", type=str, default=None, help="Base output dir. If None, infer: <logdir>/classifier_fine_tunning/stage2")
    p.add_argument("--run_name", type=str, default=None, help="Optional run name suffix for output directory.")
    p.add_argument("--auto_run_dir", type=int, default=1, help="If 1, create timestamp subdir under out_dir (default: 1).")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=1234)

    # RFF for raw-high
    p.add_argument("--rf_dim", type=int, default=3000)
    p.add_argument("--rbf_gamma", type=float, default=0.01)
    p.add_argument("--rf_type", type=str, default="orf", choices=["orf", "iid"])
    p.add_argument("--rf_seed", type=int, default=42)

    # Extraction
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--max_batches_per_client", type=int, default=-1, help="Debug: limit batches per client (-1 = no limit)")
    args = p.parse_args()

    device = resolve_device(args.gpu)
    seed_all(int(args.seed))

    ckpt = load_checkpoint(args.stage1_ckpt_path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise ValueError("Invalid Stage-1 checkpoint payload.")
    ckpt_meta = ckpt.get("meta", {}) or {}
    ckpt_args = ckpt.get("args", {}) or {}
    ckpt_state = ckpt.get("state", {}) or {}

    dataset = str(ckpt_meta.get("dataset", ckpt_args.get("dataset", None)))
    num_users = int(ckpt_meta.get("num_users", ckpt_args.get("num_users", 20)))
    num_classes = int(ckpt_meta.get("num_classes", ckpt_args.get("num_classes", 10)))

    # Infer logdir / out_dir (stage-isolated + timestamped)
    logdir = infer_logdir_from_stage1_ckpt(args.stage1_ckpt_path, ckpt_meta=ckpt_meta, ckpt_args=ckpt_args)
    base_out_dir = args.out_dir or os.path.join(logdir, "classifier_fine_tunning", "stage2")
    out_dir = make_isolated_run_dir(
        base_out_dir=base_out_dir,
        run_name=args.run_name,
        auto_timestamp_subdir=int(getattr(args, "auto_run_dir", 1)) == 1,
    )

    # Resolve split_path
    split_path = args.split_path or ckpt_meta.get("split_path", None) or ckpt_args.get("split_path", None)
    if not split_path or not os.path.exists(split_path):
        raise FileNotFoundError(f"split_path not found: {split_path}")

    # Load split + datasets
    split = load_split(split_path)
    n_list = split["n_list"]
    k_list = split["k_list"]
    user_groups = split["user_groups"]

    base = dict(ckpt_args)
    base["dataset"] = dataset
    base["num_classes"] = num_classes
    base["num_users"] = num_users
    base["gpu"] = int(args.gpu)
    base["device"] = device
    args_ds = SimpleNamespace(**base)
    train_dataset, _test_dataset, _, _, _, _ = get_dataset(args_ds, n_list, k_list)

    # Load per-client model weights
    local_sd = ckpt_state.get("local_models_full_state_dicts", None)
    if not isinstance(local_sd, dict):
        raise ValueError("Checkpoint missing state['local_models_full_state_dicts'].")

    # Build one model to infer high dim
    model0 = _build_client_model(SimpleNamespace(**base), 0)
    key0 = 0 if 0 in local_sd else "0"
    model0.load_state_dict(local_sd[key0], strict=True)
    high_dim = _infer_high_dim(model0, dataset=dataset, device=device)

    # Build RFF model for raw-high
    # Use rf_seed to initialize deterministically
    import random

    backup = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    random.seed(int(args.rf_seed))
    np.random.seed(int(args.rf_seed))
    torch.manual_seed(int(args.rf_seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.rf_seed))

    rf_model = RFF(d=high_dim, D=int(args.rf_dim), gamma=float(args.rbf_gamma), device=device, rf_type=str(args.rf_type))

    random.setstate(backup[0])
    np.random.set_state(backup[1])
    torch.set_rng_state(backup[2])

    # Compute local stats for each client
    t0 = time.time()
    client_responses: list[dict[str, Any]] = []
    for cid in range(num_users):
        model = _build_client_model(SimpleNamespace(**base), cid).to(device).eval()
        key = cid if cid in local_sd else str(cid)
        if key not in local_sd:
            raise KeyError(f"local_models_full_state_dicts missing client id={cid}")
        model.load_state_dict(local_sd[key], strict=True)

        idxs = normalize_split_indices(user_groups[int(cid)])
        dl = DataLoader(
            torch.utils.data.Subset(train_dataset, idxs),
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=int(args.num_workers),
            drop_last=False,
        )

        # Optional debug limit
        if int(args.max_batches_per_client) > 0:
            dl = list(dl)[: int(args.max_batches_per_client)]

        local_stats = compute_local_high_stats_raw(
            model=model,
            trainloader=dl,
            rf_model=rf_model,
            num_classes=num_classes,
            device=device,
        )
        client_responses.append(local_stats)

    global_stats = aggregate_global_statistics(client_responses=client_responses, class_num=num_classes, stats_level="high")

    rf_args = RFArgs(
        rf_dim=int(args.rf_dim),
        rbf_gamma=float(args.rbf_gamma),
        rf_type=str(args.rf_type),
        rf_seed=int(args.rf_seed),
    )

    meta = {
        "stage": 2,
        "feature_space": "high_raw",
        "dataset": dataset,
        "num_users": num_users,
        "num_classes": num_classes,
        "high_feature_dim": int(high_dim),
        "rf_args": asdict(rf_args),
        "split_path": split_path,
        "stage1_ckpt_path": os.path.abspath(args.stage1_ckpt_path),
        "sec_total": float(time.time() - t0),
    }

    payload = {
        "meta": meta,
        "args": vars(args),
        "state": {
            "global_stats": global_stats,
            "rf_model_state": rf_model.state_dict(),
        },
    }

    meta["logdir"] = logdir
    meta["out_dir_base"] = os.path.abspath(base_out_dir)
    meta["out_dir"] = os.path.abspath(out_dir)
    meta["cmdline"] = " ".join([str(x) for x in sys.argv])
    meta["env"] = collect_env_info()
    meta["stage1_meta"] = ckpt_meta
    meta["stage1_args"] = ckpt_args

    out_pt = os.path.join(out_dir, "global_high_stats_raw.pt")
    out_pkl = os.path.join(out_dir, "global_high_stats_raw.pkl")
    torch.save(payload, out_pt)
    with open(out_pkl, "wb") as f:
        pickle.dump(payload, f)

    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[stage2_raw_high] saved: {out_pt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


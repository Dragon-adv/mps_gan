#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
从 FedMPS 本地训练数据中抽取 low-level 特征，并对比：
  - 原始 low 特征对应的 logits（来自 model(images)）
  - 同类 fixed-lam 特征级 mixup 后，通过 model.forward_from_low 得到的 logits

输出：
  - 对每个客户端输出一个子目录：
      <out_dir>/client_XX/dump.pt
      <out_dir>/client_XX/meta.json

使用示例（Windows PowerShell）：
  py analysis/dump_local_low_logits_and_mixup.py `
    --stage1_ckpt_path "D:\path\to\stage1\ckpts\best-wo.pt" `
    --client_ids all `
    --mixup_lam 0.7 `
    --mixup_p 1.0 `
    --gpu 0
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

# Ensure project root is on sys.path when running as a script:
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from lib.checkpoint import load_checkpoint  # noqa: E402
from lib.models.models import CNNCifar  # noqa: E402
from lib.split_manager import load_split  # noqa: E402


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


def _make_default_out_dir() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(_ROOT, "analysis", "local_mixup", ts)


@torch.no_grad()
def _collect_local_low_and_logits(
    *,
    model: CNNCifar,
    dataset,
    indices: List[int],
    device: str,
    batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    从某 client 的本地训练索引（全部）收集：
      - x_low_raw: (N,D)
      - y: (N,)
      - logits_from_forward: (N,C)  来自 model(images)
      - logits_from_low: (N,C)      来自 model.forward_from_low(x_low_raw)
    """
    if len(indices) == 0:
        raise ValueError("client train indices is empty.")

    dl = DataLoader(
        Subset(dataset, [int(i) for i in indices]),
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    model.eval()
    xs: List[torch.Tensor] = []
    ys: List[torch.Tensor] = []
    logits_fwd: List[torch.Tensor] = []
    logits_low: List[torch.Tensor] = []

    for images, labels in dl:
        images = images.to(device)
        labels = labels.to(device).long()
        logits, _log_probs, _high_raw, low_raw, _proj = model(images)
        logits2, _lp2, _high2, _proj2 = model.forward_from_low(low_raw)
        xs.append(low_raw.detach().cpu().float())
        ys.append(labels.detach().cpu().long())
        logits_fwd.append(logits.detach().cpu().float())
        logits_low.append(logits2.detach().cpu().float())

    x = torch.cat(xs, dim=0)
    y = torch.cat(ys, dim=0)
    z_fwd = torch.cat(logits_fwd, dim=0)
    z_low = torch.cat(logits_low, dim=0)
    return x, y, z_fwd, z_low


def _build_class_to_indices(y: torch.Tensor) -> Dict[int, torch.Tensor]:
    out: Dict[int, torch.Tensor] = {}
    for c in torch.unique(y).tolist():
        c = int(c)
        idxs = torch.nonzero(y == c, as_tuple=False).view(-1)
        out[c] = idxs
    return out


def _sample_same_class_index(
    *,
    class_to_indices: Dict[int, torch.Tensor],
    cls: int,
    avoid_idx: int,
    rng: torch.Generator,
) -> int:
    idxs = class_to_indices.get(int(cls))
    if idxs is None or int(idxs.numel()) <= 1:
        return int(avoid_idx)
    pos = int(torch.randint(low=0, high=int(idxs.numel()), size=(1,), generator=rng).item())
    j = int(idxs[pos].item())
    if j == int(avoid_idx):
        j = int(idxs[(pos + 1) % int(idxs.numel())].item())
    return j


@torch.no_grad()
def _mixup_and_forward_from_low(
    *,
    model: CNNCifar,
    x_pool: torch.Tensor,
    y_pool: torch.Tensor,
    device: str,
    lam: float,
    p: float,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    对全部样本做同类 mixup，并返回：
      - x_mix: (N,D) mixup 后 low
      - logits_mix: (N,C) 用 forward_from_low(x_mix)
      - pair_index: (N,)  被混入的同类样本索引（本地池内 index；无 mix 时为 -1）
      - mix_mask: (N,)    是否实际应用了 mixup
      - y: (N,)
    """
    class_to_indices = _build_class_to_indices(y_pool)
    rng = torch.Generator()
    rng.manual_seed(int(seed))

    lam = float(lam)
    p = float(p)
    n = int(y_pool.numel())
    pair_index = torch.full((n,), -1, dtype=torch.long)
    mix_mask = torch.zeros((n,), dtype=torch.bool)

    x_mix = x_pool.clone()
    for i_pool in range(n):
        cls = int(y_pool[i_pool].item())
        do_mix = (p > 0.0) and (float(torch.rand((), generator=rng).item()) < p) and (lam not in (0.0, 1.0))
        j_pool = i_pool
        if do_mix:
            j_pool = _sample_same_class_index(class_to_indices=class_to_indices, cls=cls, avoid_idx=i_pool, rng=rng)
        if j_pool != i_pool:
            x_mix[i_pool] = lam * x_pool[i_pool] + (1.0 - lam) * x_pool[j_pool]
            pair_index[i_pool] = int(j_pool)
            mix_mask[i_pool] = True

    model.eval()
    # forward_from_low 分 batch，避免一次性塞太大
    bs = 2048  # 足够大但相对安全；主函数会用 args.batch_size 控制收集阶段
    logits_chunks: List[torch.Tensor] = []
    for start in range(0, n, bs):
        end = min(n, start + bs)
        logits_b, _lp2, _h2, _proj2 = model.forward_from_low(x_mix[start:end].to(device))
        logits_chunks.append(logits_b.detach().cpu().float())
    logits_mix = torch.cat(logits_chunks, dim=0)

    return x_mix.cpu(), logits_mix, pair_index, mix_mask, y_pool.cpu(), x_pool.cpu()


def main() -> int:
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("--stage1_ckpt_path", type=str, required=True, help="Stage-1 checkpoint path (best-wo.pt / latest.pt)")
    p.add_argument("--split_path", type=str, default=None, help="split.pkl path (if None, infer from ckpt meta/args)")
    p.add_argument(
        "--client_ids",
        type=str,
        default="all",
        help="Clients to process. Use 'all' or comma-separated list like '0,1,2'.",
    )
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out_dir", type=str, default=None, help="Output dir (default: analysis/local_mixup/<timestamp>)")

    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--mixup_lam", type=float, default=0.7)
    p.add_argument("--mixup_p", type=float, default=1.0)
    args = p.parse_args()

    if args.seed is None:
        args.seed = int(time.time_ns() % (2**31 - 1))

    device = _resolve_device(args.gpu)
    _seed_all(int(args.seed))

    base_out_dir = os.path.abspath(args.out_dir or _make_default_out_dir())
    os.makedirs(_maybe_win_long_path(base_out_dir), exist_ok=True)

    ckpt = load_checkpoint(args.stage1_ckpt_path, map_location="cpu")
    ckpt_meta = (ckpt.get("meta", {}) or {}) if isinstance(ckpt, dict) else {}
    ckpt_state = (ckpt.get("state", {}) or {}) if isinstance(ckpt, dict) else {}
    ckpt_args = (ckpt.get("args", {}) or {}) if isinstance(ckpt, dict) else {}

    dataset_name = str(ckpt_meta.get("dataset", ckpt_args.get("dataset", "cifar10"))).lower()
    if dataset_name != "cifar10":
        raise ValueError(f"This dumper currently supports cifar10/CNNCifar only. Got dataset={dataset_name}")

    num_users = int(ckpt_meta.get("num_users", ckpt_args.get("num_users", 20)))
    num_classes = int(ckpt_meta.get("num_classes", ckpt_args.get("num_classes", 10)))

    split_path = args.split_path or ckpt_meta.get("split_path", None) or ckpt_args.get("split_path", None)
    if not split_path or not os.path.exists(_maybe_win_long_path(split_path)):
        raise FileNotFoundError(f"split_path not found: {split_path}")
    split = load_split(split_path)
    user_groups = split.get("user_groups", None)
    if user_groups is None:
        raise KeyError("split missing key: user_groups")

    # Build CIFAR10 train dataset with the same transform used in lib.utils
    from torchvision import datasets  # local import for faster CLI parse
    from lib.utils import trans_cifar10_train  # noqa: E402

    data_dir = str(ckpt_args.get("data_dir", "../data/"))
    train_dataset = datasets.CIFAR10(
        root=os.path.join(data_dir, dataset_name),
        train=True,
        download=True,
        transform=trans_cifar10_train,
    )

    local_sd = ckpt_state.get("local_models_full_state_dicts", None)
    if not isinstance(local_sd, dict):
        raise ValueError("Stage-1 checkpoint missing state['local_models_full_state_dicts']")

    # Parse client list
    if str(args.client_ids).strip().lower() == "all":
        client_ids = list(range(int(num_users)))
    else:
        parts = [p.strip() for p in str(args.client_ids).split(",") if p.strip()]
        client_ids = [int(x) for x in parts]
        for cid in client_ids:
            if cid < 0 or cid >= int(num_users):
                raise ValueError(f"--client_ids contains out-of-range client id={cid} not in [0,{num_users})")

    for cid in client_ids:
        out_dir = os.path.join(base_out_dir, f"client_{int(cid):02d}")
        os.makedirs(_maybe_win_long_path(out_dir), exist_ok=True)

        key = cid if cid in local_sd else str(cid)
        if key not in local_sd:
            raise KeyError(f"local_models_full_state_dicts missing client id={cid}")

        model = CNNCifar(args=type("A", (), {"num_classes": num_classes})()).to(device)
        model.load_state_dict(local_sd[key], strict=True)
        model.eval()

        indices = user_groups[int(cid)] if isinstance(user_groups, dict) else user_groups[int(cid)]
        indices = [int(i) for i in indices]

        x_pool, y_pool, logits_fwd, logits_low = _collect_local_low_and_logits(
            model=model,
            dataset=train_dataset,
            indices=indices,
            device=device,
            batch_size=int(args.batch_size),
        )

        # Sanity check: forward(images) vs forward_from_low(low_raw)
        argmax_fwd = logits_fwd.argmax(dim=1)
        argmax_low = logits_low.argmax(dim=1)
        max_abs = (logits_fwd - logits_low).abs().max().item()
        mean_abs = (logits_fwd - logits_low).abs().mean().item()
        pred_disagree = float((argmax_fwd != argmax_low).float().mean().item())

        x_mix, logits_mix, pair_index, mix_mask, y_cpu, x_cpu = _mixup_and_forward_from_low(
            model=model,
            x_pool=x_pool,
            y_pool=y_pool,
            device=device,
            lam=float(args.mixup_lam),
            p=float(args.mixup_p),
            seed=int(args.seed) + 30000 + int(cid),
        )

        meta = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "script": "analysis/dump_local_low_logits_and_mixup.py",
            "out_dir": out_dir,
            "stage1_ckpt_path": os.path.abspath(args.stage1_ckpt_path),
            "split_path": os.path.abspath(split_path),
            "dataset": dataset_name,
            "num_users": int(num_users),
            "num_classes": int(num_classes),
            "client_id": int(cid),
            "device": device,
            "gpu": int(args.gpu),
            "seed": int(args.seed),
            "collect_cfg": {
                "batch_size": int(args.batch_size),
                "n_total": int(x_pool.shape[0]),
            },
            "mixup_cfg": {
                "lam": float(args.mixup_lam),
                "p": float(args.mixup_p),
                "n_mixed": int(mix_mask.long().sum().item()),
            },
            "sanity": {
                "forward_vs_forward_from_low": {
                    "max_abs": float(max_abs),
                    "mean_abs": float(mean_abs),
                    "pred_disagree_frac": float(pred_disagree),
                }
            },
        }

        # quick stats
        probs_o = torch.softmax(logits_fwd, dim=1)
        probs_m = torch.softmax(logits_mix, dim=1)
        conf_o = probs_o.max(dim=1).values.detach().cpu().numpy()
        conf_m = probs_m.max(dim=1).values.detach().cpu().numpy()
        pred_o = probs_o.argmax(dim=1)
        pred_m = probs_m.argmax(dim=1)
        meta["quick"] = {
            "pred_agree_frac": float((pred_o == pred_m).float().mean().item()),
            "conf_orig": _summarize_array(conf_o),
            "conf_mix": _summarize_array(conf_m),
        }

        dump = {
            "meta": meta,
            "pair_index_in_collect": pair_index,
            "mixup_mask": mix_mask,
            "y": y_cpu,
            "x_low": x_cpu,
            "x_mix": x_mix,
            "logits_orig_from_forward": logits_fwd,
            "logits_forward_from_low_on_orig": logits_low,
            "logits_mix": logits_mix,
        }

        out_pt = os.path.join(out_dir, "dump.pt")
        torch.save(dump, _maybe_win_long_path(out_pt))
        out_meta = os.path.join(out_dir, "meta.json")
        with open(_maybe_win_long_path(out_meta), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        print(
            f"[done][cid={cid:02d}] wrote dump.pt/meta.json | "
            f"n={int(x_pool.shape[0])} mixed={int(mix_mask.long().sum().item())} "
            f"sanity_pred_disagree={pred_disagree:.2%}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


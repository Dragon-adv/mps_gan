#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
M1 离线评估（CIFAR-10 + CNNCifar）

目标：
- 用每个 client 的 train split 计算 per-class acc
- 每类选 Top-K teachers (K=3)
- 基于 teachers 的真实 low-level 特征拟合一个 “共享”的 class-conditional 分布（默认对角高斯）
- 合成 low 特征并在 teachers 的后半段（fc0/fc1/fc2）上跑，输出：
  - teacher-consensus 指标
  - high-level 特征质量/多样性指标（在 high_norm 空间）

用法示例：
  python exps/run_m1_eval.py --stage1_ckpt_path "<LOGDIR>/stage1/ckpts/best-wo.pt"
"""

import argparse
import csv
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

# 将项目根目录添加到 sys.path（跟 federated_main.py 保持一致风格）
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

lib_dir = (Path(__file__).parent / ".." / "lib").resolve()
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))
mod_dir = (Path(__file__).parent / ".." / "lib" / "models").resolve()
if str(mod_dir) not in sys.path:
    sys.path.insert(0, str(mod_dir))

from lib.checkpoint import load_checkpoint
from lib.m1_eval import (
    build_teacher_map,
    compute_n_syn,
    extract_high_by_class,
    extract_low_by_class,
    fit_gaussian_diag,
    forward_from_low_cnncifar,
    per_class_correct_total,
    save_json,
    set_all_seeds,
)
from lib.models.models import CNNCifar
from lib.split_manager import load_split
from lib.update import DatasetSplit
from lib.utils import get_dataset


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


def _resolve_device(gpu: int) -> str:
    if torch.cuda.is_available() and gpu is not None and int(gpu) >= 0:
        torch.cuda.set_device(int(gpu))
        return "cuda"
    return "cpu"


def _write_csv(path: str, header: List[str], rows: List[List[object]]) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--stage1_ckpt_path", type=str, required=True, help="Stage-1 checkpoint path (best-wo.pt / latest.pt)")
    p.add_argument("--split_path", type=str, default=None, help="split.pkl 路径；不填则尝试从 ckpt meta 推断")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=1234)

    # teacher selection
    p.add_argument("--k_teachers", type=int, default=3)
    p.add_argument("--n_real_min", type=int, default=20, help="选 teacher 时，每类最少样本数门槛")

    # synthesis budget (策略2)
    p.add_argument("--n_syn_min", type=int, default=200)
    p.add_argument("--n_syn_max", type=int, default=2000)
    p.add_argument("--syn_r", type=float, default=1.0, help="N_syn = clamp(r * n_real)")

    # loaders / extraction
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--max_feat_per_class_teacher", type=int, default=5000, help="每个 teacher 每类最多抽取多少真实特征用于拟合分布")

    # output
    p.add_argument("--out_dir", type=str, default=None, help="输出目录；默认 <logdir>/m1_eval")

    args_cli = p.parse_args()
    set_all_seeds(int(args_cli.seed))

    payload = load_checkpoint(args_cli.stage1_ckpt_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid checkpoint payload type: {type(payload)}")
    ckpt_meta = payload.get("meta", {}) or {}
    ckpt_args = payload.get("args", {}) or {}
    ckpt_state = payload.get("state", {}) or {}

    # 强制 CIFAR-10 + CNNCifar 的 M1 起步（后续要扩展再做）
    if str(ckpt_args.get("dataset", "")) != "cifar10":
        raise ValueError(f"M1 当前脚本只支持 cifar10；ckpt.dataset={ckpt_args.get('dataset')}")

    # 构建 args namespace 给 get_dataset 使用（尽量复用 ckpt 的设置）
    base = dict(ckpt_args)
    base["dataset"] = "cifar10"
    base["num_classes"] = int(base.get("num_classes", 10))
    base["num_users"] = int(base.get("num_users", 20))
    base["device"] = _resolve_device(args_cli.gpu)
    base["gpu"] = int(args_cli.gpu)
    args = SimpleNamespace(**base)

    logdir = _infer_logdir_from_ckpt(args_cli.stage1_ckpt_path, ckpt_meta)
    out_dir = args_cli.out_dir or os.path.join(logdir, "m1_eval")
    os.makedirs(out_dir, exist_ok=True)

    split_path = args_cli.split_path or ckpt_meta.get("split_path", None) or getattr(args, "split_path", None)
    if not split_path:
        raise ValueError("未提供 split_path，且无法从 ckpt meta/args 推断。请显式传 --split_path")
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"split.pkl 不存在：{split_path}")

    split = load_split(split_path)
    n_list = split["n_list"]
    k_list = split["k_list"]
    user_groups = split["user_groups"]  # train split idxs

    train_dataset, test_dataset, _, _, _, _ = get_dataset(args, n_list, k_list)

    local_sd = ckpt_state.get("local_models_full_state_dicts", None)
    if not isinstance(local_sd, dict):
        raise ValueError("Checkpoint missing state['local_models_full_state_dicts']")

    # rebuild local models
    local_models: List[torch.nn.Module] = []
    for cid in range(args.num_users):
        m = CNNCifar(args=args).to(args.device)
        key = cid if cid in local_sd else str(cid)
        if key not in local_sd:
            raise KeyError(f"local_models_full_state_dicts missing client id={cid}")
        m.load_state_dict(local_sd[key], strict=True)
        m.eval()
        local_models.append(m)

    # 1) per-client per-class acc on LOCAL TRAIN split
    num_clients = args.num_users
    num_classes = args.num_classes
    correct_mat = torch.zeros((num_clients, num_classes), dtype=torch.long)
    total_mat = torch.zeros((num_clients, num_classes), dtype=torch.long)

    for cid in range(num_clients):
        dl = DataLoader(
            DatasetSplit(train_dataset, user_groups[cid]),
            batch_size=int(args_cli.batch_size),
            shuffle=False,
            num_workers=int(args_cli.num_workers),
            drop_last=False,
        )
        corr, tot = per_class_correct_total(local_models[cid], dl, num_classes=num_classes, device=args.device)
        correct_mat[cid] = corr
        total_mat[cid] = tot

    teacher_map = build_teacher_map(
        correct_mat=correct_mat,
        total_mat=total_mat,
        k_teachers=int(args_cli.k_teachers),
        n_real_min=int(args_cli.n_real_min),
    )

    save_json(os.path.join(out_dir, "teacher_map.json"), teacher_map)

    # teacher_scores.csv：完整 per-client per-class acc
    score_rows: List[List[object]] = []
    for cid in range(num_clients):
        for c in range(num_classes):
            n_real = int(total_mat[cid, c].item())
            acc = float(correct_mat[cid, c].item()) / float(max(1, n_real))
            score_rows.append([cid, c, n_real, f"{acc:.6f}"])
    _write_csv(
        os.path.join(out_dir, "teacher_scores.csv"),
        header=["client_id", "class_id", "n_real", "acc"],
        rows=score_rows,
    )

    # 2) 为每个 class 用 teachers 的真实 low 特征拟合一个“共享”分布，并生成 pool_c（大小 n_syn_max）
    # 同时缓存 teachers 的真实 high_mean（用于 mean_dist_high）
    n_syn_max = int(args_cli.n_syn_max)
    syn_pools: Dict[int, torch.Tensor] = {}
    teacher_high_mean: Dict[Tuple[int, int], torch.Tensor] = {}  # (teacher_id, class_id) -> (D_high,)

    # 预先为所有 teacher 抽特征（避免重复跑 dataloader 太多次）
    all_teacher_ids = sorted({tid for tids in teacher_map.values() for tid in tids})
    teacher_low_by_class: Dict[int, List[torch.Tensor]] = {}
    teacher_high_by_class: Dict[int, List[torch.Tensor]] = {}

    for tid in all_teacher_ids:
        dl_t = DataLoader(
            DatasetSplit(train_dataset, user_groups[tid]),
            batch_size=int(args_cli.batch_size),
            shuffle=True,
            num_workers=int(args_cli.num_workers),
            drop_last=False,
        )
        teacher_low_by_class[tid] = extract_low_by_class(
            local_models[tid],
            dl_t,
            num_classes=num_classes,
            device=args.device,
            max_per_class=int(args_cli.max_feat_per_class_teacher),
        )
        teacher_high_by_class[tid] = extract_high_by_class(
            local_models[tid],
            dl_t,
            num_classes=num_classes,
            device=args.device,
            max_per_class=int(args_cli.max_feat_per_class_teacher),
            use_norm=True,
        )
        for c in range(num_classes):
            h = teacher_high_by_class[tid][c]
            if h.numel() > 0:
                teacher_high_mean[(tid, c)] = h.mean(dim=0)

    # 合成 pool
    g = torch.Generator()
    g.manual_seed(int(args_cli.seed) + 999)
    for c in range(num_classes):
        tids = teacher_map.get(c, [])
        feats = []
        for tid in tids:
            x = teacher_low_by_class[tid][c]
            if x.numel() > 0:
                feats.append(x)
        if len(feats) == 0:
            # 极端兜底：标准正态
            syn_pools[c] = torch.randn((n_syn_max, 400), generator=g)
            continue
        x_all = torch.cat(feats, dim=0)
        gauss = fit_gaussian_diag(x_all, eps=1e-6)
        syn_pools[c] = gauss.sample(n_syn_max, generator=g).cpu()

    # 3) 离线评估：对每个 (client, class) 取 N_syn(i,c) 个合成 low，跑 teachers tail
    report_rows: List[List[object]] = []
    for cid in range(num_clients):
        for c in range(num_classes):
            n_real = int(total_mat[cid, c].item())
            n_syn = compute_n_syn(
                n_real=n_real,
                n_syn_min=int(args_cli.n_syn_min),
                n_syn_max=int(args_cli.n_syn_max),
                r=float(args_cli.syn_r),
            )

            tids = teacher_map.get(c, [])
            if len(tids) == 0:
                continue

            syn_low = syn_pools[c][:n_syn].to(args.device)

            logits_list = []
            pred_list = []
            high_list = []
            for tid in tids:
                logits, high = forward_from_low_cnncifar(local_models[tid], syn_low)
                preds = torch.argmax(logits, dim=1)
                logits_list.append(logits.detach())
                pred_list.append(preds.detach())
                high_list.append(high.detach())

            # per-teacher acc
            acc_ts = [float((pred == c).float().mean().item()) for pred in pred_list]
            # ensemble acc
            logits_ens = torch.stack(logits_list, dim=0).mean(dim=0)
            pred_ens = torch.argmax(logits_ens, dim=1)
            acc_ens = float((pred_ens == c).float().mean().item())

            # disagree rate
            preds_stack = torch.stack(pred_list, dim=0)  # (K, B)
            disagree = float((preds_stack != preds_stack[0:1]).any(dim=0).float().mean().item())

            # high mean distance: avg_t || mean(high_syn_t) - mean(high_real_t) ||
            dists = []
            traces = []
            for k, tid in enumerate(tids):
                high_syn = high_list[k]
                high_syn_mean = high_syn.mean(dim=0)
                high_real_mean = teacher_high_mean.get((tid, c), None)
                if high_real_mean is not None:
                    d = torch.norm(high_syn_mean - high_real_mean.to(args.device), p=2).item()
                    dists.append(float(d))
                # trace cov (diversity proxy)
                if high_syn.shape[0] >= 2:
                    centered = high_syn - high_syn_mean.view(1, -1)
                    var = (centered * centered).mean(dim=0)
                    traces.append(float(var.sum().item()))
            mean_dist_high = float(np.mean(dists)) if len(dists) > 0 else float("nan")
            trace_cov_high_syn = float(np.mean(traces)) if len(traces) > 0 else float("nan")

            report_rows.append([
                cid,
                c,
                n_real,
                n_syn,
                "|".join(str(t) for t in tids),
                f"{acc_ens:.6f}",
                f"{acc_ts[0]:.6f}" if len(acc_ts) > 0 else "",
                f"{acc_ts[1]:.6f}" if len(acc_ts) > 1 else "",
                f"{acc_ts[2]:.6f}" if len(acc_ts) > 2 else "",
                f"{disagree:.6f}",
                f"{mean_dist_high:.6f}" if not math.isnan(mean_dist_high) else "nan",
                f"{trace_cov_high_syn:.6f}" if not math.isnan(trace_cov_high_syn) else "nan",
            ])

    _write_csv(
        os.path.join(out_dir, "report.csv"),
        header=[
            "client_id",
            "class_id",
            "n_real",
            "n_syn",
            "teachers",
            "acc_ens",
            "acc_t1",
            "acc_t2",
            "acc_t3",
            "disagree",
            "mean_dist_high",
            "trace_cov_high_syn",
        ],
        rows=report_rows,
    )

    print(f"[M1] Done. Outputs saved to: {out_dir}")
    print(f"[M1] teacher_map.json, teacher_scores.csv, report.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



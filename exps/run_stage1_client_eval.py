#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
离线评估：加载 Stage-1 训练出的每个客户端模型（local_models_full_state_dicts），
并复用 Stage-1 保存的 split（split.pkl），保证模型与数据划分一致。

输出：
- 每个客户端在"自己拥有的类（classes_list[cid]）"上的每个类的详细指标：
  - 每个类的样本数量（train/test 两套 split）
  - 每个类的准确率和损失（同时在 test 集和所有本地数据上评估）
- CSV 格式：每行代表一个客户端的一个类，包含 test 和 all 两套指标
  (client_id, class_id, n_train, n_test, acc_test, loss_ce_test, n_all, acc_all, loss_ce_all)

用法示例：
  python exps/run_stage1_client_eval.py --stage1_ckpt_path "<LOGDIR>/stage1_ckpts/best-wo.pt"
"""

import argparse
import csv
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 将项目根目录添加到 sys.path（跟 federated_main.py / run_m1_eval.py 保持一致）
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

lib_dir = (Path(__file__).parent / ".." / "lib").resolve()
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))
mod_dir = (Path(__file__).parent / ".." / "lib" / "models").resolve()
if str(mod_dir) not in sys.path:
    sys.path.insert(0, str(mod_dir))

from lib.checkpoint import load_checkpoint  # noqa: E402
from lib.models.models import (  # noqa: E402
    CNNCifar,
    CNNFashion_Mnist,
    CNNFemnist,
    CNNMnist,
    ModelCT,
    ResNetWithFeatures,
)
from lib.split_manager import load_split  # noqa: E402
from lib.update import DatasetSplit  # noqa: E402
from lib.utils import get_dataset  # noqa: E402


def _infer_logdir_from_ckpt(stage1_ckpt_path: str, ckpt_meta: dict) -> str:
    meta_logdir = (ckpt_meta or {}).get("logdir", None)
    if meta_logdir:
        return meta_logdir
    ckpt_abs = os.path.abspath(stage1_ckpt_path)
    ckpt_dir = os.path.dirname(ckpt_abs)
    if os.path.basename(ckpt_dir) == "stage1_ckpts":
        return os.path.dirname(ckpt_dir)
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


def _to_int_list(x: object) -> List[int]:
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return [int(i) for i in x]
    # e.g., numpy array / torch tensor
    try:
        return [int(i) for i in list(x)]
    except Exception:
        return [int(x)]


def _build_model_for_client(args: SimpleNamespace, cid: int) -> torch.nn.Module:
    # 逻辑尽量与 exps/federated_main.py Stage-2 的模型重建一致
    if args.dataset == "mnist":
        if getattr(args, "mode", "") == "model_heter":
            if cid < 7:
                args.out_channels = 18
            elif cid < 14:
                args.out_channels = 20
            else:
                args.out_channels = 22
        else:
            args.out_channels = 20
        return CNNMnist(args=args)

    if args.dataset == "femnist":
        if getattr(args, "mode", "") == "model_heter":
            if cid < 7:
                args.out_channels = 18
            elif cid < 14:
                args.out_channels = 20
            else:
                args.out_channels = 22
        else:
            args.out_channels = 20
        return CNNFemnist(args=args)

    if args.dataset in {"cifar10", "cifar100", "flowers", "defungi", "realwaste"}:
        return CNNCifar(args=args)

    if args.dataset == "tinyimagenet":
        # tinyimagenet: 200 classes
        args.num_classes = int(getattr(args, "num_classes", 200))
        return ModelCT(out_dim=256, n_classes=args.num_classes)

    if args.dataset == "fashion":
        return CNNFashion_Mnist(args=args)

    if args.dataset == "imagenet":
        return ResNetWithFeatures(base="resnet18", num_classes=args.num_classes)

    raise ValueError(f"Unsupported dataset: {args.dataset}")


@torch.no_grad()
def _count_labels_from_loader(dl: DataLoader, num_classes: int, device: str = "cpu") -> torch.Tensor:
    """
    返回 shape=(num_classes,) 的计数（在 CPU 上）。
    """
    counts = torch.zeros(num_classes, dtype=torch.long)
    for _, y in dl:
        y = y.to(device)
        y = y.view(-1).long()
        c = torch.bincount(y.detach().cpu(), minlength=num_classes)
        counts += c
    return counts


@torch.no_grad()
def eval_owned_classes_on_loader(
    model: torch.nn.Module,
    dl: DataLoader,
    owned_classes: Set[int],
    num_classes: int,
    device: str,
) -> Tuple[int, Dict[int, int], float, float, Dict[int, float], Dict[int, float]]:
    """
    在一个 dataloader 上，仅对 owned_classes 的样本统计：
    - total_owned：owned_classes 总样本数
    - per_class_owned：dict[class_id] -> count（仅 owned_classes）
    - acc_owned：准确率（所有类的平均值）
    - loss_ce_owned：CrossEntropyLoss(logits, y)（所有类的平均值）
    - per_class_acc：dict[class_id] -> accuracy（每个类的准确率）
    - per_class_loss：dict[class_id] -> loss（每个类的CE损失）
    """
    model.eval()
    ce = nn.CrossEntropyLoss(reduction="none").to(device)  # 改为 reduction="none" 以便按类统计

    total = 0
    correct = 0
    loss_sum = 0.0

    # 按类统计：样本数、正确数、损失和
    per_class_count = torch.zeros(num_classes, dtype=torch.long)
    per_class_correct = torch.zeros(num_classes, dtype=torch.long)
    per_class_loss_sum = torch.zeros(num_classes, dtype=torch.float32)
    
    owned_tensor = None  # lazy

    for x, y in dl:
        x = x.to(device)
        y = y.to(device).view(-1).long()

        if owned_tensor is None:
            owned_tensor = torch.tensor(sorted(list(owned_classes)), device=device, dtype=torch.long)

        # mask: y in owned_classes
        # (num_owned may be small; torch.isin is fine here)
        mask = torch.isin(y, owned_tensor)
        if mask.sum().item() == 0:
            continue

        y_m = y[mask]
        out = model(x)
        if isinstance(out, (tuple, list)) and len(out) >= 1:
            logits = out[0]
        else:
            logits = out
        logits_m = logits[mask]

        # 计算每个样本的损失
        loss_per_sample = ce(logits_m, y_m)
        loss_sum += float(loss_per_sample.sum().item())
        
        pred = torch.argmax(logits_m, dim=1)
        correct_mask = (pred == y_m)
        correct += int(correct_mask.sum().item())
        total += int(y_m.numel())

        # 按类统计（只统计当前batch中出现的类，提高效率）
        unique_classes = torch.unique(y_m)
        for cls_id in unique_classes:
            cls_id_int = int(cls_id.item())
            if cls_id_int in owned_classes:
                cls_mask = (y_m == cls_id_int)
                cls_count = int(cls_mask.sum().item())
                cls_correct = int(correct_mask[cls_mask].sum().item())
                cls_loss = float(loss_per_sample[cls_mask].sum().item())
                
                per_class_count[cls_id_int] += cls_count
                per_class_correct[cls_id_int] += cls_correct
                per_class_loss_sum[cls_id_int] += cls_loss

    per_class_owned = {int(c): int(per_class_count[c].item()) for c in sorted(list(owned_classes))}
    acc = float(correct) / float(total) if total > 0 else float("nan")
    loss_ce = float(loss_sum) / float(total) if total > 0 else float("nan")
    
    # 计算每个类的准确率和损失
    per_class_acc = {}
    per_class_loss = {}
    for cls_id in sorted(list(owned_classes)):
        cls_count = int(per_class_count[cls_id].item())
        if cls_count > 0:
            cls_correct = int(per_class_correct[cls_id].item())
            cls_loss_val = float(per_class_loss_sum[cls_id].item())
            per_class_acc[cls_id] = float(cls_correct) / float(cls_count)
            per_class_loss[cls_id] = float(cls_loss_val) / float(cls_count)
        else:
            per_class_acc[cls_id] = float("nan")
            per_class_loss[cls_id] = float("nan")
    
    return total, per_class_owned, acc, loss_ce, per_class_acc, per_class_loss


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--stage1_ckpt_path", type=str, required=True, help="Stage-1 checkpoint path (best-wo.pt / latest.pt)")
    p.add_argument("--split_path", type=str, default=None, help="split.pkl 路径；不填则尝试从 ckpt meta 推断")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--out_dir", type=str, default=None, help="输出目录；默认 <logdir>/stage1_client_eval")
    args_cli = p.parse_args()

    payload = load_checkpoint(args_cli.stage1_ckpt_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid checkpoint payload type: {type(payload)}")
    ckpt_meta = payload.get("meta", {}) or {}
    ckpt_args = payload.get("args", {}) or {}
    ckpt_state = payload.get("state", {}) or {}

    base = dict(ckpt_args)
    base["device"] = _resolve_device(args_cli.gpu)
    base["gpu"] = int(args_cli.gpu)
    # 一些脚本运行必要的默认值兜底
    base.setdefault("data_dir", "../data/")
    base.setdefault("num_users", int(ckpt_meta.get("num_users", base.get("num_users", 0))))
    base.setdefault("num_classes", int(ckpt_meta.get("num_classes", base.get("num_classes", 10))))
    base.setdefault("dataset", str(ckpt_meta.get("dataset", base.get("dataset", "mnist"))))
    args = SimpleNamespace(**base)

    logdir = _infer_logdir_from_ckpt(args_cli.stage1_ckpt_path, ckpt_meta)
    out_dir = args_cli.out_dir or os.path.join(logdir, "stage1_client_eval")
    os.makedirs(out_dir, exist_ok=True)

    split_path = args_cli.split_path or ckpt_meta.get("split_path", None) or getattr(args, "split_path", None)
    if not split_path:
        raise ValueError("未提供 split_path，且无法从 ckpt meta/args 推断。请显式传 --split_path")
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"split.pkl 不存在：{split_path}")

    split = load_split(split_path)
    n_list = split["n_list"]
    k_list = split["k_list"]
    user_groups = split["user_groups"]          # train idxs
    user_groups_lt = split["user_groups_lt"]    # test idxs
    classes_list = split.get("classes_list", None)

    train_dataset, test_dataset, _, _, _, _ = get_dataset(args, n_list, k_list)

    local_sd = ckpt_state.get("local_models_full_state_dicts", None)
    if not isinstance(local_sd, dict):
        raise ValueError("Checkpoint missing state['local_models_full_state_dicts']")

    # rebuild local models
    local_models: List[torch.nn.Module] = []
    for cid in range(int(args.num_users)):
        m = _build_model_for_client(args, cid).to(args.device)
        key = cid if cid in local_sd else str(cid)
        if key not in local_sd:
            raise KeyError(f"local_models_full_state_dicts missing client id={cid}")
        m.load_state_dict(local_sd[key], strict=True)
        m.eval()
        local_models.append(m)

    num_clients = int(args.num_users)
    num_classes = int(args.num_classes)

    rows: List[List[object]] = []
    for cid in range(num_clients):
        # owned classes
        if classes_list is not None:
            owned = set(_to_int_list(classes_list[cid]))
        else:
            # fallback：从 train split 的真实标签推断该 client "拥有"的类
            dl_tmp = DataLoader(
                DatasetSplit(train_dataset, user_groups[cid]),
                batch_size=int(args_cli.batch_size),
                shuffle=False,
                num_workers=int(args_cli.num_workers),
                drop_last=False,
            )
            cnt = _count_labels_from_loader(dl_tmp, num_classes=num_classes, device=args.device)
            owned = set(torch.nonzero(cnt > 0, as_tuple=False).view(-1).tolist())

        owned_sorted = sorted(list(owned))

        # train owned counts (always compute for statistics)
        dl_train = DataLoader(
            DatasetSplit(train_dataset, user_groups[cid]),
            batch_size=int(args_cli.batch_size),
            shuffle=False,
            num_workers=int(args_cli.num_workers),
            drop_last=False,
        )
        train_total_owned, train_per_class, _, _, _, _ = eval_owned_classes_on_loader(
            model=local_models[cid],
            dl=dl_train,
            owned_classes=owned,
            num_classes=num_classes,
            device=args.device,
        )

        # 1. 在 test 集上评估
        dl_test = DataLoader(
            DatasetSplit(test_dataset, user_groups_lt[cid]),
            batch_size=int(args_cli.batch_size),
            shuffle=False,
            num_workers=int(args_cli.num_workers),
            drop_last=False,
        )
        test_total_owned, test_per_class, acc_test, loss_ce_test, test_per_class_acc, test_per_class_loss = eval_owned_classes_on_loader(
            model=local_models[cid],
            dl=dl_test,
            owned_classes=owned,
            num_classes=num_classes,
            device=args.device,
        )

        # 2. 在所有本地数据（train+test）上评估
        train_idxs = user_groups[cid]
        test_idxs = user_groups_lt[cid]
        
        # 确保索引是列表格式（user_groups 可能是 numpy 数组）
        if isinstance(train_idxs, np.ndarray):
            train_idxs = train_idxs.tolist()
        if isinstance(test_idxs, np.ndarray):
            test_idxs = test_idxs.tolist()
        
        dl_train_eval = DataLoader(
            DatasetSplit(train_dataset, train_idxs),
            batch_size=int(args_cli.batch_size),
            shuffle=False,
            num_workers=int(args_cli.num_workers),
            drop_last=False,
        )
        dl_test_eval = DataLoader(
            DatasetSplit(test_dataset, test_idxs),
            batch_size=int(args_cli.batch_size),
            shuffle=False,
            num_workers=int(args_cli.num_workers),
            drop_last=False,
        )
        
        # 分别评估 train 和 test，然后合并统计
        train_total, train_per_class_eval, _, _, train_per_class_acc, train_per_class_loss = eval_owned_classes_on_loader(
            model=local_models[cid],
            dl=dl_train_eval,
            owned_classes=owned,
            num_classes=num_classes,
            device=args.device,
        )
        test_total_for_all, test_per_class_eval, _, _, test_per_class_acc, test_per_class_loss = eval_owned_classes_on_loader(
            model=local_models[cid],
            dl=dl_test_eval,
            owned_classes=owned,
            num_classes=num_classes,
            device=args.device,
        )
        
        # 合并统计结果：直接合并样本数和正确数、损失和
        all_total_owned = train_total + test_total_for_all
        all_per_class = {}
        all_per_class_acc = {}
        all_per_class_loss = {}
        
        # 重新计算每个类的正确数和损失和（更准确）
        per_class_correct_combined = {}
        per_class_loss_sum_combined = {}
        
        for cls_id in owned_sorted:
            train_count = train_per_class_eval.get(cls_id, 0)
            test_count = test_per_class_eval.get(cls_id, 0)
            all_per_class[cls_id] = train_count + test_count
            
            # 计算合并后的正确数和损失和
            train_acc = train_per_class_acc.get(cls_id, float("nan"))
            test_acc = test_per_class_acc.get(cls_id, float("nan"))
            train_loss = train_per_class_loss.get(cls_id, float("nan"))
            test_loss = test_per_class_loss.get(cls_id, float("nan"))
            
            train_correct = int(train_acc * train_count) if not np.isnan(train_acc) and train_count > 0 else 0
            test_correct = int(test_acc * test_count) if not np.isnan(test_acc) and test_count > 0 else 0
            per_class_correct_combined[cls_id] = train_correct + test_correct
            
            train_loss_sum_val = train_loss * train_count if not np.isnan(train_loss) and train_count > 0 else 0.0
            test_loss_sum_val = test_loss * test_count if not np.isnan(test_loss) and test_count > 0 else 0.0
            per_class_loss_sum_combined[cls_id] = train_loss_sum_val + test_loss_sum_val
            
            # 计算合并后的准确率和损失
            total_count = train_count + test_count
            if total_count > 0:
                all_per_class_acc[cls_id] = float(per_class_correct_combined[cls_id]) / float(total_count)
                all_per_class_loss[cls_id] = float(per_class_loss_sum_combined[cls_id]) / float(total_count)
            else:
                all_per_class_acc[cls_id] = float("nan")
                all_per_class_loss[cls_id] = float("nan")
        
        # 计算总体准确率和损失
        total_correct = sum(per_class_correct_combined.values())
        total_loss_sum = sum(per_class_loss_sum_combined.values())
        acc_all = float(total_correct) / float(all_total_owned) if all_total_owned > 0 else float("nan")
        loss_ce_all = float(total_loss_sum) / float(all_total_owned) if all_total_owned > 0 else float("nan")

        # 为每个类生成一行数据（包含 test 和 all 两套指标）
        for cls_id in owned_sorted:
            n_train_cls = train_per_class.get(cls_id, 0)
            n_test_cls = test_per_class.get(cls_id, 0)
            n_all_cls = all_per_class.get(cls_id, 0)
            
            acc_test_cls = test_per_class_acc.get(cls_id, float("nan"))
            loss_test_cls = test_per_class_loss.get(cls_id, float("nan"))
            acc_all_cls = all_per_class_acc.get(cls_id, float("nan"))
            loss_all_cls = all_per_class_loss.get(cls_id, float("nan"))
            
            rows.append([
                cid,
                int(cls_id),
                int(n_train_cls),
                int(n_test_cls),
                f"{acc_test_cls:.6f}" if not np.isnan(acc_test_cls) else "nan",
                f"{loss_test_cls:.6f}" if not np.isnan(loss_test_cls) else "nan",
                int(n_all_cls),
                f"{acc_all_cls:.6f}" if not np.isnan(acc_all_cls) else "nan",
                f"{loss_all_cls:.6f}" if not np.isnan(loss_all_cls) else "nan",
            ])

        # 打印汇总信息
        print(
            f"[client {cid:02d}] owned={owned_sorted} | "
            f"n_train={train_total_owned} n_test={test_total_owned} n_all={all_total_owned} | "
            f"acc_test={acc_test:.4f} loss_ce_test={loss_ce_test:.4f} | "
            f"acc_all={acc_all:.4f} loss_ce_all={loss_ce_all:.4f}"
        )
        # 打印每个类的详细信息
        for cls_id in owned_sorted:
            n_test_cls = test_per_class.get(cls_id, 0)
            n_all_cls = all_per_class.get(cls_id, 0)
            acc_test_cls = test_per_class_acc.get(cls_id, float("nan"))
            loss_test_cls = test_per_class_loss.get(cls_id, float("nan"))
            acc_all_cls = all_per_class_acc.get(cls_id, float("nan"))
            loss_all_cls = all_per_class_loss.get(cls_id, float("nan"))
            
            acc_test_str = f"{acc_test_cls:.4f}" if not np.isnan(acc_test_cls) else "nan"
            loss_test_str = f"{loss_test_cls:.4f}" if not np.isnan(loss_test_cls) else "nan"
            acc_all_str = f"{acc_all_cls:.4f}" if not np.isnan(acc_all_cls) else "nan"
            loss_all_str = f"{loss_all_cls:.4f}" if not np.isnan(loss_all_cls) else "nan"
            
            print(f"  class {cls_id:2d}: n_test={n_test_cls:4d} acc_test={acc_test_str:>8s} loss_test={loss_test_str:>8s} | "
                  f"n_all={n_all_cls:4d} acc_all={acc_all_str:>8s} loss_all={loss_all_str:>8s}")

    # 生成包含 test 和 all 两套指标的 CSV 文件
    out_csv = os.path.join(out_dir, "client_owned_metrics.csv")
    
    _write_csv(
        out_csv,
        header=[
            "client_id",
            "class_id",
            "n_train",
            "n_test",
            "acc_test",
            "loss_ce_test",
            "n_all",
            "acc_all",
            "loss_ce_all",
        ],
        rows=rows,
    )

    print(f"[OK] Saved: {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



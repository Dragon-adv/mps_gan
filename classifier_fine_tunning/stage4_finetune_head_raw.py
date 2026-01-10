from __future__ import annotations

"""
Stage-4 (raw-high): finetune classifier head (fc1+fc2) using SAFS synthetic raw-high features.

Default behavior:
  - For each client, load Stage-1 model weights
  - Freeze everything except fc1+fc2
  - Train on synthetic features (optionally clamp to non-negative, then normalize before fc1)
  - Evaluate on the client's real test split (images) using full forward()
  - Save per-client head checkpoint + a summary json
"""

import argparse
import json
import os
import sys
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Subset

from utils import (
    collect_env_info,
    ensure_repo_root_on_path,
    eval_model_on_images,
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
from lib.split_manager import load_split
from lib.utils import get_dataset


def _build_client_model(args: SimpleNamespace, client_id: int) -> torch.nn.Module:
    # Keep consistent with Stage-1/Stage-2 reconstruction logic
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
    raise ValueError(f"Unsupported dataset for Stage-4(raw-head finetune): {args.dataset}")


def _extract_syn_xy(syn_dataset: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
    syn_z = torch.concat([d["synthetic_features"] for d in syn_dataset], dim=0)
    syn_y = torch.concat(
        [
            torch.full((d["synthetic_features"].size(0),), int(d["class_index"]), dtype=torch.long)
            for d in syn_dataset
        ],
        dim=0,
    )
    return syn_z, syn_y


@torch.no_grad()
def _infer_seen_classes_from_indices(
    dataset,
    indices: list[int],
    *,
    device: str,
    num_classes: int,
    batch_size: int = 1024,
) -> list[int]:
    """
    Infer which class labels appear in the given dataset indices.

    This is used to optionally filter synthetic features per-client to only the
    classes the client has seen locally (based on split user_groups).
    """
    if not indices:
        return []

    # Subset + loader to avoid loading all at once
    dl = DataLoader(Subset(dataset, indices), batch_size=int(batch_size), shuffle=False, num_workers=0)
    seen = torch.zeros(int(num_classes), dtype=torch.bool, device=device)
    for batch in dl:
        # dataset __getitem__ is expected to return (x, y) or (x, y, ...)
        if not isinstance(batch, (list, tuple)) or len(batch) < 2:
            raise ValueError("Unexpected dataset batch structure when inferring seen classes.")
        y = batch[1]
        if torch.is_tensor(y):
            y = y.to(device).long().view(-1)
        else:
            y = torch.as_tensor(y, device=device).long().view(-1)
        y = y[(y >= 0) & (y < int(num_classes))]
        if y.numel() > 0:
            seen[y.unique()] = True
    return [int(i) for i in torch.nonzero(seen, as_tuple=False).view(-1).tolist()]


def _forward_head_only(model: torch.nn.Module, x_high_raw: torch.Tensor, *, num_classes: int, clamp_nonneg: bool) -> torch.Tensor:
    """
    Head forward in raw-high space:
      x_high_raw -> (optional clamp) -> normalize -> fc1 -> relu -> fc2
    """
    if clamp_nonneg:
        x_high_raw = torch.clamp(x_high_raw, min=0.0)
    x = F.normalize(x_high_raw, dim=1)
    # CNNCifar: fc1/fc2 exist; for other models, head naming may differ.
    if not (hasattr(model, "fc1") and hasattr(model, "fc2")):
        raise RuntimeError("Model does not expose fc1/fc2; Stage-4 script currently targets CNNCifar-like heads.")
    x = F.relu(model.fc1(x))
    logits = model.fc2(x)
    return logits[:, 0:int(num_classes)]


def main() -> int:
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("--stage1_ckpt_path", type=str, required=True)
    p.add_argument("--split_path", type=str, default=None, help="split.pkl path (if None, infer from ckpt meta/args)")
    p.add_argument("--syn_path", type=str, required=True, help="class_syn_high_raw.pt path (Stage-3 output)")
    p.add_argument("--out_dir", type=str, default=None, help="Base output dir. If None, infer: <logdir>/classifier_fine_tunning/stage4")
    p.add_argument("--run_name", type=str, default=None, help="Optional run name suffix for output directory.")
    p.add_argument("--auto_run_dir", type=int, default=1, help="If 1, create timestamp subdir under out_dir (default: 1).")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=1234)

    # Training hparams (head only)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--log_interval", type=int, default=200)
    p.add_argument("--eval_interval", type=int, default=500)
    p.add_argument("--clamp_nonneg", type=int, default=1, help="Clamp synthetic raw-high to >=0 before normalize (default: 1)")
    p.add_argument(
        "--syn_class_mode",
        type=str,
        default="all",
        choices=["all", "local_seen"],
        help="Which classes' synthetic features to use for head finetune. "
        "'all': use all classes (default, original behavior). "
        "'local_seen': per-client, only use classes appearing in the client's local TRAIN split (user_groups[cid]).",
    )
    args = p.parse_args()

    device = resolve_device(args.gpu)
    seed_all(int(args.seed))

    # Load Stage-1 ckpt
    ckpt = load_checkpoint(args.stage1_ckpt_path, map_location="cpu")
    ckpt_meta = (ckpt.get("meta", {}) or {}) if isinstance(ckpt, dict) else {}
    ckpt_state = (ckpt.get("state", {}) or {}) if isinstance(ckpt, dict) else {}
    ckpt_args = (ckpt.get("args", {}) or {}) if isinstance(ckpt, dict) else {}

    dataset = str(ckpt_meta.get("dataset", ckpt_args.get("dataset", None)))
    num_users = int(ckpt_meta.get("num_users", ckpt_args.get("num_users", 20)))
    num_classes = int(ckpt_meta.get("num_classes", ckpt_args.get("num_classes", 10)))

    # Infer logdir / out_dir (stage-isolated + timestamped)
    logdir = infer_logdir_from_stage1_ckpt(args.stage1_ckpt_path, ckpt_meta=ckpt_meta, ckpt_args=ckpt_args)
    base_out_dir = args.out_dir or os.path.join(logdir, "classifier_fine_tunning", "stage4")
    out_dir = make_isolated_run_dir(
        base_out_dir=base_out_dir,
        run_name=args.run_name,
        auto_timestamp_subdir=int(getattr(args, "auto_run_dir", 1)) == 1,
    )
    os.makedirs(out_dir, exist_ok=True)

    # Resolve split_path
    split_path = args.split_path or ckpt_meta.get("split_path", None) or ckpt_args.get("split_path", None)
    if not split_path or not os.path.exists(split_path):
        raise FileNotFoundError(f"split_path not found: {split_path}")

    # Load split + datasets
    split = load_split(split_path)
    n_list = split["n_list"]
    k_list = split["k_list"]
    user_groups = split["user_groups"]
    user_groups_lt = split["user_groups_lt"]

    base = dict(ckpt_args)
    base["dataset"] = dataset
    base["num_classes"] = num_classes
    base["num_users"] = num_users
    base["gpu"] = int(args.gpu)
    base["device"] = device
    args_ds = SimpleNamespace(**base)
    train_dataset, test_dataset, _, _, _, _ = get_dataset(args_ds, n_list, k_list)

    # Load synthetic dataset (raw-high)
    syn_dataset = torch.load(args.syn_path, map_location="cpu")
    if not isinstance(syn_dataset, list) or len(syn_dataset) == 0:
        raise ValueError("syn_dataset is empty or invalid.")
    syn_z_all, syn_y_all = _extract_syn_xy(syn_dataset)
    syn_z_all = syn_z_all.to(device)
    syn_y_all = syn_y_all.to(device)

    # Stage-1 client weights
    local_sd = ckpt_state.get("local_models_full_state_dicts", None)
    if not isinstance(local_sd, dict):
        raise ValueError("Stage-1 checkpoint missing state['local_models_full_state_dicts']")

    # Meta for reproducibility
    meta_out = {
        "stage": 4,
        "feature_space": "high_raw",
        "dataset": dataset,
        "num_users": num_users,
        "num_classes": num_classes,
        "logdir": os.path.abspath(logdir),
        "out_dir_base": os.path.abspath(base_out_dir),
        "out_dir": os.path.abspath(out_dir),
        "stage1_ckpt_path": os.path.abspath(args.stage1_ckpt_path),
        "syn_path": os.path.abspath(args.syn_path),
        "split_path": os.path.abspath(split_path),
        "cmdline": " ".join([str(x) for x in sys.argv]),
        "env": collect_env_info(),
        "stage1_meta": ckpt_meta,
        "stage1_args": ckpt_args,
        "hparams": {
            "steps": int(args.steps),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "clamp_nonneg": int(args.clamp_nonneg),
            "seed": int(args.seed),
            "syn_class_mode": str(args.syn_class_mode),
        },
    }
    meta_out["args"] = vars(args)
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta_out, f, ensure_ascii=False, indent=2)

    results: list[dict[str, Any]] = []
    t_all = time.time()

    # Default synthetic loader (same for all clients); used when syn_class_mode='all'
    syn_dl_all = DataLoader(
        TensorDataset(syn_z_all, syn_y_all),
        batch_size=int(args.batch_size),
        shuffle=True,
        drop_last=True,
    )

    for cid in range(num_users):
        # Build model and load weights
        m = _build_client_model(SimpleNamespace(**base), cid).to(device)
        key = cid if cid in local_sd else str(cid)
        if key not in local_sd:
            raise KeyError(f"local_models_full_state_dicts missing client id={cid}")
        m.load_state_dict(local_sd[key], strict=True)

        # Freeze all, unfreeze head only
        for p0 in m.parameters():
            p0.requires_grad_(False)
        if not (hasattr(m, "fc1") and hasattr(m, "fc2")):
            raise RuntimeError(f"Client {cid}: model missing fc1/fc2; this Stage-4 script targets CNNCifar-like heads.")
        for p1 in m.fc1.parameters():
            p1.requires_grad_(True)
        for p2 in m.fc2.parameters():
            p2.requires_grad_(True)

        opt = torch.optim.Adam(
            [p for p in list(m.fc1.parameters()) + list(m.fc2.parameters()) if p.requires_grad],
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
        )

        # Client test loader (images)
        test_idxs = normalize_split_indices(user_groups_lt[int(cid)])
        dl_test = DataLoader(Subset(test_dataset, test_idxs), batch_size=int(args.batch_size), shuffle=False, num_workers=0)

        # Optionally filter synthetic classes to only those seen by this client locally (train split)
        seen_classes: list[int] | None = None
        syn_dl = syn_dl_all
        if str(args.syn_class_mode) == "local_seen":
            train_idxs = normalize_split_indices(user_groups[int(cid)]) if user_groups is not None else []
            seen_classes = _infer_seen_classes_from_indices(
                train_dataset,
                train_idxs,
                device=device,
                num_classes=num_classes,
                batch_size=max(256, int(args.batch_size)),
            )
            if len(seen_classes) == 0:
                print(f"[stage4_head_raw][cid={cid:02d}] syn_class_mode=local_seen but no seen classes found; fallback to all classes.")
            else:
                syn_dataset_local = [d for d in syn_dataset if int(d.get("class_index")) in set(seen_classes)]
                if len(syn_dataset_local) == 0:
                    print(
                        f"[stage4_head_raw][cid={cid:02d}] syn_class_mode=local_seen produced empty syn subset; fallback to all classes. "
                        f"seen_classes={seen_classes}"
                    )
                else:
                    z_local, y_local = _extract_syn_xy(syn_dataset_local)
                    syn_dl = DataLoader(
                        TensorDataset(z_local.to(device), y_local.to(device)),
                        batch_size=int(args.batch_size),
                        shuffle=True,
                        drop_last=True,
                    )

        # Baseline eval (before finetune) for comparison only.
        # NOTE: We do NOT allow baseline to "seed" best; best is computed from finetune-eval only.
        acc0, loss0 = eval_model_on_images(m, dl_test, device=device, num_classes=num_classes)

        # Train head using synthetic features
        m.train()
        it = iter(syn_dl)
        # Initialize to sentinel so first finetune eval always becomes "best"
        best_acc = -1.0
        best_loss = float("inf")
        best_state = None
        best_step = 0
        last_loss = None
        t0 = time.time()

        for step in range(int(args.steps)):
            try:
                xh, y = next(it)
            except StopIteration:
                it = iter(syn_dl)
                xh, y = next(it)

            xh = xh.to(device)
            y = y.to(device).long()

            logits = _forward_head_only(m, xh, num_classes=num_classes, clamp_nonneg=int(args.clamp_nonneg) == 1)
            loss = F.cross_entropy(logits, y, reduction="mean")

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            last_loss = float(loss.item())

            if (step + 1) % int(args.log_interval) == 0:
                dt = time.time() - t0
                print(f"[stage4_head_raw][cid={cid:02d}] step {step+1}/{int(args.steps)} loss={last_loss:.6f} sec={dt:.1f}")

            if (step + 1) % int(args.eval_interval) == 0 or (step + 1) == int(args.steps):
                acc1, loss1 = eval_model_on_images(m, dl_test, device=device, num_classes=num_classes)
                if float(acc1) >= float(best_acc):
                    best_acc = float(acc1)
                    best_loss = float(loss1)
                    best_step = int(step + 1)
                    # Save only head weights for compactness
                    best_state = {
                        "fc1": {k: v.detach().cpu().clone() for k, v in m.fc1.state_dict().items()},
                        "fc2": {k: v.detach().cpu().clone() for k, v in m.fc2.state_dict().items()},
                    }
                m.train()

        # Final eval
        accF, lossF = eval_model_on_images(m, dl_test, device=device, num_classes=num_classes)

        # Persist per-client head checkpoint
        out_path = os.path.join(out_dir, f"client_{cid:02d}_head.pt")
        torch.save(
            {
                "meta": {
                    "stage": 4,
                    "feature_space": "high_raw",
                    "client_id": cid,
                    "dataset": dataset,
                    "num_classes": num_classes,
                    "stage1_ckpt_path": os.path.abspath(args.stage1_ckpt_path),
                    "syn_path": os.path.abspath(args.syn_path),
                    "split_path": os.path.abspath(split_path),
                    "steps": int(args.steps),
                    "batch_size": int(args.batch_size),
                    "lr": float(args.lr),
                    "weight_decay": float(args.weight_decay),
                    "clamp_nonneg": int(args.clamp_nonneg),
                    "seed": int(args.seed),
                    "acc_before": float(acc0),
                    "acc_best": float(best_acc),
                    "best_step": int(best_step),
                    "acc_final": float(accF),
                },
                "state": {
                    "head_best": best_state,
                    "head_final": {
                        "fc1": m.fc1.state_dict(),
                        "fc2": m.fc2.state_dict(),
                    },
                },
            },
            out_path,
        )

        results.append(
            {
                "client_id": cid,
                "acc_before": float(acc0),
                "acc_best": float(best_acc),
                "best_step": int(best_step),
                "acc_final": float(accF),
                "loss_before": float(loss0),
                "loss_best": float(best_loss),
                "loss_final": float(lossF),
                "last_train_loss": float(last_loss) if last_loss is not None else None,
                "sec_client": float(time.time() - t0),
                "syn_class_mode": str(args.syn_class_mode),
                "seen_classes": seen_classes,
            }
        )

    # Summary
    acc_before = [r["acc_before"] for r in results]
    acc_best = [r["acc_best"] for r in results]
    acc_final = [r["acc_final"] for r in results]
    out_summary = {
        "meta": meta_out,
        "results": results,
        "sec_total": float(time.time() - t_all),
        "mean_acc_before": float(np.mean(acc_before)) if acc_before else 0.0,
        "mean_acc_best": float(np.mean(acc_best)) if acc_best else 0.0,
        "mean_acc_final": float(np.mean(acc_final)) if acc_final else 0.0,
    }
    with open(os.path.join(out_dir, "stage4_results.json"), "w", encoding="utf-8") as f:
        json.dump(out_summary, f, ensure_ascii=False, indent=2)

    print(f"[stage4_head_raw] done. out_dir={out_dir}")
    print(f"[stage4_head_raw] mean acc before={out_summary['mean_acc_before']:.4f} best={out_summary['mean_acc_best']:.4f} final={out_summary['mean_acc_final']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


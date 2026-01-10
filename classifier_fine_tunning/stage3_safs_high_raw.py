from __future__ import annotations

"""
Stage-3 (raw-high): SAFS feature synthesis in raw high-level feature space.

Input:
  - global_high_stats_raw.pt from classifier_fine_tunning/stage2_high_stats_raw.py
Output:
  - class_syn_high_raw.pt: list[dict{class_index, synthetic_raw_features, synthetic_features}]
"""

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List

import torch

from utils import (
    collect_env_info,
    ensure_repo_root_on_path,
    infer_logdir_from_stage1_ckpt,
    make_isolated_run_dir,
    resolve_device,
    seed_all,
)


ensure_repo_root_on_path()

from lib.safs import MeanCovAligner, feature_synthesis, make_syn_nums
from lib.sfd_utils import RFF


@dataclass
class SafsArgs:
    steps: int
    lr: float
    max_syn_num: int
    min_syn_num: int
    target_cov_eps: float
    input_cov_eps: float


def main() -> int:
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("--stats_path", type=str, required=True, help="Path to global_high_stats_raw.pt")
    p.add_argument("--out_dir", type=str, default=None, help="Base output dir. If None, infer: <logdir>/classifier_fine_tunning/stage3")
    p.add_argument("--run_name", type=str, default=None, help="Optional run name suffix for output directory.")
    p.add_argument("--auto_run_dir", type=int, default=1, help="If 1, create timestamp subdir under out_dir (default: 1).")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=1234)

    # SAFS hyperparams
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--max_syn_num", type=int, default=2000)
    p.add_argument("--min_syn_num", type=int, default=600)
    p.add_argument("--target_cov_eps", type=float, default=1e-5)
    p.add_argument("--input_cov_eps", type=float, default=1e-5)

    # Optional: override syn nums directly (comma-separated)
    p.add_argument("--syn_nums", type=str, default=None, help="Optional per-class syn nums, e.g. '2000,1800,...'")
    args = p.parse_args()

    device = resolve_device(args.gpu)
    seed_all(int(args.seed))

    payload = torch.load(args.stats_path, map_location="cpu")
    meta = payload.get("meta", {}) or {}
    state = payload.get("state", {}) or {}
    global_stats = state.get("global_stats", None)
    rf_state = state.get("rf_model_state", None)
    if global_stats is None or "high" not in global_stats:
        raise KeyError("stats payload missing state.global_stats['high']")
    if rf_state is None:
        raise KeyError("stats payload missing state.rf_model_state")

    num_classes = int(meta.get("num_classes", None) or global_stats.get("sample_per_class", torch.tensor([])).numel())
    feature_dim = int(meta.get("high_feature_dim", None) or len(global_stats["high"]["class_means"][0]))
    rf_args = (meta.get("rf_args", {}) or {})
    rf_dim = int(rf_args.get("rf_dim", None) or rf_args.get("rf_dim_high", args.__dict__.get("rf_dim", 3000)))
    rbf_gamma = float(rf_args.get("rbf_gamma", rf_args.get("rbf_gamma_high", 0.01)))
    rf_type = str(rf_args.get("rf_type", "orf"))

    # Infer logdir from stage2 meta -> stage1_ckpt_path/meta, then resolve out_dir
    stage1_ckpt_path = meta.get("stage1_ckpt_path", None)
    stage1_meta = meta.get("stage1_meta", None)
    stage1_args = meta.get("stage1_args", None)
    if isinstance(stage1_ckpt_path, str) and stage1_ckpt_path.strip():
        logdir = infer_logdir_from_stage1_ckpt(stage1_ckpt_path, ckpt_meta=stage1_meta, ckpt_args=stage1_args)
    else:
        # fallback: stats_path directory
        logdir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(args.stats_path)), "..", ".."))
    base_out_dir = args.out_dir or os.path.join(logdir, "classifier_fine_tunning", "stage3")
    out_dir = make_isolated_run_dir(
        base_out_dir=base_out_dir,
        run_name=args.run_name,
        auto_timestamp_subdir=int(getattr(args, "auto_run_dir", 1)) == 1,
    )

    os.makedirs(out_dir, exist_ok=True)

    rf_model = RFF(d=feature_dim, D=rf_dim, gamma=rbf_gamma, device=device, rf_type=rf_type)
    rf_model.load_state_dict(rf_state, strict=True)
    rf_model = rf_model.to(device).eval()

    class_means: List[torch.Tensor] = global_stats["high"]["class_means"]
    class_covs: List[torch.Tensor] = global_stats["high"]["class_covs"]
    class_rf_means: List[torch.Tensor] = global_stats["high"]["class_rf_means"]
    sample_per_class: torch.Tensor = global_stats["sample_per_class"]

    if args.syn_nums is not None:
        syn_nums = [int(x.strip()) for x in str(args.syn_nums).split(",") if x.strip()]
        if len(syn_nums) != num_classes:
            raise ValueError(f"--syn_nums length mismatch: got {len(syn_nums)} vs num_classes={num_classes}")
    else:
        syn_nums = make_syn_nums(
            class_sizes=sample_per_class.tolist(),
            max_num=int(args.max_syn_num),
            min_num=int(args.min_syn_num),
        )

    # SAFS requires syn_num > feature_dim for stable covariance estimates
    if min(syn_nums) <= feature_dim:
        raise ValueError(
            f"SAFS requires min(syn_nums) > feature_dim; got min={min(syn_nums)} feature_dim={feature_dim}. "
            f"Adjust --max_syn_num/--min_syn_num/--syn_nums."
        )

    aligners: list[MeanCovAligner] = []
    for c in range(num_classes):
        aligners.append(
            MeanCovAligner(
                target_mean=class_means[c],
                target_cov=class_covs[c],
                target_cov_eps=float(args.target_cov_eps),
            )
        )

    t0 = time.time()
    syn_dataset = feature_synthesis(
        feature_dim=feature_dim,
        class_num=num_classes,
        device=device,
        aligners=aligners,
        rf_model=rf_model,
        class_rf_means=class_rf_means,
        steps=int(args.steps),
        lr=float(args.lr),
        syn_num_per_class=syn_nums,
        input_cov_eps=float(args.input_cov_eps),
    )

    out_path = os.path.join(out_dir, "class_syn_high_raw.pt")
    torch.save(syn_dataset, out_path)

    meta_out = {
        "stage": 3,
        "feature_space": "high_raw",
        "stats_path": os.path.abspath(args.stats_path),
        "logdir": os.path.abspath(logdir),
        "out_dir_base": os.path.abspath(base_out_dir),
        "out_dir": os.path.abspath(out_dir),
        "cmdline": " ".join([str(x) for x in sys.argv]),
        "env": collect_env_info(),
        "num_classes": num_classes,
        "feature_dim": feature_dim,
        "rf": {"rf_dim": rf_dim, "rbf_gamma": rbf_gamma, "rf_type": rf_type},
        "safs": asdict(
            SafsArgs(
                steps=int(args.steps),
                lr=float(args.lr),
                max_syn_num=int(args.max_syn_num),
                min_syn_num=int(args.min_syn_num),
                target_cov_eps=float(args.target_cov_eps),
                input_cov_eps=float(args.input_cov_eps),
            )
        ),
        "syn_nums": syn_nums,
        "sec_total": float(time.time() - t0),
    }
    # record full CLI args for reproduction
    meta_out["args"] = vars(args)
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta_out, f, ensure_ascii=False, indent=2)

    print(f"[stage3_safs_high_raw] saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


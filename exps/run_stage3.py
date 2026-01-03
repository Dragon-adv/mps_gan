#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage-3 runner for FedMPS: train stats-conditioned low-level feature generator.

Usage examples:
  python exps/run_stage3.py --log_dir ../newresults/ours/my_run
  python exps/run_stage3.py --stage2_stats_path ../newresults/ours/my_run/stage2_stats/global_stats.pt
  python exps/run_stage3.py --gpu 0 --steps 5000 --batch_size 512 --lr 1e-3

Note:
  This script is a thin wrapper over exps/federated_main.py with --stage 3.
"""

import argparse
import os
import shlex
import subprocess
import sys


def main() -> int:
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("--log_dir", type=str, default=None, help="传给 federated_main.py 的 --log_dir")
    p.add_argument("--stage2_stats_path", type=str, default=None, help="Stage-2 的 global_stats.pt 路径")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--extra", type=str, default="", help="额外透传参数字符串")
    args = p.parse_args()

    log_dir = args.log_dir
    if log_dir is None:
        # default: infer from stats_path if provided
        if args.stage2_stats_path:
            # expected: <log_dir>/stage2_stats/global_stats.pt
            stage2_dir = os.path.dirname(os.path.abspath(args.stage2_stats_path))
            if os.path.basename(stage2_dir) == "stage2_stats":
                log_dir = os.path.dirname(stage2_dir)
            else:
                log_dir = stage2_dir
        else:
            raise ValueError("Please provide --log_dir or --stage2_stats_path")

    cmd = [
        sys.executable,
        "exps/federated_main.py",
        "--stage", "3",
        "--log_dir", log_dir,
        "--gpu", str(args.gpu),
        "--stage2_stats_path", str(args.stage2_stats_path) if args.stage2_stats_path else os.path.join(log_dir, "stage2_stats", "global_stats.pt"),
        "--gen_steps", str(args.steps),
        "--gen_batch_size", str(args.batch_size),
        "--gen_lr", str(args.lr),
        "--gen_seed", str(args.seed),
    ]

    if args.extra:
        cmd.extend(shlex.split(args.extra))

    print("Running command:")
    print(" ".join(shlex.quote(x) for x in cmd))
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())



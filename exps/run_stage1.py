#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Stage-1 runner for FedMPS (ours) - CIFAR-10 default
#
# 作用：
# - 把一阶段常用参数集中在一个py入口里，便于维护/复现实验
# - 支持命令行覆盖（传入的参数会覆盖默认参数）
# - 通过 latest.pt 支持断点续训（--resume_ckpt_path）
#
# 用法示例：
#   python exps/run_stage1.py --gpu 0 --rounds 300
#   python exps/run_stage1.py --log_dir ../newresults/ours/my_run --reuse_split 1
#   python exps/run_stage1.py --resume_ckpt_path ../newresults/ours/my_run/stage1_ckpts/latest.pt

import argparse
import os
import shlex
import subprocess
import sys
from datetime import datetime


def build_default_args() -> dict:
    # 你的默认命令（可在此统一维护）
    return {
        # core
        "alg": "ours",
        "dataset": "cifar10",
        "num_classes": 10,
        "num_users": 20,
        "ways": 3,
        "shots": 100,
        "train_shots_max": 110,
        "test_shots": 15,
        "stdev": 1,
        "rounds": 300,
        "gpu": 0,
        # FedMPS/ABBL weights
        "alph": 0.1,
        "beta": 0.02,
        "gama": 5,
        # Stage-1 persistence / ckpt
        "stage": 1,
        "reuse_split": 1,
        "save_latest_ckpt": 1,
        "latest_ckpt_interval": 1,
        # history snapshots (latest_rXXXX.pt)
        "save_latest_history": 1,
        "latest_history_interval": 25,
        "save_best_ckpt": 1,
        # reduce best checkpoint count: overwrite best-wo.pt / best-wp.pt by default
        "best_ckpt_overwrite": 1,
        "save_stage1_components": 1,
    }


def parse_overrides() -> argparse.Namespace:
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("--log_dir", type=str, default=None, help="传给 federated_main.py 的 --log_dir")
    p.add_argument("--split_path", type=str, default=None, help="传给 federated_main.py 的 --split_path")
    p.add_argument("--resume_ckpt_path", type=str, default=None, help="latest.pt 路径")

    # 常用覆盖项（需要更多可以继续加）
    p.add_argument("--gpu", type=int, default=None)
    p.add_argument("--rounds", type=int, default=None)
    p.add_argument("--num_users", type=int, default=None)
    p.add_argument("--ways", type=int, default=None)
    p.add_argument("--shots", type=int, default=None)
    p.add_argument("--train_shots_max", type=int, default=None)
    p.add_argument("--test_shots", type=int, default=None)
    p.add_argument("--stdev", type=int, default=None)
    p.add_argument("--alph", type=float, default=None)
    p.add_argument("--beta", type=float, default=None)
    p.add_argument("--gama", type=float, default=None)
    p.add_argument("--reuse_split", type=int, default=None)
    p.add_argument("--latest_ckpt_interval", type=int, default=None)
    p.add_argument("--save_latest_history", type=int, default=None)
    p.add_argument("--latest_history_interval", type=int, default=None)
    p.add_argument("--save_best_ckpt", type=int, default=None)
    p.add_argument("--best_ckpt_overwrite", type=int, default=None)

    # 透传其它未知参数（例如你新增的各种实验开关），用 --extra "--key value --k2 v2"
    p.add_argument("--extra", type=str, default="", help="额外透传参数字符串")
    return p.parse_args()


def main() -> int:
    base = build_default_args()
    ov = parse_overrides()

    # 覆盖默认值
    for k, v in vars(ov).items():
        if k in ("extra",):
            continue
        if v is not None:
            base[k] = v

    # 默认 log_dir：固定到一个可复用目录（便于 split.pkl & latest.pt 一致）
    # 若你不想固定目录，可以显式传 --log_dir
    if base.get("log_dir") is None:
        ts = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
        base["log_dir"] = os.path.join("..", "newresults", base["alg"], f"{ts}_{base['dataset']}_n{base['ways']}")

    # split_path 默认挂在 log_dir 下
    if base.get("split_path") is None:
        base["split_path"] = os.path.join(base["log_dir"], "split.pkl")

    cmd = [sys.executable, "exps/federated_main.py"]
    for k, v in base.items():
        # bool/int/float/str 都统一按 --k v 形式传递
        cmd.append(f"--{k}")
        cmd.append(str(v))

    if ov.extra:
        cmd.extend(shlex.split(ov.extra))

    print("Running command:")
    print(" ".join(shlex.quote(x) for x in cmd))
    print(f"log_dir = {base['log_dir']}")
    print(f"split_path = {base['split_path']}")
    if base.get("resume_ckpt_path"):
        print(f"resume_ckpt_path = {base['resume_ckpt_path']}")

    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Stage-1 runner script (cross-platform).
集中管理一阶段常用参数，避免每次手写超长命令；也便于后续多阶段复用 split/ckpt。
"""

import argparse
import os
import subprocess
import sys
import datetime


def build_default_logdir(args):
    ts = datetime.datetime.now().strftime("%Y-%m-%d/%H.%M.%S")
    return os.path.join("..", "newresults", args.alg, f"{ts}_{args.dataset}_n{args.ways}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--rounds", type=int, default=300)
    p.add_argument("--num_users", type=int, default=20)
    p.add_argument("--ways", type=int, default=3)
    p.add_argument("--shots", type=int, default=100)
    p.add_argument("--train_shots_max", type=int, default=110)
    p.add_argument("--test_shots", type=int, default=15)
    p.add_argument("--stdev", type=int, default=1)
    p.add_argument("--dataset", type=str, default="cifar10")
    p.add_argument("--num_classes", type=int, default=10)
    p.add_argument("--alg", type=str, default="ours")
    p.add_argument("--alph", type=float, default=0.1)
    p.add_argument("--beta", type=float, default=0.02)
    p.add_argument("--gama", type=float, default=5.0)

    # persistence / resume
    p.add_argument("--log_dir", type=str, default=None)
    p.add_argument("--split_path", type=str, default=None)
    p.add_argument("--reuse_split", type=int, default=1)
    p.add_argument("--resume_ckpt_path", type=str, default=None)
    p.add_argument("--latest_ckpt_interval", type=int, default=1)

    args, unknown = p.parse_known_args()

    log_dir = args.log_dir or build_default_logdir(args)
    split_path = args.split_path or os.path.join(log_dir, "split.pkl")

    cmd = [
        sys.executable, "exps/federated_main.py",
        "--alg", args.alg,
        "--dataset", args.dataset,
        "--num_classes", str(args.num_classes),
        "--num_users", str(args.num_users),
        "--ways", str(args.ways),
        "--shots", str(args.shots),
        "--train_shots_max", str(args.train_shots_max),
        "--test_shots", str(args.test_shots),
        "--stdev", str(args.stdev),
        "--alph", str(args.alph),
        "--beta", str(args.beta),
        "--gama", str(args.gama),
        "--rounds", str(args.rounds),
        "--gpu", str(args.gpu),
        "--log_dir", log_dir,
        "--split_path", split_path,
        "--reuse_split", str(args.reuse_split),
        "--latest_ckpt_interval", str(args.latest_ckpt_interval),
    ]

    if args.resume_ckpt_path:
        cmd += ["--resume_ckpt_path", args.resume_ckpt_path]

    # passthrough extra args
    cmd += unknown

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()



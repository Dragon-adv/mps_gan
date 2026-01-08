r"""
解析 stage4_cvae_pool.log 并绘制训练/验证曲线。

使用示例（Windows PowerShell）:
py analysis/plot_stage4_cvae_pool.py --log_path "D:\cursor_workspace\FedMPS_GAN\newresults\ours\2026-01-01_20.43.57_cifar10_n3\stage4\finetune_cvae_pool\cvae_pool_pr1_pmin200_pmax5000_syn0p2_a1_r1_client_classes_steps2000_bs128_lr0p0001_wd0_seed1500692655\stage4_cvae_pool.log" --output_dir "analysis/plots_stage4"

依赖: matplotlib（如缺失请先安装 pip install matplotlib）
"""

from __future__ import annotations

import argparse
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import DefaultDict, Dict, List, Tuple

import matplotlib.pyplot as plt


@dataclass
class ClientLog:
    train_steps: List[int] = field(default_factory=list)
    train_loss: List[float] = field(default_factory=list)
    real_ce: List[float] = field(default_factory=list)
    syn_ce: List[float] = field(default_factory=list)
    eval_steps: List[int] = field(default_factory=list)
    eval_acc: List[float] = field(default_factory=list)
    eval_loss: List[float] = field(default_factory=list)
    acc_before: float | None = None
    acc_after: float | None = None
    best_step: int | None = None


def parse_log(log_path: str) -> Dict[int, ClientLog]:
    train_re = re.compile(
        r"\[cid=(\d+)\]\s+step\s+(\d+)/\d+\s+loss=([-\d\.eE]+)\s+real_ce=([-\d\.eE]+)\s+syn_ce=([-\d\.eE]+)"
    )
    eval_re = re.compile(
        r"\[cid=(\d+)\]\s+eval@(\d+):\s+acc=([-\d\.eE]+)\s+loss=([-\d\.eE]+)"
    )
    done_re = re.compile(
        r"\[cid=(\d+)\]\s+done:\s+acc\s+([-\d\.eE]+)\s*->\s*([-\d\.eE]+).*best_step=(\d+)"
    )

    clients: Dict[int, ClientLog] = defaultdict(ClientLog)

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            if m := train_re.search(line):
                cid = int(m.group(1))
                step = int(m.group(2))
                loss = float(m.group(3))
                real_ce = float(m.group(4))
                syn_ce = float(m.group(5))
                cl = clients[cid]
                cl.train_steps.append(step)
                cl.train_loss.append(loss)
                cl.real_ce.append(real_ce)
                cl.syn_ce.append(syn_ce)
                continue

            if m := eval_re.search(line):
                cid = int(m.group(1))
                step = int(m.group(2))
                acc = float(m.group(3))
                loss = float(m.group(4))
                cl = clients[cid]
                cl.eval_steps.append(step)
                cl.eval_acc.append(acc)
                cl.eval_loss.append(loss)
                continue

            if m := done_re.search(line):
                cid = int(m.group(1))
                cl = clients[cid]
                cl.acc_before = float(m.group(2))
                cl.acc_after = float(m.group(3))
                cl.best_step = int(m.group(4))
                continue

    return clients


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def plot_client_curves(
    cid: int, log: ClientLog, output_dir: str, show_train: bool
) -> None:
    _ensure_dir(output_dir)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Eval curves
    axes[0].plot(log.eval_steps, log.eval_acc, marker="o", label="eval_acc")
    axes[0].set_title(f"Client {cid:02d} Eval Acc")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Accuracy")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(log.eval_steps, log.eval_loss, marker="o", color="tab:red", label="eval_loss")
    axes[1].set_title(f"Client {cid:02d} Eval Loss")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Loss")
    axes[1].grid(True, alpha=0.3)

    subtitle = []
    if log.acc_before is not None and log.acc_after is not None:
        subtitle.append(f"before {log.acc_before:.4f} -> after {log.acc_after:.4f}")
    if log.best_step is not None:
        subtitle.append(f"best_step={log.best_step}")
    if subtitle:
        fig.suptitle("; ".join(subtitle), fontsize=10)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"client_{cid:02d}_eval.png"), dpi=150)
    plt.close(fig)

    if show_train and log.train_steps:
        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
        ax.plot(log.train_steps, log.train_loss, label="loss")
        ax.plot(log.train_steps, log.real_ce, label="real_ce")
        ax.plot(log.train_steps, log.syn_ce, label="syn_ce")
        ax.set_title(f"Client {cid:02d} Train Loss")
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"client_{cid:02d}_train.png"), dpi=150)
        plt.close(fig)


def plot_aggregated_eval(clients: Dict[int, ClientLog], output_dir: str) -> None:
    # 聚合所有客户端的 eval acc/loss（按 step 求均值/标准差）
    acc_by_step: DefaultDict[int, List[float]] = defaultdict(list)
    loss_by_step: DefaultDict[int, List[float]] = defaultdict(list)
    for cl in clients.values():
        for s, acc in zip(cl.eval_steps, cl.eval_acc):
            acc_by_step[s].append(acc)
        for s, ls in zip(cl.eval_steps, cl.eval_loss):
            loss_by_step[s].append(ls)

    def mean_std(items: List[float]) -> Tuple[float, float]:
        if not items:
            return 0.0, 0.0
        mean = sum(items) / len(items)
        var = sum((x - mean) ** 2 for x in items) / len(items)
        return mean, var**0.5

    steps = sorted(acc_by_step.keys())
    mean_acc = []
    std_acc = []
    mean_loss = []
    std_loss = []
    for s in steps:
        m, sd = mean_std(acc_by_step[s])
        mean_acc.append(m)
        std_acc.append(sd)
        m2, sd2 = mean_std(loss_by_step[s])
        mean_loss.append(m2)
        std_loss.append(sd2)

    _ensure_dir(output_dir)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(steps, mean_acc, label="mean acc", color="tab:blue")
    axes[0].fill_between(
        steps,
        [m - s for m, s in zip(mean_acc, std_acc)],
        [m + s for m, s in zip(mean_acc, std_acc)],
        color="tab:blue",
        alpha=0.2,
        label="±1 std",
    )
    axes[0].set_title("Eval Accuracy (mean ± std)")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Accuracy")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(steps, mean_loss, label="mean loss", color="tab:red")
    axes[1].fill_between(
        steps,
        [m - s for m, s in zip(mean_loss, std_loss)],
        [m + s for m, s in zip(mean_loss, std_loss)],
        color="tab:red",
        alpha=0.2,
        label="±1 std",
    )
    axes[1].set_title("Eval Loss (mean ± std)")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Loss")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "aggregated_eval.png"), dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_path", required=True, help="stage4_cvae_pool.log 路径")
    parser.add_argument(
        "--output_dir",
        default="analysis/plots_stage4_cvae_pool",
        help="输出图片目录",
    )
    parser.add_argument(
        "--clients",
        type=str,
        default="all",
        help='指定客户端，例如 "0,1,2"，默认 all',
    )
    parser.add_argument(
        "--no_train",
        action="store_true",
        help="只画 eval 曲线，不画训练 loss",
    )
    args = parser.parse_args()

    clients_logs = parse_log(args.log_path)

    # 选择客户端
    if args.clients != "all":
        selected = {int(x.strip()) for x in args.clients.split(",") if x.strip()}
    else:
        selected = set(clients_logs.keys())

    for cid, log in clients_logs.items():
        if cid not in selected:
            continue
        plot_client_curves(
            cid, log, os.path.join(args.output_dir, "clients"), show_train=not args.no_train
        )

    plot_aggregated_eval(clients_logs, os.path.join(args.output_dir, "aggregate"))
    print(f"完成，图片保存在: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()


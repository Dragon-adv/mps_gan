#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从 client_owned_metrics.csv 中选择 topK 个最优模型作为 GAN 网络中的鉴别器。

基于 test 和 all 上的 acc 和 loss_ce 来评估本地模型对特征的理解程度。

输出：
- discriminator_map.json: 每个类选择的 topK 个客户端 ID
- discriminator_scores.csv: 每个客户端在每个类上的详细评分
- discriminator_summary.json: 选择过程的汇总信息
- discriminator_report.txt: 可读性报告
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def compute_discriminator_score(
    acc_test: float,
    loss_ce_test: float,
    acc_all: float,
    loss_ce_all: float,
    weight_test: float = 0.5,
    weight_all: float = 0.5,
    acc_weight: float = 0.7,
    loss_weight: float = 0.3,
    normalize_loss: bool = True,
    loss_max: float = 5.0,
) -> Tuple[float, Dict[str, float]]:
    """
    计算鉴别器评分
    
    Args:
        acc_test: test 集准确率
        loss_ce_test: test 集 CE 损失
        acc_all: 所有数据准确率
        loss_ce_all: 所有数据 CE 损失
        weight_test: test 集权重（默认 0.5）
        weight_all: all 集权重（默认 0.5）
        acc_weight: 准确率权重（默认 0.7）
        loss_weight: 损失权重（默认 0.3）
        normalize_loss: 是否归一化损失（默认 True）
        loss_max: 损失的最大值，用于归一化（默认 5.0）
    
    Returns:
        (final_score, details_dict): 综合评分和详细分解
    """
    # 处理 NaN
    has_nan = False
    if np.isnan(acc_test) or np.isnan(loss_ce_test):
        acc_test, loss_ce_test = 0.0, loss_max
        has_nan = True
    if np.isnan(acc_all) or np.isnan(loss_ce_all):
        acc_all, loss_ce_all = 0.0, loss_max
        has_nan = True
    
    # 归一化损失
    if normalize_loss:
        # 使用 sigmoid 归一化：1 / (1 + exp((loss - center) / scale))
        # center=2.0, scale=1.0 使得 loss=0->0.88, loss=2->0.5, loss=5->0.05
        loss_norm_test = 1.0 / (1.0 + np.exp((loss_ce_test - 2.0) / 1.0))
        loss_norm_all = 1.0 / (1.0 + np.exp((loss_ce_all - 2.0) / 1.0))
    else:
        # 线性归一化
        loss_norm_test = max(0.0, 1.0 - loss_ce_test / loss_max)
        loss_norm_all = max(0.0, 1.0 - loss_ce_all / loss_max)
    
    # 综合 test 和 all 的得分
    score_test = acc_weight * acc_test + loss_weight * loss_norm_test
    score_all = acc_weight * acc_all + loss_weight * loss_norm_all
    
    # 加权平均
    final_score = weight_test * score_test + weight_all * score_all
    
    details = {
        'acc_test': float(acc_test) if not np.isnan(acc_test) else float('nan'),
        'loss_ce_test': float(loss_ce_test) if not np.isnan(loss_ce_test) else float('nan'),
        'acc_all': float(acc_all) if not np.isnan(acc_all) else float('nan'),
        'loss_ce_all': float(loss_ce_all) if not np.isnan(loss_ce_all) else float('nan'),
        'loss_norm_test': float(loss_norm_test),
        'loss_norm_all': float(loss_norm_all),
        'score_test': float(score_test),
        'score_all': float(score_all),
        'final_score': float(final_score),
        'has_nan': has_nan,
    }
    
    return final_score, details


def build_discriminator_map(
    csv_path: str,
    k_discriminators: int = 3,
    n_samples_min: int = 10,
    weight_test: float = 0.5,
    weight_all: float = 0.5,
    acc_weight: float = 0.7,
    loss_weight: float = 0.3,
    normalize_loss: bool = True,
    loss_max: float = 5.0,
) -> Tuple[Dict[int, List[int]], pd.DataFrame, Dict]:
    """
    从 CSV 文件构建鉴别器映射
    
    Args:
        csv_path: client_owned_metrics.csv 路径
        k_discriminators: 每个类选择的鉴别器数量
        n_samples_min: 每类最少样本数门槛
        weight_test: test 集权重
        weight_all: all 集权重
        acc_weight: 准确率权重
        loss_weight: 损失权重
        normalize_loss: 是否归一化损失
        loss_max: 损失最大值
    
    Returns:
        (discriminator_map, scores_df, summary_dict):
        - discriminator_map: Dict[class_id, List[client_id]]
        - scores_df: 包含详细评分的 DataFrame
        - summary_dict: 选择过程的汇总信息
    """
    # 读取 CSV
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV 文件不存在: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # 验证必要的列
    required_cols = ['client_id', 'class_id', 'n_test', 'acc_test', 'loss_ce_test', 'n_all', 'acc_all', 'loss_ce_all']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV 文件缺少必要的列: {missing_cols}")
    
    # 计算每个 (client_id, class_id) 的得分
    scores_data = []
    for _, row in df.iterrows():
        score, details = compute_discriminator_score(
            acc_test=row['acc_test'],
            loss_ce_test=row['loss_ce_test'],
            acc_all=row['acc_all'],
            loss_ce_all=row['loss_ce_all'],
            weight_test=weight_test,
            weight_all=weight_all,
            acc_weight=acc_weight,
            loss_weight=loss_weight,
            normalize_loss=normalize_loss,
            loss_max=loss_max,
        )
        
        scores_data.append({
            'client_id': int(row['client_id']),
            'class_id': int(row['class_id']),
            'n_train': int(row['n_train']),
            'n_test': int(row['n_test']),
            'n_all': int(row['n_all']),
            'acc_test': details['acc_test'],
            'loss_ce_test': details['loss_ce_test'],
            'acc_all': details['acc_all'],
            'loss_ce_all': details['loss_ce_all'],
            'loss_norm_test': details['loss_norm_test'],
            'loss_norm_all': details['loss_norm_all'],
            'score_test': details['score_test'],
            'score_all': details['score_all'],
            'final_score': details['final_score'],
            'has_nan': details['has_nan'],
        })
    
    scores_df = pd.DataFrame(scores_data)
    
    # 按类选择 topK
    discriminator_map = {}
    num_classes = int(scores_df['class_id'].max() + 1)
    
    class_stats = []
    for class_id in range(num_classes):
        class_scores = scores_df[scores_df['class_id'] == class_id].copy()
        
        if len(class_scores) == 0:
            discriminator_map[class_id] = []
            class_stats.append({
                'class_id': class_id,
                'total_candidates': 0,
                'qualified_candidates': 0,
                'selected': [],
            })
            continue
        
        # 筛选样本数足够的客户端
        mask = (class_scores['n_test'] >= n_samples_min) | (class_scores['n_all'] >= n_samples_min)
        qualified = class_scores[mask].copy()
        
        if len(qualified) == 0:
            # fallback: 允许所有客户端
            qualified = class_scores.copy()
            mask = class_scores['n_test'] >= 0
        
        # 按得分降序排序
        qualified = qualified.sort_values('final_score', ascending=False)
        
        # 选择 topK
        topk_clients = qualified.head(k_discriminators)['client_id'].tolist()
        discriminator_map[class_id] = topk_clients
        
        class_stats.append({
            'class_id': class_id,
            'total_candidates': len(class_scores),
            'qualified_candidates': len(qualified),
            'selected': topk_clients,
            'selected_scores': qualified.head(k_discriminators)['final_score'].tolist(),
        })
    
    # 构建汇总信息
    summary = {
        'parameters': {
            'k_discriminators': k_discriminators,
            'n_samples_min': n_samples_min,
            'weight_test': weight_test,
            'weight_all': weight_all,
            'acc_weight': acc_weight,
            'loss_weight': loss_weight,
            'normalize_loss': normalize_loss,
            'loss_max': loss_max,
        },
        'statistics': {
            'total_classes': num_classes,
            'total_client_class_pairs': len(scores_df),
            'classes_with_discriminators': sum(1 for v in discriminator_map.values() if len(v) > 0),
            'classes_without_discriminators': sum(1 for v in discriminator_map.values() if len(v) == 0),
        },
        'per_class_stats': class_stats,
        'score_statistics': {
            'mean_final_score': float(scores_df['final_score'].mean()),
            'std_final_score': float(scores_df['final_score'].std()),
            'min_final_score': float(scores_df['final_score'].min()),
            'max_final_score': float(scores_df['final_score'].max()),
        },
    }
    
    return discriminator_map, scores_df, summary


def save_results(
    out_dir: str,
    discriminator_map: Dict[int, List[int]],
    scores_df: pd.DataFrame,
    summary: Dict,
    csv_path: str,
) -> None:
    """保存所有结果文件"""
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. 保存 discriminator_map.json
    map_path = os.path.join(out_dir, 'discriminator_map.json')
    with open(map_path, 'w', encoding='utf-8') as f:
        json.dump(discriminator_map, f, indent=2, ensure_ascii=False)
    print(f"[OK] Saved discriminator_map.json: {map_path}")
    
    # 2. 保存 discriminator_scores.csv
    scores_path = os.path.join(out_dir, 'discriminator_scores.csv')
    scores_df.to_csv(scores_path, index=False, float_format='%.6f')
    print(f"[OK] Saved discriminator_scores.csv: {scores_path}")
    
    # 3. 保存 discriminator_summary.json
    summary_path = os.path.join(out_dir, 'discriminator_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[OK] Saved discriminator_summary.json: {summary_path}")
    
    # 4. 生成可读性报告
    report_path = os.path.join(out_dir, 'discriminator_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("鉴别器选择报告\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"输入文件: {csv_path}\n")
        f.write(f"输出目录: {out_dir}\n\n")
        
        f.write("参数设置:\n")
        f.write("-" * 80 + "\n")
        params = summary['parameters']
        f.write(f"  每个类选择的鉴别器数量 (k_discriminators): {params['k_discriminators']}\n")
        f.write(f"  每类最少样本数门槛 (n_samples_min): {params['n_samples_min']}\n")
        f.write(f"  Test 集权重 (weight_test): {params['weight_test']}\n")
        f.write(f"  All 集权重 (weight_all): {params['weight_all']}\n")
        f.write(f"  准确率权重 (acc_weight): {params['acc_weight']}\n")
        f.write(f"  损失权重 (loss_weight): {params['loss_weight']}\n")
        f.write(f"  损失归一化 (normalize_loss): {params['normalize_loss']}\n")
        f.write(f"  损失最大值 (loss_max): {params['loss_max']}\n\n")
        
        f.write("统计信息:\n")
        f.write("-" * 80 + "\n")
        stats = summary['statistics']
        f.write(f"  总类别数: {stats['total_classes']}\n")
        f.write(f"  总客户端-类别对: {stats['total_client_class_pairs']}\n")
        f.write(f"  有鉴别器的类别数: {stats['classes_with_discriminators']}\n")
        f.write(f"  无鉴别器的类别数: {stats['classes_without_discriminators']}\n\n")
        
        score_stats = summary['score_statistics']
        f.write("评分统计:\n")
        f.write("-" * 80 + "\n")
        f.write(f"  平均得分: {score_stats['mean_final_score']:.6f}\n")
        f.write(f"  标准差: {score_stats['std_final_score']:.6f}\n")
        f.write(f"  最小得分: {score_stats['min_final_score']:.6f}\n")
        f.write(f"  最大得分: {score_stats['max_final_score']:.6f}\n\n")
        
        f.write("每个类的选择结果:\n")
        f.write("-" * 80 + "\n")
        for class_stat in summary['per_class_stats']:
            class_id = class_stat['class_id']
            selected = class_stat['selected']
            f.write(f"\n类别 {class_id}:\n")
            f.write(f"  候选客户端数: {class_stat['total_candidates']}\n")
            f.write(f"  符合条件数: {class_stat['qualified_candidates']}\n")
            f.write(f"  选中的客户端: {selected}\n")
            if selected:
                f.write(f"  选中客户端的得分: {[f'{s:.4f}' for s in class_stat['selected_scores']]}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("完整鉴别器映射 (class_id -> [client_id, ...]):\n")
        f.write("=" * 80 + "\n")
        for class_id in sorted(discriminator_map.keys()):
            clients = discriminator_map[class_id]
            if clients:
                f.write(f"  类别 {class_id:2d}: {clients}\n")
            else:
                f.write(f"  类别 {class_id:2d}: [] (无符合条件的鉴别器)\n")
    
    print(f"[OK] Saved discriminator_report.txt: {report_path}")


def main() -> int:
    p = argparse.ArgumentParser(
        description="从 client_owned_metrics.csv 中选择 topK 个最优模型作为鉴别器"
    )
    p.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="client_owned_metrics.csv 文件路径",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="输出目录；默认与 CSV 文件同目录下的 discriminator_selection",
    )
    p.add_argument(
        "--k_discriminators",
        type=int,
        default=3,
        help="每个类选择的鉴别器数量（默认 3）",
    )
    p.add_argument(
        "--n_samples_min",
        type=int,
        default=10,
        help="每类最少样本数门槛（默认 10）",
    )
    p.add_argument(
        "--weight_test",
        type=float,
        default=0.5,
        help="Test 集权重（默认 0.5）",
    )
    p.add_argument(
        "--weight_all",
        type=float,
        default=0.5,
        help="All 集权重（默认 0.5）",
    )
    p.add_argument(
        "--acc_weight",
        type=float,
        default=0.7,
        help="准确率权重（默认 0.7）",
    )
    p.add_argument(
        "--loss_weight",
        type=float,
        default=0.3,
        help="损失权重（默认 0.3）",
    )
    p.add_argument(
        "--no_normalize_loss",
        action="store_true",
        help="不使用损失归一化（默认使用 sigmoid 归一化）",
    )
    p.add_argument(
        "--loss_max",
        type=float,
        default=5.0,
        help="损失的最大值，用于归一化（默认 5.0）",
    )
    
    args = p.parse_args()
    
    # 验证参数
    if args.weight_test + args.weight_all != 1.0:
        print(f"[WARNING] weight_test + weight_all = {args.weight_test + args.weight_all} != 1.0，将自动归一化")
        total = args.weight_test + args.weight_all
        args.weight_test /= total
        args.weight_all /= total
    
    if args.acc_weight + args.loss_weight != 1.0:
        print(f"[WARNING] acc_weight + loss_weight = {args.acc_weight + args.loss_weight} != 1.0，将自动归一化")
        total = args.acc_weight + args.loss_weight
        args.acc_weight /= total
        args.loss_weight /= total
    
    # 确定输出目录
    if args.out_dir is None:
        csv_dir = os.path.dirname(os.path.abspath(args.csv_path))
        args.out_dir = os.path.join(csv_dir, "discriminator_selection")
    
    print("=" * 80)
    print("鉴别器选择工具")
    print("=" * 80)
    print(f"输入 CSV: {args.csv_path}")
    print(f"输出目录: {args.out_dir}")
    print(f"参数: k={args.k_discriminators}, n_min={args.n_samples_min}")
    print(f"       weight_test={args.weight_test:.2f}, weight_all={args.weight_all:.2f}")
    print(f"       acc_weight={args.acc_weight:.2f}, loss_weight={args.loss_weight:.2f}")
    print("=" * 80)
    
    # 构建鉴别器映射
    try:
        discriminator_map, scores_df, summary = build_discriminator_map(
            csv_path=args.csv_path,
            k_discriminators=args.k_discriminators,
            n_samples_min=args.n_samples_min,
            weight_test=args.weight_test,
            weight_all=args.weight_all,
            acc_weight=args.acc_weight,
            loss_weight=args.loss_weight,
            normalize_loss=not args.no_normalize_loss,
            loss_max=args.loss_max,
        )
        
        # 保存结果
        save_results(
            out_dir=args.out_dir,
            discriminator_map=discriminator_map,
            scores_df=scores_df,
            summary=summary,
            csv_path=args.csv_path,
        )
        
        print("\n" + "=" * 80)
        print("选择完成！")
        print("=" * 80)
        print(f"共为 {summary['statistics']['classes_with_discriminators']} 个类别选择了鉴别器")
        print(f"平均得分: {summary['score_statistics']['mean_final_score']:.4f} ± {summary['score_statistics']['std_final_score']:.4f}")
        print(f"\n结果文件保存在: {args.out_dir}")
        
        return 0
        
    except Exception as e:
        print(f"\n[ERROR] 处理失败: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


## classifier_fine_tunning (multi-stage, raw-high space)

本目录实现一个**离线多阶段**流水线（不改动 Stage-1 联邦训练），用于：

- **Stage-2**：统计 **raw 高级特征**（例如 `CNNCifar.fc0` 的 ReLU 输出空间，对应 `high_level_features_raw`）的全局分布统计：
  - 每类 `mean / cov / rff_mean / sample_per_class`
- **Stage-3**：使用 **SAFS**（`lib/safs.py`）基于 Stage-2 的统计量合成 **raw 高级特征**
- **Stage-4**：使用合成的 raw 高级特征 **只微调分类头**（`fc1+fc2`），并在每个 client 的 test split 上评估

### 为什么选 raw（方案A）

为了让 Stage-2 的 `mean/cov/rff_mean` 与 Stage-3 SAFS 的对齐目标/核均值约束严格处于同一空间，
本流水线 **统一在 raw-high 空间**做统计与合成（RFF 也喂 raw-high）。

### 产物约定

- Stage-2 输出：`<out_dir>/global_high_stats_raw.pt`
- Stage-3 输出：`<out_dir>/class_syn_high_raw.pt`
- Stage-4 输出：
  - `client_XX_head.pt`（每个 client 的 head 权重）
  - `stage4_results.json`（汇总 before/after）

### 运行方式（示例）

1) Stage-2（统计 raw-high）：

```bash
python classifier_fine_tunning/stage2_high_stats_raw.py ^
  --stage1_ckpt_path "<log_dir>\\stage1\\ckpts\\best-wo.pt" ^
  --out_dir "<log_dir>\\classifier_finetune\\stage2" ^
  --rf_dim 3000 --rbf_gamma 0.01 --rf_type orf
```

2) Stage-3（SAFS 合成 raw-high）：

```bash
python classifier_fine_tunning/stage3_safs_high_raw.py ^
  --stats_path "<log_dir>\\classifier_finetune\\stage2\\global_high_stats_raw.pt" ^
  --out_dir "<log_dir>\\classifier_finetune\\stage3" ^
  --steps 200 --lr 0.1 --max_syn_num 2000 --min_syn_num 600
```

3) Stage-4（仅微调 head：fc1+fc2）：

```bash
python classifier_fine_tunning/stage4_finetune_head_raw.py ^
  --stage1_ckpt_path "<log_dir>\\stage1\\ckpts\\best-wo.pt" ^
  --split_path "<log_dir>\\split.pkl" ^
  --syn_path "<log_dir>\\classifier_finetune\\stage3\\class_syn_high_raw.pt" ^
  --out_dir "<log_dir>\\classifier_finetune\\stage4" ^
  --steps 2000 --batch_size 128 --lr 1e-4
```


### 我建议你先做的 3 个实验（投入小、信息量大）

- Exp-A（最推荐起步）：synthetic 样本 距离过滤 + syn_weight 线性 warmup。

- Exp-B（对比学习路线）：real/syn 成对生成 + 在 projected_features 上加 MySupConLoss（同时把 projector 加进 optimizer）。

- Exp-C（对齐路线）：对 x_syn_id 加 high_mean 对齐项（把你已有 OOC 正则思路复用到 ID 生成样本）。
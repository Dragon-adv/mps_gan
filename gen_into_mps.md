我先快速摸清你现有 FedMPS 多阶段训练框架里 **“stage 如何切换、客户端/服务器每轮做什么、模型/ckpt怎么保存、特征在哪里取”**，再对照 FedGen 的“条件生成/蒸馏”范式，给出一套 **可落地的引入 Generator 的分阶段方案**（训练目标、数据流、选优本地模型策略、评估指标与消融、实现落点到哪些模块）。接下来我会

先在仓库里定位 `stage`、`Generator`/`GAN`、`ckpt`、`feature` 相关实现。

[4 tools called]

Nice：你这个仓库里其实已经有一条“**Stage-2 聚合低级特征统计量 + SAFS 合成特征**”的路线（`lib/safs.py`、`feature_synthesis()`、`fine_tune_global_model_safs()`），并且还内置了完整的 `FedGen/` 参考实现。下面我会把 `exps/federated_main.py` 里 stage2 的数据流、以及 `models.py` 输出的 low/high/proj 特征接口读出来，然后基于“SAFS→Generator”的替换/并行思路给你一份可执行的多阶段规划（含你说的“每类选优本地模型做合成样本验证”的评估闭环）。

[5 tools called]

## 你现在的框架里，Generator 最自然的“插入点”在哪里

从代码结构看，你的多阶段入口已经成型：

- **Stage-1**：正常联邦训练，产出每个客户端的完整模型权重（以及可拆分的 low/high/projector/head 组件）。
- **Stage-2**：**离线加载 Stage-1 checkpoint**，重建所有客户端模型，并对特征做**全局统计量聚合**（目前主要支撑 SAFS 的特征合成/微调）；Stage-2 运行完会直接退出，不进入联邦训练循环。

相关证据（你已经把“多阶段”接口打通了）：

```163:188:d:\cursor_workspace\FedMPS_GAN\mps_gan\lib\options.py
    parser.add_argument('--stage', type=int, default=1,
                        help='Training stage selector (reserved for multi-stage pipeline). Default: 1')
    # ===================== Stage-2 statistics aggregation =====================
    parser.add_argument('--stage1_ckpt_path', type=str, default=None,
                        help='[Stage-2] Path to a Stage-1 checkpoint to load (e.g., <log_dir>/stage1_ckpts/best-wo.pt). '
                             'If provided, stage-2 will infer log_dir/split_path from the checkpoint meta when possible.')
    parser.add_argument('--stage2_out_dir', type=str, default=None,
                        help='[Stage-2] Output directory to save stage-2 aggregated statistics. '
                             'If None, defaults to <ckpt.meta.logdir>/stage2_stats (or <log_dir>/stage2_stats).')
    parser.add_argument('--save_stage1_components', type=int, default=1,
                        help='Whether to export 4-component state_dicts (low/high/projector/head) into checkpoints (0/1). Default: 1')
```

```1133:1257:d:\cursor_workspace\FedMPS_GAN\mps_gan\exps\federated_main.py
    # ===================== Stage-2: Statistics Aggregation (low-only) =====================
    stage = int(getattr(args, 'stage', 1))
    stage2_payload = None
    if stage == 2:
        if getattr(args, 'stage1_ckpt_path', None) is None:
            raise ValueError('Stage-2 requires --stage1_ckpt_path (e.g., <log_dir>/stage1_ckpts/best-wo.pt).')
        ...
    # Run Stage-2 early and exit (no training loop)
    if stage == 2:
        ...
        # Stage-2 requires a persisted split for consistency
        if getattr(args, 'split_path', None) is None:
            raise ValueError('Stage-2 requires split_path (can be inferred from checkpoint meta).')
        if not os.path.exists(args.split_path):
            raise FileNotFoundError(...)
        loaded_split = load_split(args.split_path)
        ...
        # Rebuild client models and load per-client weights from Stage-1 checkpoint
        local_model_list = []
        for i in range(args.num_users):
            ...
            local_model_list.append(local_model)
        local_sd = ckpt_state.get('local_models_full_state_dicts', None)
        ...
        for cid, m in enumerate(local_model_list):
            ...
            m.load_state_dict(local_sd[key], strict=True)
```

同时，你的模型 forward 已经明确产出了 **low / high / projected features**，这对“在低级特征空间做生成，然后用高级特征/分类头验证效果”非常匹配：

```146:174:d:\cursor_workspace\FedMPS_GAN\mps_gan\lib\models\models.py
    def forward(self, x):
        # 1. 低级特征阶段
        feat_low_raw = self.pool(F.relu(self.conv1(x)))
        feat_low_raw = self.pool(F.relu(self.conv2(feat_low_raw)))
        feat_low_flat = feat_low_raw.view(-1, feat_low_raw.shape[1] * feat_low_raw.shape[2] * feat_low_raw.shape[3])
        low_level_features_raw = feat_low_flat
        ...
        # 2. 高级特征阶段
        feat_high_flat = feat_low_raw.view(-1, 16 * 5 * 5)
        feat_high_encoded = F.relu(self.fc0(feat_high_flat))
        high_level_features_raw = feat_high_encoded
        ...
        return logits, log_probs, high_level_features_raw, low_level_features_raw, projected_features
```

**结论**：把 Generator 引入你这个 FedMPS 多阶段框架，最稳妥的路径是——  
**保留 Stage-1（学到本地表征）→ 扩展 Stage-2（聚合统计 + 选 teacher）→ 新增 Stage-3（训练 generator/产出合成低级特征）→ 新增 Stage-4（客户端用合成低级特征补齐类别，训练高层/分类头）**。  
这条路完全贴合你 README 里描述的“低级特征+噪声保护→全局训练GAN→下发补齐→训练高级编码器/分类头”。

---

## 目标拆解：你想验证的“低级 vs 高级”到底是什么

你提出的核心实验问题其实可以拆成两类指标闭环：

- **A. 合成的是不是“有用的低级特征”**  
  把合成低级特征喂给（若干）本地模型的 **高层编码器**，看生成出来的 **高级特征**是否：
  - 能被“该类的优质本地模型”稳定判为该类（teacher-consensus）。
  - 在高级特征空间里靠近真实该类的高层原型/均值，且多样性足够（不塌缩）。

- **B. 合成低级特征能不能带来“后续分类器收益”**  
  用“真实低级特征 + 合成低级特征”训练高层/头部（可冻结 low encoder），看：
  - 客户端平均准确率、长尾类准确率、方差（公平性）是否提升。
  - 对缺失类/稀少类的提升是否显著。

这两类对应你说的：“如果合成的是低级特征，就通过运行后的高级特征效果 + 后续分类器效果判断”。

---

## 详细规划：在你现有多阶段框架里加入 Generator（推荐方案）

### Stage-1（保持不变，但建议新增“选优所需的记录”）
**输入**：各客户端本地数据  
**输出**：  
- `local_models_full_state_dicts`（你 Stage-2 已依赖）  
- 每个客户端、每个类别的“质量分数”所需统计（建议加到 ckpt meta 或单独文件）：
  - **class-wise accuracy / loss**（在本地验证集或本地 test split 上）
  - **class sample count**（防止“样本太少但 acc 虚高”）

**为什么要加这步**：你后面要“每类选几个优质本地模型”，没有 class-wise 指标就只能用全局 acc（会选偏：某客户端对某类很差但总体好）。

---

### Stage-2（扩展：继续做统计聚合 + 做 per-class teacher 选择）
你 Stage-2 已经做了“加载 Stage-1 ckpt → 重建所有本地模型 → 聚合统计”（且 SAFS 用到了 global mean/cov/RFF mean）。我们只需要在 Stage-2 里再产出一个“teacher 清单”。

**Stage-2 输出建议**（落到 `log_dir/stage2_stats/`）：
- **全局统计量**（你已有/或接近已有）：每类 low/high 的均值、协方差、RFF mean 等（供 generator 训练的分布约束项）。
- **teacher_map**：`{class c: [client_id1, client_id2, ...]}`

**teacher 选择策略（强烈建议可解释 + 可复现）**：
- 候选集：只从“本地确实拥有该类”的客户端里选（例如 `classes_list` 约束）。
- 排序分数（示例）：  
  \( s_{i,c} = \text{Acc}_{i,c} - \lambda \cdot \text{Loss}_{i,c} \)  
  并加两个约束：
  - `n_{i,c} >= n_min`（样本太少不选）
  - 选 `TopK`（K=2~5 通常足够）
- 同时输出“备选老师”（TopK+M），以便做消融：K 对效果的影响。

---

### Stage-3（新增：FedGen 风格训练 Generator，但在“低级特征空间”做生成）
这是关键：**generator 生成的是 low-level feature（或 low feature map），不是图片**。原因：
- 你模型已经把 low/high 明确分层，且你后续就是要用 low 合成去训练 high/head；
- 生成图片的训练成本、稳定性、隐私风险都更大；
- 你现有 SAFS 已经在特征层做合成，说明这条路线与你代码生态一致。

#### 3.1 生成器形式（建议从简单 MLP 条件生成开始）
- 输入：`(y, z)`  
  - `y`：类别标签（embedding）
  - `z`：高斯噪声
- 输出：`x_low_syn`（合成低级特征向量，维度=你 low_level_features_raw 的维度）

> 由于你 low feature 是从 feature map flatten 得到的，只要维度固定，就可以 reshape 回去，再接上 teacher 的后半段网络（conv2 之后/或 high encoder）。

#### 3.2 训练信号（把 FedGen “teacher/student/diversity”思想对齐到你这里）
你想“每类选优质本地模型跑一下合成样本”，这其实天然就是 **teacher ensemble**。

对每个 class c：
- teacher 集合：`T_c = {model_i | i in teacher_map[c]}`
- 对每个 batch 合成特征 `x_low_syn`，把它喂给每个 teacher 的后半段网络，得到：
  - logits：`p_i(y | x_low_syn)`
  - 高级特征：`h_i(x_low_syn)`（取 high_level_features_raw 或其 normalize）

推荐损失（按“必须/可选”分层）：

- **必须：Teacher 分类一致性（“能被判成该类”）**  
  - `L_cls = avg_i CE(p_i, y)`  
  或者用 ensemble logit：`p̄ = avg_i p_i`，`CE(p̄, y)`
- **必须：多样性/防塌缩（FedGen diversity 对应物）**  
  - 在 high 或 proj 空间做 repulsion：同类样本之间最大化 pairwise distance / 协方差体积
- **强烈建议：分布约束（对齐你 Stage-2 的统计量/或 SAFS 的 RFF 约束）**  
  - `L_stat_mean/cov`：让合成 low（或合成 high）在 batch 统计上匹配全局每类 mean/cov  
  - 或 `L_rff`：匹配每类 RFF mean（你 SAFS 已经用 `rf_model` 体系）
- **可选：跨 teacher 一致性（减少“只迎合某个 teacher”）**  
  - KL(p_i || p̄) 的和，让不同 teacher 对同一合成样本的预测一致

最终：`L = α L_cls + β L_div + γ L_stat + δ L_consistency`

> 这套和 FedGen 的精神一致：**用多个本地模型提供“监督/蒸馏”信号，同时用多样性正则避免 generator 塌缩**。区别只是：FedGen 更常见是生成“latent 表征”，你这里生成的是“low-level feature”。

#### 3.3 Stage-3 的产物
- `generator.pt`（模型权重）
- `syn_bank.pkl/pt`（可选）：每类预生成 N 个 low features，用于 Stage-4 快速下发/复现实验

---

### Stage-4（新增：客户端利用合成低级特征补齐稀缺类，训练高层/分类头）
**核心做法**（和你 README 描述一致）：

- 客户端本地：用真实图片过 low encoder 得到 `x_low_real`
- 从服务器拿到 `x_low_syn`（带标签）
- 混合组成训练集：`D_low = {(x_low_real, y_real)} ∪ {(x_low_syn, y_syn)}`
- 训练对象建议从稳到激进分两档：
  - **稳妥版**：冻结 low encoder，仅训练 `high_encoder + projector + classifier_head`
  - **激进版**：允许小学习率联合微调 high/head，low 仍冻结（或只对 BN/最后层解冻）

**你想要的两类验证在 Stage-4 都能实现**：

- 看高级特征：训练过程中/训练后，比较 `h(x_low_syn)` 与 `h(x_low_real)` 的类内/类间距离
- 看分类器效果：直接看各客户端 test acc、长尾类 acc、公平性 std

---

## 你提到的“每类选优质本地模型跑合成样本”的评估闭环（建议固定成标准报告）

对每个 class c、每个 teacher i ∈ T_c，抽取合成样本集合 `S_c`：

- **Teacher-consensus 指标（最直观）**
  - `Acc_teacher_i(c) = mean_{x∈S_c} [argmax p_i(x) == c]`
  - `Acc_ensemble(c)`：对 ensemble 预测同样算一次
  - `Var_teacher(c)`：不同 teacher 的一致性（方差越小越说明“合成样本是通用特征”）

- **高级特征质量**
  - `|| mean(h_syn^c) - proto_high^c ||`（或与真实 high mean 的距离）
  - `intra/inter`：类内散度 vs 类间间隔（看是否塌缩/混类）

- **下游分类收益（最终 KPI）**
  - 客户端平均 acc、长尾类 acc、std(acc)（公平性）
  - 特别关注“缺失类/稀少类”的提升幅度（这是 generator 的价值点）

---

## 工程落地建议（按里程碑切，避免一次改太大）

- **M1（最快验证）：离线 Stage-3，只做“合成→teacher 评估”**  
  不下发给客户端训练，先把你关心的“低→高特征是否合理”跑通。
- **M2：加 Stage-4 客户端高层训练**  
  看是否提升长尾/缺失类；这一步才决定 generator 是否值得继续投入。
- **M3：把统计约束（mean/cov/RFF）接入 generator loss**  
  让生成不只是“迎合 teacher”，而是更接近全局分布（与现有 SAFS 体系强耦合，收益通常明显）。
- **M4：联邦化/隐私强化（如果你需要更真实的 FL）**  
  例如不把完整本地模型权重集中到服务器，而是让客户端参与训练 generator（更复杂，但更贴近 FedGen 原论文设定）。

---

## 关键风险点（提前避坑）

- **“生成 low feature”要能喂进 teacher 的后半段网络**：你 low feature 是 flatten 的，训练时需要固定 reshape 规则（不同 backbone/数据集 shape 不同）。
- **teacher 选择偏差**：只按总体 acc 选会严重偏；一定要用 class-wise 指标 + 样本数门槛。
- **generator 过拟合某些 teacher**：需要跨 teacher 一致性/多样性正则，否则你会看到“某些 teacher 上很好、换 teacher 就崩”。

---

如果你愿意，我可以下一步把你 `exps/federated_main.py` 里 Stage-2 统计量具体保存了哪些字段（mean/cov/RFF/proto）再精确对齐到 Stage-3 的 loss 设计，并给出一份“建议新增的参数表（--stage 3/4 的 args）+ 实验脚本组织方式（类似你 `run_stage1.py`）”。你现在更倾向先做 **M1 离线评估**，还是直接冲 **M2 下发训练看最终精度**？
import torch
import torch.nn as nn
import torch.nn.functional as F


class NTD_Loss(nn.Module):
    """
    Not-true Distillation Loss (非真值蒸馏损失)
    该损失函数出自论文: "Preserving Privacy and Efficiency in Federated Learning via Goldfish Loss"
    或相关的联邦学习逻辑蒸馏研究。
    """

    def __init__(self, num_classes=10, tau=3, beta=1):
        super(NTD_Loss, self).__init__()
        self.CE = nn.CrossEntropyLoss()
        self.MSE = nn.MSELoss()
        self.KLDiv = nn.KLDivLoss(reduction="batchmean")  # 计算KL散度，按batch取平均
        self.num_classes = num_classes
        self.tau = tau  # 温度系数 (Temperature)
        self.beta = beta  # NTD损失的权重系数

    def forward(self, logits, targets, dg_logits):
        """
        Args:
            logits: 当前模型的输出预测 (Local model logits)
            targets: 样本的真实标签 (Ground truth labels)
            dg_logits: 指导模型的输出预测 (Global/Teacher model logits)
        """
        # 1. 标准交叉熵损失：保证模型对真值类别的预测准确性
        ce_loss = self.CE(logits, targets)

        # 2. 计算 NTD 损失：学习非真值类别之间的相对分布结构
        ntd_loss = self._ntd_loss(logits, dg_logits, targets)

        # 总损失 = 分类损失 + 权重 * 蒸馏损失
        loss = ce_loss + self.beta * ntd_loss

        return loss

    def _ntd_loss(self, logits, dg_logits, targets):
        """Not-tue Distillation Loss 计算“非正确类别”部分的蒸馏损失"""

        # 提取当前模型中除了真值以外的 logits (Shape: [Batch, Num_classes-1])
        logits = self.refine_as_not_true(logits, targets, self.num_classes)
        # 对非真值部分进行温度缩放并计算 Log Softmax
        pred_probs = F.log_softmax(logits / self.tau, dim=1)

        # 提取全局/教师模型中除了真值以外的 logits
        with torch.no_grad():
            dg_logits = self.refine_as_not_true(dg_logits, targets, self.num_classes)
            # 对教师模型输出进行温度缩放并计算 Softmax
            dg_probs = torch.softmax(dg_logits / self.tau, dim=1)

        # 计算 KL 散度，乘以 tau^2 是为了保持梯度量级的稳定性
        loss = (self.tau ** 2) * self.KLDiv(pred_probs, dg_probs)

        return loss

    def refine_as_not_true(self, logits, targets, num_classes):
        """
        核心操作：从 logits 中剔除 Ground Truth 对应的数值。
        例如：logits=[a, b, c], target=1 (即类别b), 则返回 [a, c]
        """
        # 生成一个 [0, 1, 2, ..., num_classes-1] 的序列
        nt_positions = torch.arange(0, num_classes).to(logits.device)
        # 广播至 [Batch_size, num_classes]
        nt_positions = nt_positions.repeat(logits.size(0), 1)

        # 逻辑掩码：筛选出所有不等于 target 的索引位置
        # 这一步会将每一行中对应真值标签的那个索引删掉
        nt_positions = nt_positions[nt_positions[:, :] != targets.view(-1, 1)]

        # 将一维序列重新 view 成 [Batch_size, num_classes - 1]
        nt_positions = nt_positions.view(-1, num_classes - 1)

        # 使用 gather 根据索引从原始 logits 中提取出“非真值”部分的 logits
        logits = torch.gather(logits, 1, nt_positions)

        return logits
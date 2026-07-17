"""ALTAS 四组件: Generator(Gen) / Extractor(Ext) / Critic(C) / predictor(P)。"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    """Gen: 生成掩码。返回 ``(pi, mask)``:
    - ``pi``:   Gumbel-Softmax 注入噪声前的"保留"概率期望 (用于 L1 期望惩罚)
    - ``mask``: Gumbel-Softmax 采样得到的近似 0/1 掩码 (送入 Ext/C/P)

    较深的 MLP 让网络能在内部特征流中自主合成出"控制逻辑", 以适应
    INVASE 论文 Syn4-Syn6 这种带 switch 特征的条件分布。
    """

    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.input_dim = input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim * 2),
        )

    def forward(self, x, tau=1.0, hard=True):
        logits = self.net(x).view(-1, self.input_dim, 2)
        probs = F.softmax(logits, dim=-1)
        pi = probs[:, :, 1]
        samples = F.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)
        mask = samples[:, :, 1]
        return pi, mask


class Extractor(nn.Module):
    """Ext: 共享特征提取层，输出 latent representation h。"""

    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x):
        return self.net(x)


class Critic(nn.Module):
    """C: 判别 h 来自完整特征还是掩码特征（接收 Ext 的输出）。"""

    def __init__(self, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, h):
        return self.net(h)


class predictor(nn.Module):
    """P: 下游任务分类（接收 Ext 的输出）。"""

    def __init__(self, hidden_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, h):
        return self.net(h)

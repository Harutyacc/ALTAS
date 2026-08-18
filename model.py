"""ALTAS 四组件: Generator(Gen) / Extractor(Ext) / Critic(C) / predictor(P)。

参考 INVASE 论文官方实现的层数 / 隐层维度 / 层间归一化风格:
    INVASE_Selector  -> Generator   : 3 层 Linear + LayerNorm + ReLU, hidden_dim=100
    INVASE_Predictor -> predictor   : 3 层 Linear + BatchNorm1d + ReLU, hidden_dim=200 (内部)
    Extractor / Critic 按 ALTAS 自身需要 (共享表征 + WGAN-GP) 仿照同样骨架。

注: Generator 末层输出 ``input_dim * 2`` 通道 logits 供 ``F.gumbel_softmax``
采样硬掩码 (ALTAS 训练链路依赖 Gumbel-Softmax, 这里保留); 其余三个组件的
输出层不带额外激活 (Predictor 让 CrossEntropyLoss 处理 softmax)。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    """Gen: 生成掩码 (仿 INVASE_Selector)。

    3 层 Linear + LayerNorm + ReLU, hidden_dim=100 (默认); 末层输出 2-channel
    logits 供 ``F.gumbel_softmax`` 采样硬掩码。

    返回 ``(pi, mask)``:
      - ``pi``:   Gumbel-Softmax 注入噪声前的"保留"概率期望 (用于 L1 期望惩罚)
      - ``mask``: Gumbel-Softmax 采样得到的近似 0/1 掩码 (送入 Ext/C/P 链路)
    """

    def __init__(self, input_dim, hidden_dim=100):
        super().__init__()
        self.input_dim = input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim * 2),
        )
        # 初始化末层：让“保留通道 (channel 1)”的 bias 稍微大一点，初期保留率约 40%
        with torch.no_grad():
            self.net[-1].bias.data[:input_dim] = 0.0      # 丢弃通道
            self.net[-1].bias.data[input_dim:] = 0.5     # 保留通道

    def forward(self, x, tau=1.0, hard=True):
        logits = self.net(x).view(-1, self.input_dim, 2)
        probs = F.softmax(logits, dim=-1)
        pi = probs[:, :, 1]
        samples = F.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)
        mask = samples[:, :, 1]
        return pi, mask


class Extractor(nn.Module):
    """Ext: 共享特征提取, 输出 latent h (仿 INVASE_Predictor 主干)。

    3 层 Linear + BatchNorm1d + ReLU (对齐 INVASE 风格); 输出 hidden_dim。
    """

    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x):
        return self.net(x)


class Critic(nn.Module):
    """C: 判别 h (WGAN-GP 判别器)。

    仿 INVASE_Predictor 的 3 层主干, 但去掉 BatchNorm
    (WGAN-GP 经验上不用 BN, 会破坏梯度惩罚约束), 用 LeakyReLU 替代 ReLU。
    """

    def __init__(self, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h):
        return self.net(h)


class predictor(nn.Module):
    """P: 分类 (完整对齐 INVASE_Predictor)。

    3 层 Linear + BatchNorm1d + ReLU, 内部隐层 200 (对齐 INVASE_Predictor);
    无 Dropout, 末层直接输出 num_classes logits (CrossEntropyLoss 自带 Softmax)。
    """

    def __init__(self, hidden_dim, num_classes, internal_hidden=200):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, internal_hidden),
            nn.BatchNorm1d(internal_hidden),
            nn.ReLU(),
            nn.Linear(internal_hidden, internal_hidden),
            nn.BatchNorm1d(internal_hidden),
            nn.ReLU(),
            nn.Linear(internal_hidden, num_classes),
        )

    def forward(self, h):
        return self.net(h)
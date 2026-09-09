"""ALTAS 的四个神经网络组件。

模块仅定义网络前向计算，不包含优化器或训练流程。
"""

import torch
from torch import nn
from torch.nn import functional as F


class FeatureMaskGenerator(nn.Module):
    """为每个样本生成实例级特征掩码。

    网络为每个输入特征输出“丢弃/保留”两个 logits，并通过
    Gumbel-Softmax 得到可反向传播的近似二值掩码。
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim * 2),
        )

    def forward(
        self,
        inputs: torch.Tensor,
        temperature: float = 1.0,
        hard: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """返回特征保留概率和采样后的掩码，形状均为 ``[B, D]``。"""
        logits = self.network(inputs).view(-1, self.input_dim, 2)
        retention_probabilities = F.softmax(logits, dim=-1)[:, :, 1]
        samples = F.gumbel_softmax(logits, tau=temperature, hard=hard, dim=-1)
        mask = samples[:, :, 1]
        return retention_probabilities, mask


class FeatureExtractor(nn.Module):
    """将原始或掩码输入映射到共享隐空间。"""

    def __init__(self, input_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """返回形状为 ``[B, hidden_dim]`` 的隐表示。"""
        return self.network(inputs)


class LatentCritic(nn.Module):
    """判断隐表示来自完整输入还是掩码输入。"""

    def __init__(self, hidden_dim: int = 64) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, latent_features: torch.Tensor) -> torch.Tensor:
        """为每个隐表示返回一个未归一化评分。"""
        return self.network(latent_features)


class TaskPredictor(nn.Module):
    """根据共享隐表示完成下游分类任务。"""

    def __init__(self, hidden_dim: int, num_classes: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, latent_features: torch.Tensor) -> torch.Tensor:
        """返回形状为 ``[B, num_classes]`` 的分类 logits。"""
        return self.network(latent_features)

"""ALTAS 实验的集中配置。

配置对象仅描述数据、模型和训练超参数，不在模块导入时创建全局状态。
"""

from dataclasses import dataclass


TRUE_FEATURE_DIM = 11
"""合成数据中可能参与标签生成的特征总数（索引 0 至 10）。"""

TOP_K_DISPLAY_MULTIPLIER = 3
"""报告中展示的特征数相对于平均保留特征数的倍数。"""

PLOT_STYLE = "seaborn-v0_8-whitegrid"
FIGURE_DPI = 300


@dataclass(frozen=True, slots=True)
class DataConfig:
    """合成数据生成和数据加载配置。"""

    num_samples: int = 20_000
    input_dim: int = 100
    dataset_type: str = "Syn1"
    batch_size: int = 256
    train_ratio: float = 0.8


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """模型结构配置。"""

    input_dim: int = 100
    num_classes: int = 2
    hidden_dim: int = 64


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """优化器、损失权重和训练循环配置。"""

    epochs: int = 1_000
    generator_learning_rate: float = 1e-5
    task_learning_rate: float = 1e-4
    adversarial_weight: float = 1.0
    prediction_weight: float = 1.0
    sparsity_weight: float = 0.3
    critic_steps: int = 2
    gradient_penalty_weight: float = 10.0
    temperature_start: float = 1.0
    temperature_min: float = 0.1
    temperature_decay: float = 0.995
    log_interval: int = 10
    masked_prediction_weight: float = 0.5

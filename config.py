"""集中放置实验超参数与常量，避免散落在 main/训练循环各处。"""

from dataclasses import dataclass


TRUE_FEATURE_DIM = 11
TOP_K_PRINT_MULTIPLIER = 3

DATA_STYLE = "seaborn-v0_8-whitegrid"
FIGURE_DPI = 300
TSNE_PERPLEXITY = 30
TSNE_MAX_ITER = 1000
TSNE_RANDOM_STATE = 42


@dataclass
class DataConfig:
    num_samples: int = 20000
    input_dim: int = 100
    dataset_type: str = "Syn4"
    batch_size: int = 256
    train_ratio: float = 0.8


@dataclass
class TrainConfig:
    epochs: int = 500
    lr_gen: float = 1e-5          # 提高 Generator 学习率以匹配博弈速度
    lr_ep: float = 1e-4
    
    # ─── 纳什均衡博弈动态权重 ───────────────────────────
    alpha: float = 1.0            # 对抗损失权重 (信息阻尼项)
    beta: float = 1.0             # 分类损失权重 (关键：从 0 改为 1.0，引入任务监督)
    gamma_init: float = 0.5       # L1 初始稀疏度乘子
    lr_gamma: float = 0.01        # 对偶乘子 gamma 的自适应学习率
    target_w_dist: float = 1   # 隐空间允许的最大流形偏离容限 epsilon_W
    # ──────────────────────────────────────────────────
    
    n_critic: int = 2             # 快-慢时间尺度比 (TTSA 收敛保证)
    lambda_gp: float = 10.0
    tau_start: float = 1.0
    tau_min: float = 0.1
    tau_decay: float = 0.995
    log_every: int = 10
    p_mask_weight: float = 0.5


@dataclass
class ModelConfig:
    input_dim: int = 100
    num_classes: int = 2
    hidden_dim: int = 64
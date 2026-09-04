"""数据集构造、DataLoader、特征选择评估与可视化工具。"""

from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, TensorDataset

from config import (
    DATA_STYLE,
    FIGURE_DPI,
    TRUE_FEATURE_DIM,
    TSNE_MAX_ITER,
    TSNE_PERPLEXITY,
    TSNE_RANDOM_STATE,
)


_CLASS_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]


plt.style.use(DATA_STYLE)


def generate_synthetic_data(
    num_samples: int = 2000,
    input_dim: int = 100,
    dataset_type: str = "Syn4",
    save_csv: bool = True,
    csv_filename: str | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """生成 INVASE 论文 Syn1-Syn6 模拟数据。

    返回:  (X_all, Y_all, S_all)
      X_all: [num_samples, input_dim] 特征
      Y_all: [num_samples]            二分类标签
      S_all: [num_samples, input_dim] 实例级真实特征掩码
    """
    if input_dim < TRUE_FEATURE_DIM:
        raise ValueError(
            f"Syn 数据集至少需要 {TRUE_FEATURE_DIM} 个特征维度 (索引 0-{TRUE_FEATURE_DIM-1})"
        )

    X = torch.randn(num_samples, input_dim)
    S = torch.zeros(num_samples, input_dim)

    logit1 = torch.exp(X[:, 0] * X[:, 1])
    logit2 = torch.exp(X[:, 2] ** 2 + X[:, 3] ** 2 + X[:, 4] ** 2 + X[:, 5] ** 2 - 4.0)
    logit3 = torch.exp(
        -10 * torch.sin(0.2 * X[:, 6])
        + torch.abs(X[:, 7])
        + X[:, 8]
        + torch.exp(-X[:, 9])
        - 2.4
    )

    prob1 = logit1 / (1 + logit1)
    prob2 = logit2 / (1 + logit2)
    prob3 = logit3 / (1 + logit3)

    idx_neg = X[:, 10] < 0
    idx_pos = ~idx_neg

    synth_rules: Dict[str, List[Tuple]] = {
        "Syn1": [("static", prob1, (0, 2))],
        "Syn2": [("static", prob2, (2, 6))],
        "Syn3": [("static", prob3, (6, 10))],
        "Syn4": [("switch", prob1, prob2, (0, 2), (2, 6))],
        "Syn5": [("switch", prob1, prob3, (0, 2), (6, 10))],
        "Syn6": [("switch", prob2, prob3, (2, 6), (6, 10))],
    }
    if dataset_type not in synth_rules:
        raise ValueError(f"不支持的数据集类型 '{dataset_type}'（可选 Syn1-Syn6）")

    rule = synth_rules[dataset_type][0]
    if rule[0] == "static":
        _, prob, (lo, hi) = rule
        probs = prob
        S[:, lo:hi] = 1.0
    else:
        _, p_neg, p_pos, (lo_n, hi_n), (lo_p, hi_p) = rule
        probs = torch.zeros(num_samples)
        probs[idx_neg] = p_neg[idx_neg]
        probs[idx_pos] = p_pos[idx_pos]
        S[idx_neg, lo_n:hi_n] = 1.0
        S[idx_pos, lo_p:hi_p] = 1.0
        S[:, 10] = 1.0

    Y = torch.bernoulli(probs).long()

    if save_csv:
        feat_df = pd.DataFrame(X.numpy(), columns=[f"Feature_{i}" for i in range(input_dim)])
        mask_df = pd.DataFrame(S.numpy(), columns=[f"Mask_{i}" for i in range(input_dim)])
        feat_df["Target"] = Y.numpy()
        df = pd.concat([feat_df, mask_df], axis=1)
        if csv_filename is None:
            csv_filename = f"syn_{dataset_type}_n{num_samples}_d{input_dim}.csv"
        df.to_csv(csv_filename, index=False)
        print(f"[System] {dataset_type} 数据集已保存至 {csv_filename}")

    return X, Y, S


def get_dataloaders(
    X_all: torch.Tensor,
    Y_all: torch.Tensor,
    S_all: torch.Tensor,
    batch_size: int = 64,
    train_ratio: float = 0.8,
) -> Tuple[DataLoader, DataLoader]:
    """按 train_ratio 顺序切分 DataLoader，每个 batch 返回 (x, y, s)。"""
    num_samples = len(X_all)
    train_size = int(train_ratio * num_samples)

    train_ds = TensorDataset(X_all[:train_size], Y_all[:train_size], S_all[:train_size])
    test_ds = TensorDataset(X_all[train_size:], Y_all[train_size:], S_all[train_size:])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def evaluate_feature_selection(
    S_pred: torch.Tensor, S_true: torch.Tensor
) -> Tuple[float, float]:
    """按实例计算 TPR/FDR（百分比）。

    Args:
        S_pred: Generator 输出的硬掩码, 形状 ``[batch, input_dim]``, 元素为 0 / 1。
                推导流程见 ``evaluate.collect_test_outputs``: 用 ``tau=1e-3, hard=True``
                调 Generator 直接拿到 one-hot 化的 mask, 因此这里无需再二值化。
        S_true: DataLoader 中的真实掩码 ground truth, 同样为 0 / 1。

    Returns:
        ``(mean_tpr, mean_fdr)``, 单位为百分比。
    """
    if S_pred.shape != S_true.shape:
        raise ValueError(f"S_pred 与 S_true 形状不一致: {S_pred.shape} vs {S_true.shape}")

    S_pred_bin = S_pred.float()
    S_true_f = S_true.float()

    tp = torch.sum(S_pred_bin * S_true_f, dim=1)
    actual_pos = torch.sum(S_true_f, dim=1)
    predicted_pos = torch.sum(S_pred_bin, dim=1)
    fp = predicted_pos - tp

    eps = 1e-8
    tpr = tp / (actual_pos + eps)

    fdr = torch.zeros_like(tpr)
    has_pred = predicted_pos > 0
    fdr[has_pred] = fp[has_pred] / predicted_pos[has_pred]

    return tpr.mean().item() * 100.0, fdr.mean().item() * 100.0


def plot_comparative_tsne(
    raw_features: np.ndarray,
    latent_features: np.ndarray,
    labels: np.ndarray,
    save_path: str = "fig3_tsne_comparison.png",
) -> None:
    """画 1x2 双子图，对比 X_mask 与 h_fake 的 t-SNE 流形。"""
    print(f"\n[Visualization] 运行双视角 t-SNE 降维 (样本数: {len(raw_features)})...")

    common: Dict[str, Any] = dict(
        n_components=2,
        random_state=TSNE_RANDOM_STATE,
        perplexity=TSNE_PERPLEXITY,
        max_iter=TSNE_MAX_ITER,
    )
    print("[Visualization] 降维: 原始掩码特征 (Before Ext)...")
    raw_2d = TSNE(**common).fit_transform(raw_features)
    print("[Visualization] 降维: 隐空间特征 (After Ext)...")
    latent_2d = TSNE(**common).fit_transform(latent_features)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    classes = np.unique(labels)

    titles = [
        ("Before Ext: Raw Masked Features ($X_{mask}$)", raw_2d),
        ("After Ext: Latent Manifold ($h_{fake}$)", latent_2d),
    ]

    for ax, (title, xy) in zip(axes, titles):
        for i, cls in enumerate(classes):
            idx = labels == cls
            ax.scatter(
                xy[idx, 0],
                xy[idx, 1],
                label=f"Class {int(cls)}",
                alpha=0.75,
                c=_CLASS_COLORS[i % len(_CLASS_COLORS)],
                edgecolors="w",
                linewidth=0.5,
                s=40,
            )
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("t-SNE Dimension 1", fontsize=12)
        ax.legend(fontsize=11)
    axes[0].set_ylabel("t-SNE Dimension 2", fontsize=12)

    plt.suptitle(
        "Comparing masked input X⊙mask vs. latent h after Extractor (t-SNE)",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"[Visualization] 对比 t-SNE 已保存至: '{save_path}'\n")


def smooth(y, window: int = 21) -> np.ndarray:
    """中心移动平均, 首尾用边界值填充。"""
    y = np.asarray(y, dtype=float)
    if window <= 1 or len(y) < window:
        return y
    w = window
    pad = w // 2
    padded = np.concatenate([np.full(pad, y[0]), y, np.full(pad, y[-1])])
    kernel = np.ones(w) / w
    return np.convolve(padded, kernel, mode="valid")[: len(y)]


def plot_training_loss(
    history: Dict[str, list], save_path: str = "fig1_shared_loss.png"
) -> None:
    """所有损失的合成视图, 全部使用原始 epoch 值 (无平滑)。

    上: Critic / Predictor / Generator-total (共享 y 轴)
    下: Generator 三段细分 adv / L1 (共享 y 轴 / x 轴)
    每个子图独立图例, 加白底防止数据穿透遮挡。
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax_top, ax_bot = axes

    epochs = np.arange(1, len(history["loss_c"]) + 1)

    def _draw(ax, key, color, label, linestyle="-"):
        y = np.asarray(history[key])
        ax.plot(epochs, y, color=color, linewidth=1.6, label=label, linestyle=linestyle)
        return y

    _draw(ax_top, "loss_c",    "#C44E52", "$L_C$ (Critic)")
    _draw(ax_top, "loss_p",    "#4C72B0", "$L_P$ (Predictor)")
    _draw(ax_top, "loss_gen",  "#000000", "$L_{Gen}$ (total)", linestyle=":")
    ax_top.set_title("Critic / Predictor / Generator-total losses")
    ax_top.set_ylabel("Loss")
    ax_top.grid(True, alpha=0.3)
    ax_top.tick_params(labelbottom=False)
    ax_top.legend(
        loc="upper right", fontsize=10,
        frameon=True, framealpha=0.92, edgecolor="#CCCCCC",
    )

    _draw(ax_bot, "loss_gen_adv", "#8172B2", "Gen.adv")
    _draw(ax_bot, "loss_gen_l1",  "#55A467", "Gen.L1")
    ax_bot.set_title("Generator decomposition: adv / L1")
    ax_bot.set_xlabel("Epoch")
    ax_bot.set_ylabel("Loss")
    ax_bot.grid(True, alpha=0.3)
    ax_bot.legend(
        loc="upper right", fontsize=10,
        frameon=True, framealpha=0.92, edgecolor="#CCCCCC",
    )

    fig.suptitle(
        "Decomposed training losses of ALTAS",
        fontweight="bold",
        fontsize=15,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    plt.savefig(save_path, dpi=FIGURE_DPI)
    plt.close(fig)

def plot_retention_vs_accuracy(
    history: Dict[str, list], save_path: str = "fig2_shared_retention.png"
) -> None:
    """双 y 轴：特征保留率 vs 训练准确率。"""
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.set_xlabel("Epochs", fontsize=12)
    ax1.set_ylabel("Feature Retention Rate", color="tab:purple", fontsize=12)
    ax1.plot(history["retention_rate"], color="tab:purple", linewidth=2, label="Retention Rate")
    ax1.tick_params(axis="y", labelcolor="tab:purple")
    ax1.set_ylim(0, 1.05)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Training Accuracy", color="tab:orange", fontsize=12)
    ax2.plot(history["mask_x_acc"], color="tab:orange", linewidth=2, linestyle="--", label="Mask-X Accuracy")
    ax2.tick_params(axis="y", labelcolor="tab:orange")
    ax2.set_ylim(0, 1.05)

    plt.title(
        "Feature retention rate vs. masked-input accuracy",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()
    plt.savefig(save_path, dpi=FIGURE_DPI)
    plt.close()


def apply_mask_with_shuffle(
    x: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """把 Gen 输出的硬掩码作用到输入, 被掩位置用 batch 内其他样本的同特征填充。

    公式::

        idx         = randperm(B)              # batch 内打乱的索引
        x_shuffled  = x[idx]                   # 边际分布与 x 完全一致
        x_mask      = x * mask + (1 - mask) * x_shuffled

    解决的痛点:
      - mask=0 的位置不会塌缩成"恒为 0", 下游 Extractor / Predictor
        能区分"这个 0 是被掩掉的"与"这个特征天然就接近 0"。
      - Critic 看到的分布偏移是"同分布但样本错位", 真的有信息损失,
        WGAN-GP 梯度方向更清晰。
      - 因为 ``x_shuffled`` 用整数索引 gather, 对 x 保持可微, 因此在
        Generator 分支对 mask 求梯度依旧传导: d(x_mask)/d(mask) = x - x_shuffled。
    """
    idx = torch.randperm(x.size(0), device=x.device)
    x_shuffled = x[idx]
    return x * mask + (1.0 - mask) * x_shuffled

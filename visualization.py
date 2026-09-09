"""训练过程的静态可视化。"""

from pathlib import Path

import matplotlib
import numpy as np

from config import FIGURE_DPI, PLOT_STYLE


# 训练脚本只保存图片；使用无界面后端可避免服务器环境依赖 Tcl/Tk。
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


plt.style.use(PLOT_STYLE)


def plot_training_losses(
    history: dict[str, list[float]],
    save_path: str | Path,
) -> None:
    """绘制三方总损失以及生成器各目标项的 epoch 曲线。"""
    figure, (total_axis, component_axis) = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True
    )
    epochs = np.arange(1, len(history["critic_loss"]) + 1)

    total_axis.plot(epochs, history["critic_loss"], label="Critic", color="#C44E52")
    total_axis.plot(
        epochs, history["predictor_loss"], label="Predictor", color="#4C72B0"
    )
    total_axis.plot(
        epochs,
        history["generator_loss"],
        label="Generator (total)",
        color="#000000",
        linestyle=":",
    )
    total_axis.set(title="Training losses", ylabel="Loss")
    total_axis.grid(True, alpha=0.3)
    total_axis.legend(framealpha=0.92)

    component_axis.plot(
        epochs,
        history["generator_adversarial_loss"],
        label="Adversarial",
        color="#8172B2",
    )
    component_axis.plot(
        epochs,
        history["generator_prediction_loss"],
        label="Prediction",
        color="#CCB974",
    )
    component_axis.plot(
        epochs,
        history["generator_sparsity_loss"],
        label="Sparsity",
        color="#55A467",
    )
    component_axis.set(
        title="Generator loss components",
        xlabel="Epoch",
        ylabel="Loss",
    )
    component_axis.grid(True, alpha=0.3)
    component_axis.legend(framealpha=0.92)

    figure.suptitle("ALTAS training history", fontsize=15, fontweight="bold")
    figure.tight_layout()
    figure.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)


def plot_retention_and_accuracy(
    history: dict[str, list[float]],
    save_path: str | Path,
) -> None:
    """以双纵轴绘制特征保留率和掩码输入分类准确率。"""
    figure, retention_axis = plt.subplots(figsize=(10, 5))
    epochs = np.arange(1, len(history["retention_rate"]) + 1)

    retention_axis.plot(
        epochs,
        history["retention_rate"],
        color="tab:purple",
        linewidth=2,
    )
    retention_axis.set(
        xlabel="Epoch",
        ylabel="Feature retention rate",
        ylim=(0, 1.05),
    )
    retention_axis.tick_params(axis="y", labelcolor="tab:purple")

    accuracy_axis = retention_axis.twinx()
    accuracy_axis.plot(
        epochs,
        history["masked_accuracy"],
        color="tab:orange",
        linewidth=2,
        linestyle="--",
    )
    accuracy_axis.set(ylabel="Masked-input accuracy", ylim=(0, 1.05))
    accuracy_axis.tick_params(axis="y", labelcolor="tab:orange")

    retention_axis.set_title(
        "Feature retention rate vs. masked-input accuracy",
        fontweight="bold",
    )
    figure.tight_layout()
    figure.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)

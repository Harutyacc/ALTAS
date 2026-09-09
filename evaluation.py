"""测试集评估与特征选择报告。"""

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import TOP_K_DISPLAY_MULTIPLIER
from masking import apply_shuffle_replacement_mask
from trainer import ALTASTrainer


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """测试集预测与特征选择指标。"""

    accuracy: float
    true_positive_rate: float
    false_discovery_rate: float
    average_selected_features: float
    input_dim: int
    predicted_masks: np.ndarray
    true_masks: np.ndarray


@torch.no_grad()
def evaluate_model(
    trainer: ALTASTrainer,
    test_loader: DataLoader,
) -> EvaluationReport:
    """在测试集上计算分类准确率和实例级特征选择指标。"""
    trainer.eval()
    correct_predictions = 0
    sample_count = 0
    predicted_mask_batches: list[np.ndarray] = []
    true_mask_batches: list[np.ndarray] = []

    for input_batch, label_batch, true_mask_batch in test_loader:
        input_batch = input_batch.to(trainer.device)
        label_batch = label_batch.to(trainer.device)

        _, predicted_mask = trainer.generator(
            input_batch,
            temperature=1e-3,
            hard=True,
        )
        masked_inputs = apply_shuffle_replacement_mask(input_batch, predicted_mask)
        logits = trainer.predictor(trainer.extractor(masked_inputs))

        correct_predictions += (logits.argmax(dim=1) == label_batch).sum().item()
        sample_count += label_batch.size(0)
        predicted_mask_batches.append(predicted_mask.cpu().numpy())
        true_mask_batches.append(true_mask_batch.numpy())

    if not predicted_mask_batches:
        raise ValueError("test_loader 为空，无法执行评估")

    predicted_masks = np.concatenate(predicted_mask_batches, axis=0)
    true_masks = np.concatenate(true_mask_batches, axis=0)
    true_positive_rate, false_discovery_rate = compute_selection_metrics(
        torch.from_numpy(predicted_masks),
        torch.from_numpy(true_masks),
    )
    average_selected_features = float(predicted_masks.sum(axis=1).mean())

    return EvaluationReport(
        accuracy=correct_predictions / sample_count,
        true_positive_rate=true_positive_rate,
        false_discovery_rate=false_discovery_rate,
        average_selected_features=average_selected_features,
        input_dim=predicted_masks.shape[1],
        predicted_masks=predicted_masks,
        true_masks=true_masks,
    )


def compute_selection_metrics(
    predicted_masks: torch.Tensor,
    true_masks: torch.Tensor,
) -> tuple[float, float]:
    """计算逐样本平均的真正率和错误发现率，返回百分比数值。"""
    if predicted_masks.shape != true_masks.shape:
        raise ValueError(
            "预测掩码与真实掩码形状不一致："
            f"{predicted_masks.shape} 与 {true_masks.shape}"
        )

    predicted = predicted_masks.float()
    expected = true_masks.float()
    true_positives = (predicted * expected).sum(dim=1)
    actual_positives = expected.sum(dim=1)
    predicted_positives = predicted.sum(dim=1)
    false_positives = predicted_positives - true_positives

    true_positive_rates = true_positives / actual_positives.clamp_min(1e-8)
    false_discovery_rates = torch.zeros_like(true_positive_rates)
    has_predictions = predicted_positives > 0
    false_discovery_rates[has_predictions] = (
        false_positives[has_predictions] / predicted_positives[has_predictions]
    )
    return (
        true_positive_rates.mean().item() * 100.0,
        false_discovery_rates.mean().item() * 100.0,
    )


def print_feature_report(report: EvaluationReport) -> list[int]:
    """打印整体评估和高频特征，并返回建议导出的特征索引。"""
    export_count = min(
        max(math.ceil(report.average_selected_features), 1),
        report.input_dim,
    )
    display_count = min(
        export_count * TOP_K_DISPLAY_MULTIPLIER,
        report.input_dim,
    )
    selection_counts = report.predicted_masks.sum(axis=0)
    selection_frequencies = report.predicted_masks.mean(axis=0)
    sorted_indices = np.argsort(selection_counts)[::-1]
    relevant_features = report.true_masks.any(axis=0)

    print("=" * 64)
    print("测试集评估与特征选择报告")
    print("=" * 64)
    print(f"分类准确率：        {report.accuracy * 100:.2f}%")
    print(f"真正率（TPR）：     {report.true_positive_rate:.2f}%")
    print(f"错误发现率（FDR）： {report.false_discovery_rate:.2f}%")
    print(
        "平均保留特征数：    "
        f"{report.average_selected_features:.1f} / {report.input_dim}"
    )
    print("-" * 64)
    print(f"选择频率最高的 {display_count} 个特征")
    print("-" * 64)

    for rank, feature_index in enumerate(sorted_indices[:display_count], start=1):
        relevance = "真实相关" if relevant_features[feature_index] else "噪声特征"
        print(
            f"Top {rank:02d}: feature_{feature_index:03d} | "
            f"选择 {int(selection_counts[feature_index]):4d} 次 "
            f"({selection_frequencies[feature_index] * 100:5.1f}%) | "
            f"{relevance}"
        )
    print("=" * 64)
    return sorted_indices[:export_count].astype(int).tolist()

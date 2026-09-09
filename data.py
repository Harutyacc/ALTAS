"""合成数据生成与数据加载。"""

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from config import TRUE_FEATURE_DIM


def generate_synthetic_data(
    num_samples: int = 2_000,
    input_dim: int = 100,
    dataset_type: str = "Syn4",
    save_csv: bool = False,
    csv_path: str | Path | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """生成 INVASE 论文定义的 Syn1 至 Syn6 合成数据。

    Returns:
        ``(features, labels, true_masks)``，形状依次为 ``[N, D]``、
        ``[N]`` 和 ``[N, D]``。
    """
    if input_dim < TRUE_FEATURE_DIM:
        raise ValueError(
            f"Syn 数据集至少需要 {TRUE_FEATURE_DIM} 个特征维度"
            f"（索引 0 至 {TRUE_FEATURE_DIM - 1}）"
        )
    if dataset_type not in {f"Syn{index}" for index in range(1, 7)}:
        raise ValueError(f"不支持的数据集类型 {dataset_type!r}，可选 Syn1 至 Syn6")

    features = torch.randn(num_samples, input_dim)
    true_masks = torch.zeros(num_samples, input_dim)

    probability_1 = torch.sigmoid(features[:, 0] * features[:, 1])
    probability_2 = torch.sigmoid(
        features[:, 2].square()
        + features[:, 3].square()
        + features[:, 4].square()
        + features[:, 5].square()
        - 4.0
    )
    probability_3 = torch.sigmoid(
        -10 * torch.sin(0.2 * features[:, 6])
        + torch.abs(features[:, 7])
        + features[:, 8]
        + torch.exp(-features[:, 9])
        - 2.4
    )

    probabilities = _apply_synthetic_rule(
        dataset_type,
        features,
        true_masks,
        (probability_1, probability_2, probability_3),
    )
    labels = torch.bernoulli(probabilities).long()

    if save_csv:
        destination = Path(csv_path or f"syn_{dataset_type}_n{num_samples}_d{input_dim}.csv")
        _save_dataset_csv(features, labels, true_masks, destination)
        print(f"[数据] {dataset_type} 数据集已保存至 {destination}")

    return features, labels, true_masks


def _apply_synthetic_rule(
    dataset_type: str,
    features: torch.Tensor,
    true_masks: torch.Tensor,
    probabilities: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """应用指定的标签生成规则，并原地填写真实特征掩码。"""
    probability_1, probability_2, probability_3 = probabilities
    static_rules = {
        "Syn1": (probability_1, slice(0, 2)),
        "Syn2": (probability_2, slice(2, 6)),
        "Syn3": (probability_3, slice(6, 10)),
    }
    if dataset_type in static_rules:
        selected_probabilities, feature_slice = static_rules[dataset_type]
        true_masks[:, feature_slice] = 1.0
        return selected_probabilities

    switch_rules = {
        "Syn4": (probability_1, probability_2, slice(0, 2), slice(2, 6)),
        "Syn5": (probability_1, probability_3, slice(0, 2), slice(6, 10)),
        "Syn6": (probability_2, probability_3, slice(2, 6), slice(6, 10)),
    }
    negative_probs, positive_probs, negative_slice, positive_slice = switch_rules[dataset_type]
    negative_switch = features[:, 10] < 0
    positive_switch = ~negative_switch

    selected_probabilities = torch.empty(features.size(0))
    selected_probabilities[negative_switch] = negative_probs[negative_switch]
    selected_probabilities[positive_switch] = positive_probs[positive_switch]
    true_masks[negative_switch, negative_slice] = 1.0
    true_masks[positive_switch, positive_slice] = 1.0
    true_masks[:, 10] = 1.0
    return selected_probabilities


def create_data_loaders(
    features: torch.Tensor,
    labels: torch.Tensor,
    true_masks: torch.Tensor,
    batch_size: int = 64,
    train_ratio: float = 0.8,
) -> tuple[DataLoader, DataLoader]:
    """按样本顺序切分训练集和测试集，并创建数据加载器。"""
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio 必须严格位于 0 和 1 之间")
    if not (len(features) == len(labels) == len(true_masks)):
        raise ValueError("features、labels 和 true_masks 的样本数必须相同")

    train_size = int(train_ratio * len(features))
    train_dataset = TensorDataset(
        features[:train_size], labels[:train_size], true_masks[:train_size]
    )
    test_dataset = TensorDataset(
        features[train_size:], labels[train_size:], true_masks[train_size:]
    )
    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
    )


def _save_dataset_csv(
    features: torch.Tensor,
    labels: torch.Tensor,
    true_masks: torch.Tensor,
    destination: Path,
) -> None:
    """将特征、标签和真实掩码保存到一个 CSV 文件。"""
    feature_frame = pd.DataFrame(
        features.numpy(), columns=[f"feature_{index}" for index in range(features.size(1))]
    )
    mask_frame = pd.DataFrame(
        true_masks.numpy(),
        columns=[f"true_mask_{index}" for index in range(true_masks.size(1))],
    )
    feature_frame["target"] = labels.numpy()
    pd.concat([feature_frame, mask_frame], axis=1).to_csv(destination, index=False)

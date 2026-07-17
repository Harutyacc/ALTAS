"""测试集评估与 Top-K 特征重要性报告。"""

import math
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import TOP_K_PRINT_MULTIPLIER, TRUE_FEATURE_DIM
from train import ALTASTrainer
from utils import evaluate_feature_selection


@dataclass
class TestReport:
    test_accuracy: float
    tpr: float
    fdr: float
    avg_features: float
    input_dim: int
    masks: np.ndarray
    true_masks: np.ndarray
    x_masks: np.ndarray
    h_masks: np.ndarray
    labels: np.ndarray


@torch.no_grad()
def collect_test_outputs(
    trainer: ALTASTrainer, test_loader: DataLoader, device: torch.device
) -> TestReport:
    """在测试集上推理，收集掩码、隐特征与标签，聚合出完整 TestReport。"""
    trainer.set_eval()
    test_correct = 0
    total = 0
    masks, true_masks, x_masks, h_masks, labels = [], [], [], [], []

    for X_batch, Y_batch, S_batch in test_loader:
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
        _, mask = trainer.gen(X_batch, tau=1e-3, hard=True)
        x_mask = X_batch * mask
        h_mask = trainer.ext(x_mask)
        logits = trainer.predictor(h_mask)
        preds = logits.argmax(dim=1)

        masks.append(mask.cpu().numpy())
        true_masks.append(S_batch.numpy())
        x_masks.append(x_mask.cpu().numpy())
        h_masks.append(h_mask.cpu().numpy())
        labels.append(Y_batch.cpu().numpy())
        test_correct += (preds == Y_batch).sum().item()
        total += Y_batch.size(0)

    masks_np = _concat(masks)
    true_np = _concat(true_masks)
    x_np = _concat(x_masks)
    h_np = _concat(h_masks)
    lbl_np = _concat(labels)

    avg_features = float(np.mean(np.sum(masks_np, axis=1)))
    tpr, fdr = evaluate_feature_selection(torch.tensor(masks_np), torch.tensor(true_np))

    return TestReport(
        test_accuracy=test_correct / max(1, total),
        tpr=tpr,
        fdr=fdr,
        avg_features=avg_features,
        input_dim=masks_np.shape[1],
        masks=masks_np,
        true_masks=true_np,
        x_masks=x_np,
        h_masks=h_np,
        labels=lbl_np,
    )


def print_top_k_report(report: TestReport) -> List[int]:
    """打印 Top-K 特征并返回保存到文件的索引列表。

    - K      = ceil(avg_features) (导出文件用的索引数)
    - K_print = K * TOP_K_PRINT_MULTIPLIER (供人查看的扩展量)
    两者都夹紧到 [1, input_dim], 防止 K_print 越界或打印空报告。
    """
    K = min(max(math.ceil(report.avg_features), 1), report.input_dim)
    K_print = min(K * TOP_K_PRINT_MULTIPLIER, report.input_dim)

    print("=" * 50)
    print("🎯 测试集推理报告 (Shared Architecture)")
    print("=" * 50)
    print(f"• 测试集准确率:     {report.test_accuracy * 100:.2f}%")
    print(f"• TPR (召回率):     {report.tpr:.2f}%")
    print(f"• FDR (误发现率):   {report.fdr:.2f}%")
    print(f"• 平均保留特征数量: {report.avg_features:.1f} / {report.input_dim}")
    print(f"• 整体保留率:       {(report.avg_features / report.input_dim) * 100:.2f}%")
    print("-" * 50)
    print(f"🧩 Top {K_print} 最常被选中的特征")
    print("-" * 50)

    counts = np.sum(report.masks, axis=0)
    freq = np.mean(report.masks, axis=0)
    sorted_idx = np.argsort(counts)[::-1]

    for i in range(K_print):
        idx = int(sorted_idx[i])
        tag = "★ [真实关联]" if idx < TRUE_FEATURE_DIM else "  [纯噪声]"
        print(
            f"Top {i+1:02d}: 特征 Index {idx:03d} | "
            f"被选次数: {int(counts[idx]):4d} 次 ({freq[idx] * 100:5.1f}%) | {tag}"
        )
    print("=" * 50)
    return sorted_idx[:K].tolist()


def _concat(parts):
    return np.concatenate(parts, axis=0)

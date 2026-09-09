"""ALTAS 实验入口。

本模块负责装配配置、数据、训练器、评估与结果导出。运行方式：

``python train.py``
"""

import json
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import TextIO

import numpy as np
import torch

from config import DataConfig, ModelConfig, TrainingConfig
from data import generate_synthetic_data, create_data_loaders
from evaluation import evaluate_model, print_feature_report
from trainer import ALTASTrainer
from training import run_training
from visualization import plot_retention_and_accuracy, plot_training_losses


class TeeStream:
    """将写入内容同步转发至多个文本流。"""

    def __init__(self, *streams: TextIO) -> None:
        self._streams = streams

    def write(self, text: str) -> int:
        """向所有目标流写入相同文本。"""
        for stream in self._streams:
            stream.write(text)
        return len(text)

    def flush(self) -> None:
        """刷新所有目标流。"""
        for stream in self._streams:
            stream.flush()

    def __getattr__(self, name: str):
        """将未实现的文本流属性委托给首个目标流。"""
        return getattr(self._streams[0], name)


@contextmanager
def capture_terminal_output(log_path: Path) -> Iterator[None]:
    """在保留终端输出的同时，将标准输出和错误写入日志文件。"""
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
        sys.stdout = TeeStream(original_stdout, log_file)  # type: ignore[assignment]
        sys.stderr = TeeStream(original_stderr, log_file)  # type: ignore[assignment]
        try:
            yield
        except Exception:
            print(
                "[日志] 运行异常，完整堆栈已写入 terminal.log",
                file=original_stderr,
            )
            raise
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


def create_run_directory(root: str | Path = "output") -> Path:
    """创建带时间戳的独立实验输出目录。"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_directory = Path(root) / f"run_{timestamp}"
    run_directory.mkdir(parents=True, exist_ok=False)
    return run_directory


def save_configuration(
    run_directory: Path,
    data_config: DataConfig,
    model_config: ModelConfig,
    training_config: TrainingConfig,
) -> None:
    """将本次实验的完整配置保存为易读的 JSON 文件。"""
    snapshot = {
        "data": asdict(data_config),
        "model": asdict(model_config),
        "training": asdict(training_config),
    }
    (run_directory / "config.json").write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    """执行一次完整的 ALTAS 训练、评估和结果导出流程。"""
    data_config = DataConfig()
    model_config = ModelConfig(input_dim=data_config.input_dim)
    training_config = TrainingConfig()
    run_directory = create_run_directory()
    save_configuration(
        run_directory,
        data_config,
        model_config,
        training_config,
    )

    with capture_terminal_output(run_directory / "terminal.log"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[系统] 输出目录：{run_directory}")
        print(f"[系统] 计算设备：{device}")

        features, labels, true_masks = generate_synthetic_data(
            num_samples=data_config.num_samples,
            input_dim=data_config.input_dim,
            dataset_type=data_config.dataset_type,
        )
        train_loader, test_loader = create_data_loaders(
            features,
            labels,
            true_masks,
            batch_size=data_config.batch_size,
            train_ratio=data_config.train_ratio,
        )

        trainer = ALTASTrainer(
            input_dim=model_config.input_dim,
            num_classes=model_config.num_classes,
            hidden_dim=model_config.hidden_dim,
            device=device,
            generator_learning_rate=training_config.generator_learning_rate,
            task_learning_rate=training_config.task_learning_rate,
            adversarial_weight=training_config.adversarial_weight,
            prediction_weight=training_config.prediction_weight,
            sparsity_weight=training_config.sparsity_weight,
            critic_steps=training_config.critic_steps,
            gradient_penalty_weight=training_config.gradient_penalty_weight,
            masked_prediction_weight=training_config.masked_prediction_weight,
        )

        print("\n[训练] 开始训练 ALTAS 模型")
        history = run_training(
            trainer,
            train_loader,
            epochs=training_config.epochs,
            temperature_start=training_config.temperature_start,
            temperature_min=training_config.temperature_min,
            temperature_decay=training_config.temperature_decay,
            log_interval=training_config.log_interval,
        )
        _save_training_outputs(run_directory, history)

        print("\n[评估] 开始测试集推理")
        report = evaluate_model(trainer, test_loader)
        selected_feature_indices = print_feature_report(report)
        _save_evaluation_outputs(
            run_directory,
            trainer,
            report.predicted_masks,
            report.true_masks,
            selected_feature_indices,
        )


def _save_training_outputs(
    run_directory: Path,
    history: dict[str, list[float]],
) -> None:
    """保存训练曲线和逐 epoch 指标。"""
    plot_training_losses(history, run_directory / "training_losses.png")
    plot_retention_and_accuracy(
        history,
        run_directory / "retention_and_accuracy.png",
    )
    np.savez(
        run_directory / "training_history.npz",
        **{name: np.asarray(values) for name, values in history.items()},
    )
    print(f"[导出] 训练结果已保存至 {run_directory}")


def _save_evaluation_outputs(
    run_directory: Path,
    trainer: ALTASTrainer,
    predicted_masks: np.ndarray,
    true_masks: np.ndarray,
    selected_feature_indices: list[int],
) -> None:
    """保存测试掩码、推荐特征索引和生成器权重。"""
    np.savez(
        run_directory / "evaluation_masks.npz",
        predicted_masks=predicted_masks,
        true_masks=true_masks,
    )
    np.save(
        run_directory / "selected_feature_indices.npy",
        np.asarray(selected_feature_indices, dtype=np.int64),
    )
    torch.save(
        trainer.generator.state_dict(),
        run_directory / "feature_mask_generator.pt",
    )
    print(f"[导出] 评估结果和模型权重已保存至 {run_directory}")


if __name__ == "__main__":
    main()

"""跨 epoch 的训练循环与指标聚合。"""

from collections import defaultdict

from torch.utils.data import DataLoader

from trainer import ALTASTrainer


def initialize_history() -> dict[str, list[float]]:
    """为训练器定义的每项指标创建空历史序列。"""
    return {name: [] for name in ALTASTrainer.METRIC_NAMES}


def run_training(
    trainer: ALTASTrainer,
    train_loader: DataLoader,
    epochs: int,
    temperature_start: float = 1.0,
    temperature_min: float = 0.1,
    temperature_decay: float = 0.995,
    log_interval: int = 10,
) -> dict[str, list[float]]:
    """执行完整训练循环并返回按 epoch 聚合的指标历史。"""
    if epochs < 1:
        raise ValueError("epochs 必须大于或等于 1")
    if log_interval < 1:
        raise ValueError("log_interval 必须大于或等于 1")

    history = initialize_history()
    epoch_sums: defaultdict[str, float] = defaultdict(float)
    temperature = temperature_start
    num_batches = max(1, len(train_loader))

    trainer.train()
    for epoch_index in range(epochs):
        epoch_sums.clear()
        for input_batch, label_batch, _ in train_loader:
            metrics = trainer.train_step(
                input_batch,
                label_batch,
                temperature=temperature,
            )
            for name, value in metrics.items():
                epoch_sums[name] += value

        for name in history:
            history[name].append(epoch_sums[name] / num_batches)

        temperature = max(temperature_min, temperature * temperature_decay)
        if (epoch_index + 1) % log_interval == 0:
            _print_epoch_metrics(
                epoch=epoch_index + 1,
                epochs=epochs,
                temperature=temperature,
                metrics={name: values[-1] for name, values in history.items()},
            )

    return history


def _print_epoch_metrics(
    epoch: int,
    epochs: int,
    temperature: float,
    metrics: dict[str, float],
) -> None:
    """以紧凑的单行格式打印一个 epoch 的聚合指标。"""
    print(
        f"Epoch [{epoch:04d}/{epochs}] | Temp: {temperature:.3f} | "
        f"Critic: {metrics['critic_loss']:+.3f} | "
        f"Predictor: {metrics['predictor_loss']:+.3f} | "
        f"Full acc: {metrics['full_accuracy'] * 100:.1f}% | "
        f"Generator: {metrics['generator_loss']:+.3f} | "
        f"Retain: {metrics['retention_rate'] * 100:.1f}% | "
        f"Masked acc: {metrics['masked_accuracy'] * 100:.1f}% | "
        f"W-distance: {metrics['wasserstein_distance']:+.3f}"
    )

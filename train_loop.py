"""训练循环与历史记录的工具函数。"""

from collections import defaultdict
from typing import Dict

from torch.utils.data import DataLoader

from train import ALTASTrainer


HISTORY_KEYS = ALTASTrainer.HISTORY_KEYS


def init_history() -> Dict[str, list]:
    return {k: [] for k in HISTORY_KEYS}


def run_training(
    trainer: ALTASTrainer,
    train_loader: DataLoader,
    epochs: int,
    tau_start: float = 1.0,
    tau_min: float = 0.1,
    tau_decay: float = 0.995,
    log_every: int = 10,
) -> Dict[str, list]:
    """执行完整训练循环并返回聚合后的 history。"""
    history = init_history()
    epoch_sums = defaultdict(float)
    tau = tau_start
    num_batches = max(1, len(train_loader))

    trainer.set_train()
    for epoch in range(epochs):
        for k in epoch_sums:
            epoch_sums[k] = 0.0
        for X_batch, Y_batch, _ in train_loader:
            X_batch, Y_batch = X_batch.to(trainer.device), Y_batch.to(trainer.device)
            metrics = trainer.train_step(X_batch, Y_batch, tau=tau)
            for k, v in metrics.items():
                if k in history:
                    epoch_sums[k] += v

        for k in history:
            history[k].append(epoch_sums[k] / num_batches)

        tau = max(tau_min, tau * tau_decay)

        if (epoch + 1) % log_every == 0:
            last = {k: history[k][-1] for k in history}
            print(
                f"Epoch [{epoch+1:03d}/{epochs}] | Tau: {tau:.3f} | "
                f"Loss_C: {last['loss_c']:+.3f} | "
                f"Loss_P: {last['loss_p']:+.3f} | "
                f"Full_X_Acc: {last['full_x_acc']*100:.1f}% | "
                f"Loss_Gen: {last['loss_gen']:+.3f} | "
                f"Retain: {last['retention_rate']*100:.1f}% | "
                f"Mask_X_Acc: {last['mask_x_acc']*100:.1f}% | "
                f"Loss_Gen_adv: {last['loss_gen_adv']:+.3f} | "
                f"Loss_Gen_l1: {last['loss_gen_l1']:+.3f} | "
                f"Loss_Gen_p: {last['loss_gen_p']:+.3f} | "
                f"W_Distance: {last['w_distance']:+.3f} |"
            )

    return history

"""实验入口：装配数据 / 模型 / 训练 / 评估，把终端输出与 config 一起落盘。"""

import json
import sys
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from config import DataConfig, TrainConfig
from evaluate import collect_test_outputs, print_top_k_report
from train import ALTASTrainer
from train_loop import run_training
from utils import (
    generate_synthetic_data,
    get_dataloaders,
    plot_comparative_tsne,
    plot_retention_vs_accuracy,
    plot_training_loss,
)


@contextmanager
def tee_output(log_path: Path):
    """把 stdout/stderr 同时写到终端和 log_path。"""
    log_f = open(log_path, "w", encoding="utf-8", buffering=1)

    class _Tee:
        def __init__(self, *streams):
            self._streams = streams

        def write(self, s):
            for stream in self._streams:
                stream.write(s)
            return len(s)

        def flush(self):
            for stream in self._streams:
                stream.flush()

        def __getattr__(self, name):
            return getattr(self._streams[0], name)

    saved_out, saved_err = sys.stdout, sys.stderr
    sys.stdout = _Tee(saved_out, log_f)
    sys.stderr = _Tee(saved_err, log_f)
    try:
        yield log_f
    except Exception:
        print("[tee_output] 异常退出, 栈信息已被保存至 terminal.log", file=saved_err)
        raise
    finally:
        sys.stdout, sys.stderr = saved_out, saved_err
        log_f.close()


def make_run_dir(root: str = "output") -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(root) / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def main():
    data_cfg = DataConfig()
    train_cfg = TrainConfig()
    run_dir = make_run_dir()

    cfg_snapshot = {"data": asdict(data_cfg), "train": asdict(train_cfg)}
    (run_dir / "config.json").write_text(
        json.dumps(cfg_snapshot, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[Export] config 已先写至 '{run_dir / 'config.json'}'")

    with tee_output(run_dir / "terminal.log"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[System] Run dir: {run_dir}")
        print(f"[System] 正在使用 {device} 进行计算...")

        X, Y, S = generate_synthetic_data(
            num_samples=data_cfg.num_samples,
            input_dim=data_cfg.input_dim,
            dataset_type=data_cfg.dataset_type,
            save_csv=False,
        )
        train_loader, test_loader = get_dataloaders(
            X, Y, S, batch_size=data_cfg.batch_size, train_ratio=data_cfg.train_ratio
        )

        trainer = ALTASTrainer(
            input_dim=data_cfg.input_dim,
            num_classes=2,
            device=device,
            lr_gen=train_cfg.lr_gen,
            lr_ep=train_cfg.lr_ep,
            alpha=train_cfg.alpha,
            beta=train_cfg.beta,
            gamma_init=train_cfg.gamma_init,
            lr_gamma=train_cfg.lr_gamma,
            target_w_dist=train_cfg.target_w_dist,
            n_critic=train_cfg.n_critic,
            lambda_gp=train_cfg.lambda_gp,
            p_mask_weight=train_cfg.p_mask_weight,
        )

        print("\n[System] 开始训练多任务共享特征架构...")
        history = run_training(
            trainer,
            train_loader,
            epochs=train_cfg.epochs,
            tau_start=train_cfg.tau_start,
            tau_min=train_cfg.tau_min,
            tau_decay=train_cfg.tau_decay,
            log_every=train_cfg.log_every,
        )

        plot_training_loss(history, save_path=str(run_dir / "fig1_shared_loss.png"))
        plot_retention_vs_accuracy(history, save_path=str(run_dir / "fig2_shared_retention.png"))
        np.savez(run_dir / "history.npz", **history)
        print(f"[Export] 训练历史已保存至 '{run_dir / 'history.npz'}'")

        print("\n[System] 开始在测试集上进行推理评估...")
        report = collect_test_outputs(trainer, test_loader, device)
        top_k_indices = print_top_k_report(report)
        np.savez(
            run_dir / "test_arrays.npz",
            x_masks=report.x_masks,
            h_masks=report.h_masks,
            labels=report.labels,
            masks=report.masks,
            true_masks=report.true_masks,
        )
        print(f"[Export] 测试集数组已保存至 '{run_dir / 'test_arrays.npz'}'")

        np.save(run_dir / "selected_feature_indices.npy", np.asarray(top_k_indices, dtype=np.int64))
        print(f"[Export] 特征索引已保存至 '{run_dir / 'selected_feature_indices.npy'}'")
        torch.save(trainer.gen.state_dict(), run_dir / "generator_model.pth")
        print(f"[Export] 模型权重已保存至 '{run_dir / 'generator_model.pth'}'")

        plot_comparative_tsne(
            raw_features=report.x_masks,
            latent_features=report.h_masks,
            labels=report.labels,
            save_path=str(run_dir / "fig3_tsne_comparison.png"),
        )


if __name__ == "__main__":
    main()

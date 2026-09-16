"""Syn4 A/B test: random-mask predictor warmup followed by a sparsity ramp.

Run from the repository root: python -m diagnostics.pair_warmup
True masks are used for evaluation only, never for training.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from config import DataConfig, ModelConfig, TrainingConfig
from data import generate_synthetic_data
from diagnostics.counterfactual_masks import evaluate_split, fixed_eval_loader
from trainer import ALTASTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-samples", type=int, default=DataConfig().num_samples)
    parser.add_argument("--batch-size", type=int, default=DataConfig().batch_size)
    parser.add_argument("--input-dim", type=int, default=DataConfig().input_dim)
    parser.add_argument("--hidden-dim", type=int, default=ModelConfig().hidden_dim)
    parser.add_argument("--train-ratio", type=float, default=DataConfig().train_ratio)
    parser.add_argument("--warmup-epochs", type=int, default=100)
    parser.add_argument("--joint-epochs", type=int, default=200)
    parser.add_argument("--ramp-epochs", type=int, default=100)
    parser.add_argument("--warmup-keep", type=float, default=0.9)
    parser.add_argument("--checkpoints", type=str, default="0,50,100,200")
    parser.add_argument("--eval-samples", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--output", type=Path, default=Path("diagnostics/results/syn4_pair_warmup.json")
    )
    args = parser.parse_args()
    if args.num_samples < 2 or args.input_dim < 11 or args.batch_size < 1:
        parser.error("num-samples >= 2, input-dim >= 11, batch-size >= 1 required")
    if not 0 < args.train_ratio < 1 or not 0 < int(args.num_samples * args.train_ratio) < args.num_samples:
        parser.error("train-ratio must leave samples in both splits")
    if args.hidden_dim < 1 or args.warmup_epochs < 0 or args.joint_epochs < 1:
        parser.error("hidden-dim >= 1, warmup-epochs >= 0, joint-epochs >= 1 required")
    if args.ramp_epochs < 1 or args.eval_samples < 1 or not 0 < args.warmup_keep <= 1:
        parser.error("ramp-epochs and eval-samples >= 1; 0 < warmup-keep <= 1 required")
    try:
        checkpoints = {int(value.strip()) for value in args.checkpoints.split(",")}
    except ValueError:
        parser.error("checkpoints must be comma-separated integers")
    if any(epoch < 0 for epoch in checkpoints):
        parser.error("checkpoints must be nonnegative")
    args.checkpoints = sorted({epoch for epoch in checkpoints if epoch <= args.joint_epochs} | {args.joint_epochs})
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    return args


def make_trainer(args: argparse.Namespace, config: TrainingConfig) -> ALTASTrainer:
    return ALTASTrainer(
        input_dim=args.input_dim, num_classes=2, hidden_dim=args.hidden_dim,
        device=torch.device(args.device),
        generator_learning_rate=config.generator_learning_rate,
        task_learning_rate=config.task_learning_rate,
        adversarial_weight=config.adversarial_weight,
        prediction_weight=config.prediction_weight,
        sparsity_weight=config.sparsity_weight,
        critic_steps=config.critic_steps,
        gradient_penalty_weight=config.gradient_penalty_weight,
        masked_prediction_weight=config.masked_prediction_weight,
    )


def pair_probe(trainer: ALTASTrainer, loader: DataLoader, seed: int) -> dict[str, float]:
    """On negative-switch cases, compare the pair to switch-only, same replacement."""
    sums = {"pair_ce": 0.0, "switch_only_ce": 0.0, "pair_correct": 0, "count": 0}
    original_modes = [module.training for module in trainer.modules()]
    cuda_devices = [torch.cuda.current_device()] if trainer.device.type == "cuda" else []
    try:
        trainer.eval()
        with torch.no_grad(), torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(seed + 20_000)
            for batch_x, batch_y, _ in loader:
                x, y = batch_x.to(trainer.device), batch_y.to(trainer.device)
                members = x[:, 10] < 0
                if not members.any():
                    continue
                x, y = x[members], y[members]
                # One identical shuffle for both masks.
                shuffled = x[torch.randperm(len(x), device=trainer.device)]
                pair = torch.zeros_like(x)
                pair[:, [0, 1, 10]] = 1
                single = torch.zeros_like(x)
                single[:, 10] = 1
                for name, mask in (("pair", pair), ("switch_only", single)):
                    logits = trainer.predictor(trainer.extractor(x * mask + shuffled * (1 - mask)))
                    sums[f"{name}_ce"] += F.cross_entropy(logits, y, reduction="sum").item()
                    if name == "pair":
                        sums["pair_correct"] += int(logits.argmax(1).eq(y).sum().item())
                sums["count"] += len(x)
    finally:
        for module, mode in zip(trainer.modules(), original_modes):
            module.train(mode)
    n = sums["count"]
    return {
        "negative_samples": n,
        "pair_ce": round(sums["pair_ce"] / n, 6),
        "switch_only_ce": round(sums["switch_only_ce"] / n, 6),
        "pair_ce_gain": round((sums["switch_only_ce"] - sums["pair_ce"]) / n, 6),
        "pair_accuracy": round(sums["pair_correct"] / n, 6),
    }


def pair_probability(trainer: ALTASTrainer, loader: DataLoader) -> dict[str, float]:
    totals = {"negative": [0.0, 0.0, 0.0, 0], "nonnegative": [0.0, 0.0, 0.0, 0]}
    with torch.no_grad():
        for batch_x, _, _ in loader:
            x = batch_x.to(trainer.device)
            p = trainer.generator.network(x).reshape(-1, x.size(1), 2).softmax(-1)[..., 1]
            for name, members in (("negative", x[:, 10] < 0), ("nonnegative", x[:, 10] >= 0)):
                totals[name][0] += p[members, 0].sum().item()
                totals[name][1] += p[members, 1].sum().item()
                totals[name][2] += (p[members, 0] * p[members, 1]).sum().item()
                totals[name][3] += int(members.sum().item())
    return {
        name: {"p0": round(v[0] / v[3], 6), "p1": round(v[1] / v[3], 6),
               "pair_independent_sampling_probability": round(v[2] / v[3], 8)}
        for name, v in totals.items()
    }


def main() -> None:
    args = parse_args()
    config = TrainingConfig()
    torch.manual_seed(args.seed)
    features, labels, true_masks = generate_synthetic_data(
        num_samples=args.num_samples, input_dim=args.input_dim, dataset_type="Syn4"
    )
    split = int(args.num_samples * args.train_ratio)
    train_data = TensorDataset(features[:split], labels[:split])
    test_loader = fixed_eval_loader(
        features[split:], labels[split:], true_masks[split:], args.batch_size, args.eval_samples
    )
    union = true_masks[:split].any(0, keepdim=True).float().to(args.device)
    report = {"settings": {**vars(args), "output": str(args.output), "training": asdict(config)}, "arms": {}}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Syn4 train={split} test={len(features)-split} device={args.device}", flush=True)
    for arm in ("baseline", "warmup_ramp"):
        # Both arms start with exactly the same network weights and optimizer state.
        torch.manual_seed(args.seed + 1)
        trainer = make_trainer(args, config)
        trainer.train()
        arm_report = {"warmup": {}, "snapshots": {}}
        report["arms"][arm] = arm_report
        if arm == "warmup_ramp":
            warmup_loader = DataLoader(
                train_data, batch_size=args.batch_size, shuffle=True,
                generator=torch.Generator().manual_seed(args.seed + 2),
            )
            pair_seen = 0
            sample_seen = 0
            for epoch in range(1, args.warmup_epochs + 1):
                loss_total = 0.0
                for x, y in warmup_loader:
                    x, y = x.to(trainer.device), y.to(trainer.device)
                    mask = (torch.rand_like(x) < args.warmup_keep).float()
                    pair_seen += int((mask[:, 0] * mask[:, 1]).sum().item())
                    sample_seen += len(x)
                    loss_total += trainer._update_predictor(x, y, mask)
                if epoch == 1 or epoch % 20 == 0 or epoch == args.warmup_epochs:
                    print(f"{arm} warmup {epoch}: predictor loss={loss_total/len(warmup_loader):.4f}", flush=True)
            arm_report["warmup"] = {
                "epochs": args.warmup_epochs,
                "pair_seen_rate": round(pair_seen / sample_seen, 6) if sample_seen else None,
                "pair_probe": pair_probe(trainer, test_loader, args.seed),
            }
            print(f"{arm} warmup probe: {arm_report['warmup']}", flush=True)

        joint_loader = DataLoader(
            train_data, batch_size=args.batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(args.seed + 3),
        )
        # Identical batch order and stochastic stream at the start of joint training.
        torch.manual_seed(args.seed + 4)
        temperature = config.temperature_start

        def snapshot(epoch: int) -> None:
            scores = evaluate_split(trainer, test_loader, union, repeats=1, seed=args.seed)
            entry = {"temperature": temperature, "sparsity_weight": trainer.sparsity_weight,
                     "pair_probability": pair_probability(trainer, test_loader), "test": scores,
                     "pair_probe": pair_probe(trainer, test_loader, args.seed)}
            arm_report["snapshots"][str(epoch)] = entry
            negative = scores["conditions"]["learned"]["switch_negative"]
            positive = scores["conditions"]["learned"]["switch_nonnegative"]
            print(
                f"{arm} epoch {epoch}: sparse={trainer.sparsity_weight:.3f}, "
                f"neg p01={entry['pair_probability']['negative']['pair_independent_sampling_probability']:.4f}, "
                f"neg exact={negative['exact_mask_accuracy']:.3f}, "
                f"pos exact={positive['exact_mask_accuracy']:.3f}, "
                f"neg pair CE gain={entry['pair_probe']['pair_ce_gain']:+.4f}",
                flush=True,
            )
            save()

        if 0 in args.checkpoints:
            snapshot(0)
        for epoch in range(1, args.joint_epochs + 1):
            if arm == "warmup_ramp":
                trainer.sparsity_weight = config.sparsity_weight * min(1.0, (epoch - 1) / args.ramp_epochs)
            loss_total = 0.0
            for x, y in joint_loader:
                loss_total += trainer.train_step(x, y, temperature)["generator_loss"]
            temperature = max(config.temperature_min, temperature * config.temperature_decay)
            if epoch in args.checkpoints:
                print(f"{arm} epoch {epoch}: train generator loss={loss_total/len(joint_loader):.4f}", flush=True)
                snapshot(epoch)
        save()
    print(f"Saved: {args.output}", flush=True)


if __name__ == "__main__":
    main()

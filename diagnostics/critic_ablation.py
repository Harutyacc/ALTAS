"""Paired Syn4 ablation of the generator's Critic/adversarial gradient.

Critic updates are retained in both arms to preserve the update sequence and
random-number consumption. Only the generator adversarial coefficient differs.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from config import DataConfig, ModelConfig, TrainingConfig
from data import generate_synthetic_data
from diagnostics.counterfactual_masks import evaluate_split, fixed_eval_loader
from diagnostics.pair_warmup import pair_probability
from trainer import ALTASTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=str, default="42,43")
    parser.add_argument("--num-samples", type=int, default=DataConfig().num_samples)
    parser.add_argument("--input-dim", type=int, default=DataConfig().input_dim)
    parser.add_argument("--train-ratio", type=float, default=DataConfig().train_ratio)
    parser.add_argument("--batch-size", type=int, default=DataConfig().batch_size)
    parser.add_argument("--hidden-dim", type=int, default=ModelConfig().hidden_dim)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--checkpoints", type=str, default="0,50,100,200")
    parser.add_argument("--eval-samples", type=int, default=4_000)
    parser.add_argument("--eval-repeats", type=int, default=3)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--output", type=Path, default=Path("diagnostics/results/syn4_critic_ablation.json")
    )
    args = parser.parse_args()
    try:
        args.seeds = [int(value.strip()) for value in args.seeds.split(",")]
        checkpoints = {int(value.strip()) for value in args.checkpoints.split(",")}
    except ValueError:
        parser.error("seeds and checkpoints must be comma-separated integers")
    if not args.seeds or any(seed < 0 for seed in args.seeds) or len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be distinct nonnegative integers")
    if args.num_samples < 2 or args.input_dim < 11 or args.batch_size < 1 or args.hidden_dim < 1:
        parser.error("num-samples >= 2, input-dim >= 11, batch-size and hidden-dim >= 1 required")
    split = int(args.num_samples * args.train_ratio)
    if not 0 < args.train_ratio < 1 or not 0 < split < args.num_samples:
        parser.error("train-ratio must leave samples in both splits")
    if args.epochs < 1 or args.eval_samples < 1 or args.eval_repeats < 1:
        parser.error("epochs, eval-samples and eval-repeats must be positive")
    if any(epoch < 0 for epoch in checkpoints):
        parser.error("checkpoints must be nonnegative")
    args.checkpoints = sorted({epoch for epoch in checkpoints if epoch <= args.epochs} | {args.epochs})
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    return args


def make_trainer(
    args: argparse.Namespace, config: TrainingConfig, adversarial_weight: float
) -> ALTASTrainer:
    return ALTASTrainer(
        input_dim=args.input_dim, num_classes=2, hidden_dim=args.hidden_dim,
        device=torch.device(args.device),
        generator_learning_rate=config.generator_learning_rate,
        task_learning_rate=config.task_learning_rate,
        adversarial_weight=adversarial_weight,
        prediction_weight=config.prediction_weight,
        sparsity_weight=config.sparsity_weight,
        critic_steps=config.critic_steps,
        gradient_penalty_weight=config.gradient_penalty_weight,
        masked_prediction_weight=config.masked_prediction_weight,
    )


def summarize(scores: dict[str, object], pairs: dict[str, object], weight: float) -> dict[str, object]:
    learned = scores["conditions"]["learned"]
    summary = {
        "adversarial_weight": weight,
        "pair_sampling_probability": pairs,
        "groups": {},
    }
    for group in ("all", "switch_negative", "switch_nonnegative"):
        entry = learned[group]
        summary["groups"][group] = {
            "accuracy": entry["accuracy"],
            "cross_entropy": entry["prediction"],
            "selected_features": entry["selected_features"],
            "exact_mask_rate": entry["exact_mask_accuracy"],
            "frequency_0_to_10": entry["selection_frequency_0_to_10"],
            "prediction_plus_sparsity": round(
                entry["prediction"] + TrainingConfig().sparsity_weight * entry["sparsity"], 6
            ),
        }
    return summary


def main() -> None:
    args = parse_args()
    config = TrainingConfig()
    report: dict[str, object] = {
        "settings": {**vars(args), "output": str(args.output), "training": asdict(config)},
        "notes": "Critic updates remain enabled in both arms; only its coefficient in the generator loss changes. Cross-arm comparisons use prediction_plus_sparsity, not the arm-specific total loss.",
        "seeds": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    for seed in args.seeds:
        torch.manual_seed(seed)
        features, labels, masks = generate_synthetic_data(
            num_samples=args.num_samples, input_dim=args.input_dim, dataset_type="Syn4"
        )
        split = int(args.num_samples * args.train_ratio)
        train_data = TensorDataset(features[:split], labels[:split])
        train_eval = fixed_eval_loader(
            features[:split], labels[:split], masks[:split], args.batch_size, args.eval_samples
        )
        test_eval = fixed_eval_loader(
            features[split:], labels[split:], masks[split:], args.batch_size, args.eval_samples
        )
        union = masks[:split].any(0, keepdim=True).float().to(args.device)
        seed_report = {"full_critic": {}, "no_generator_critic": {}}
        report["seeds"][str(seed)] = seed_report
        print(f"seed {seed}: train={split}, test={len(features)-split}, device={args.device}", flush=True)
        for arm, weight in (("full_critic", config.adversarial_weight), ("no_generator_critic", 0.0)):
            # Reset all network initialization, batch order and stochastic training draws.
            torch.manual_seed(seed + 1)
            trainer = make_trainer(args, config, weight)
            loader = DataLoader(
                train_data, batch_size=args.batch_size, shuffle=True,
                generator=torch.Generator().manual_seed(seed + 2),
            )
            torch.manual_seed(seed + 3)
            arm_report = {"adversarial_weight": weight, "snapshots": {}}
            seed_report[arm] = arm_report
            temperature = config.temperature_start

            def snapshot(epoch: int) -> None:
                scores = evaluate_split(trainer, test_eval, union, args.eval_repeats, seed)
                pairs = pair_probability(trainer, test_eval)
                item = {
                    "temperature": temperature,
                    "test": summarize(scores, pairs, weight),
                }
                if epoch == args.epochs:
                    train_scores = evaluate_split(trainer, train_eval, union, args.eval_repeats, seed)
                    train_pairs = pair_probability(trainer, train_eval)
                    item["train"] = summarize(train_scores, train_pairs, weight)
                arm_report["snapshots"][str(epoch)] = item
                test = item["test"]
                negative = test["groups"]["switch_negative"]
                positive = test["groups"]["switch_nonnegative"]
                print(
                    f"seed {seed} {arm} epoch {epoch}: "
                    f"acc={test['groups']['all']['accuracy']:.3f}, "
                    f"selected={test['groups']['all']['selected_features']:.2f}, "
                    f"neg/pos exact={negative['exact_mask_rate']:.3f}/{positive['exact_mask_rate']:.3f}, "
                    f"neg pair={pairs['negative']['pair_independent_sampling_probability']:.5f}",
                    flush=True,
                )
                save()

            if 0 in args.checkpoints:
                snapshot(0)
            for epoch in range(1, args.epochs + 1):
                trainer.train()
                for batch_x, batch_y in loader:
                    trainer.train_step(batch_x, batch_y, temperature)
                temperature = max(config.temperature_min, temperature * config.temperature_decay)
                if epoch in args.checkpoints:
                    snapshot(epoch)
        save()
    print(f"Saved report: {args.output}", flush=True)


if __name__ == "__main__":
    main()

"""Compare oracle and union masks under the same jointly trained ALTAS models.

Run from the repository root with ``python -m diagnostics.counterfactual_masks``.
This re-runs ALTAS training without changing its loss or update sequence.
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
from data import create_data_loaders, generate_synthetic_data
from trainer import ALTASTrainer


CONDITIONS = ("oracle", "union", "learned")
GROUPS = ("all", "switch_negative", "switch_nonnegative")
COMPONENTS = ("prediction", "adversarial", "sparsity", "total")


def parse_args() -> argparse.Namespace:
    data_defaults = DataConfig()
    model_defaults = ModelConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("Syn4", "Syn5", "Syn6"), default="Syn4")
    parser.add_argument("--num-samples", type=int, default=data_defaults.num_samples)
    parser.add_argument("--input-dim", type=int, default=data_defaults.input_dim)
    parser.add_argument("--train-ratio", type=float, default=data_defaults.train_ratio)
    parser.add_argument("--batch-size", type=int, default=data_defaults.batch_size)
    parser.add_argument("--hidden-dim", type=int, default=model_defaults.hidden_dim)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--checkpoints", type=str, default="0,50,100,200")
    parser.add_argument("--eval-samples", type=int, default=4_000)
    parser.add_argument("--eval-repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    parser.add_argument("--model-output", type=Path, help="Optional final four-network checkpoint")
    args = parser.parse_args()

    if args.num_samples < 2 or args.input_dim < 11:
        parser.error("num-samples must be >= 2 and input-dim must be >= 11")
    if not 0 < args.train_ratio < 1:
        parser.error("train-ratio must lie strictly between 0 and 1")
    split = int(args.num_samples * args.train_ratio)
    if not 0 < split < args.num_samples:
        parser.error("train-ratio must leave samples in both splits")
    for name in ("batch_size", "hidden_dim", "epochs", "eval_samples", "eval_repeats"):
        if getattr(args, name) < 1:
            parser.error(f"{name.replace('_', '-')} must be >= 1")
    try:
        parsed_checkpoints = {int(value.strip()) for value in args.checkpoints.split(",")}
    except ValueError:
        parser.error("checkpoints must be comma-separated nonnegative epoch numbers")
    if any(epoch < 0 for epoch in parsed_checkpoints):
        parser.error("checkpoints must be nonnegative")
    args.checkpoints = sorted(
        {epoch for epoch in parsed_checkpoints if epoch <= args.epochs} | {args.epochs}
    )
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available")
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return args


def fixed_eval_loader(
    features: torch.Tensor,
    labels: torch.Tensor,
    true_masks: torch.Tensor,
    batch_size: int,
    max_samples: int,
) -> DataLoader:
    return DataLoader(
        TensorDataset(
            features[:max_samples], labels[:max_samples], true_masks[:max_samples]
        ),
        batch_size=batch_size,
        shuffle=False,
    )


def evaluate_split(
    trainer: ALTASTrainer,
    loader: DataLoader,
    union_mask: torch.Tensor,
    repeats: int,
    seed: int,
) -> dict[str, object]:
    device = trainer.device
    totals = {
        condition: {
            group: {
                "count": 0,
                "prediction": 0.0,
                "adversarial": 0.0,
                "sparsity": 0.0,
                "total": 0.0,
                "correct": 0,
                "selected": 0.0,
                "exact": 0,
            }
            for group in GROUPS
        }
        for condition in CONDITIONS
    }
    learned_feature_totals = {
        group: torch.zeros(11, device=device) for group in GROUPS
    }
    original_modes = [module.training for module in trainer.modules()]
    cuda_devices = [torch.cuda.current_device()] if device.type == "cuda" else []
    try:
        trainer.eval()
        with torch.no_grad(), torch.random.fork_rng(devices=cuda_devices):
            for repeat in range(repeats):
                torch.manual_seed(seed + 10_000 + repeat)
                for input_batch, label_batch, true_mask_batch in loader:
                    inputs = input_batch.to(device)
                    labels = label_batch.to(device)
                    oracle_mask = true_mask_batch.to(device)
                    masks = {
                        "oracle": oracle_mask,
                        "union": union_mask.expand_as(oracle_mask),
                    }
                    pair_logits = trainer.generator.network(inputs).reshape(
                        -1, inputs.size(1), 2
                    )
                    keep_probabilities = pair_logits.softmax(dim=-1)[..., 1]
                    masks["learned"] = (keep_probabilities >= 0.5).float()

                    # Identical shuffled replacement for all three masks.
                    shuffled = inputs[torch.randperm(len(inputs), device=device)]
                    members_by_group = {
                        "all": torch.ones_like(labels, dtype=torch.bool),
                        "switch_negative": inputs[:, 10] < 0,
                        "switch_nonnegative": inputs[:, 10] >= 0,
                    }
                    for condition, mask in masks.items():
                        masked_inputs = inputs * mask + shuffled * (1.0 - mask)
                        latent = trainer.extractor(masked_inputs)
                        logits = trainer.predictor(latent)
                        prediction = F.cross_entropy(logits, labels, reduction="none")
                        adversarial = -trainer.critic(latent).flatten()
                        if condition == "learned":
                            sparsity = keep_probabilities.mean(dim=1)
                        else:
                            sparsity = mask.mean(dim=1)
                        total = (
                            trainer.prediction_weight * prediction
                            + trainer.adversarial_weight * adversarial
                            + trainer.sparsity_weight * sparsity
                        )
                        correct = logits.argmax(dim=1).eq(labels)
                        selected = mask.sum(dim=1)
                        exact = mask.bool().eq(oracle_mask.bool()).all(dim=1)
                        for group, members in members_by_group.items():
                            entry = totals[condition][group]
                            entry["count"] += int(members.sum().item())
                            entry["prediction"] += prediction[members].sum().item()
                            entry["adversarial"] += adversarial[members].sum().item()
                            entry["sparsity"] += sparsity[members].sum().item()
                            entry["total"] += total[members].sum().item()
                            entry["correct"] += int(correct[members].sum().item())
                            entry["selected"] += selected[members].sum().item()
                            entry["exact"] += int(exact[members].sum().item())
                            if condition == "learned":
                                learned_feature_totals[group] += mask[members, :11].sum(dim=0)
    finally:
        for module, was_training in zip(trainer.modules(), original_modes):
            module.train(was_training)

    scores: dict[str, dict[str, dict[str, float | int | None]]] = {}
    for condition in CONDITIONS:
        scores[condition] = {}
        for group in GROUPS:
            entry = totals[condition][group]
            count = entry["count"]
            scores[condition][group] = {
                "samples": count // repeats,
                **{
                    component: round(entry[component] / count, 6) if count else None
                    for component in COMPONENTS
                },
                "accuracy": round(entry["correct"] / count, 6) if count else None,
                "selected_features": round(entry["selected"] / count, 4) if count else None,
                "exact_mask_accuracy": round(entry["exact"] / count, 6) if count else None,
            }
            if condition == "learned":
                scores[condition][group]["selection_frequency_0_to_10"] = (
                    [round(value / count, 5) for value in learned_feature_totals[group].tolist()]
                    if count
                    else None
                )

    delta = {}
    for group in GROUPS:
        oracle = scores["oracle"][group]
        union = scores["union"][group]
        delta[group] = {
            component: round(oracle[component] - union[component], 6)
            if oracle[component] is not None and union[component] is not None
            else None
            for component in COMPONENTS
        }
    return {"conditions": scores, "oracle_minus_union": delta}


def save_report(output: Path | None, report: dict[str, object]) -> None:
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    training_config = TrainingConfig()
    torch.manual_seed(args.seed)
    features, labels, true_masks = generate_synthetic_data(
        num_samples=args.num_samples,
        input_dim=args.input_dim,
        dataset_type=args.dataset,
    )
    train_loader, _ = create_data_loaders(
        features,
        labels,
        true_masks,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
    )
    split = int(args.num_samples * args.train_ratio)
    train_eval_loader = fixed_eval_loader(
        features[:split], labels[:split], true_masks[:split], args.batch_size, args.eval_samples
    )
    test_eval_loader = fixed_eval_loader(
        features[split:], labels[split:], true_masks[split:], args.batch_size, args.eval_samples
    )
    device = torch.device(args.device)
    union_mask = true_masks[:split].any(dim=0, keepdim=True).float().to(device)
    trainer = ALTASTrainer(
        input_dim=args.input_dim,
        num_classes=2,
        hidden_dim=args.hidden_dim,
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
    report: dict[str, object] = {
        "settings": {
            "dataset": args.dataset,
            "num_samples": args.num_samples,
            "input_dim": args.input_dim,
            "train_ratio": args.train_ratio,
            "batch_size": args.batch_size,
            "hidden_dim": args.hidden_dim,
            "epochs": args.epochs,
            "checkpoints": args.checkpoints,
            "eval_samples": args.eval_samples,
            "eval_repeats": args.eval_repeats,
            "seed": args.seed,
            "device": args.device,
            "training": asdict(training_config),
        },
        "snapshots": {},
    }
    print(
        f"{args.dataset}: train={split}, test={len(features) - split}, "
        f"batch={args.batch_size}, device={device}, "
        f"union features={int(union_mask.sum().item())}",
        flush=True,
    )

    def snapshot(epoch: int, temperature: float) -> None:
        train_scores = evaluate_split(
            trainer, train_eval_loader, union_mask, args.eval_repeats, args.seed
        )
        test_scores = evaluate_split(
            trainer, test_eval_loader, union_mask, args.eval_repeats, args.seed
        )
        report["snapshots"][str(epoch)] = {
            "temperature": temperature,
            "train": train_scores,
            "test": test_scores,
        }
        delta = test_scores["oracle_minus_union"]["all"]
        learned = test_scores["conditions"]["learned"]["all"]
        print(
            f"epoch {epoch:>4}: test oracle-union "
            f"CE={delta['prediction']:+.5f}, adv={delta['adversarial']:+.5f}, "
            f"sparse={delta['sparsity']:+.5f}, total={delta['total']:+.5f}; "
            f"learned selected={learned['selected_features']:.2f}",
            flush=True,
        )
        save_report(args.output, report)

    temperature = training_config.temperature_start
    if 0 in args.checkpoints:
        snapshot(0, temperature)
    trainer.train()
    for epoch in range(1, args.epochs + 1):
        metric_sums = {name: 0.0 for name in ALTASTrainer.METRIC_NAMES}
        for input_batch, label_batch, _ in train_loader:
            metrics = trainer.train_step(input_batch, label_batch, temperature)
            for name, value in metrics.items():
                metric_sums[name] += value
        temperature = max(
            training_config.temperature_min,
            temperature * training_config.temperature_decay,
        )
        if epoch in args.checkpoints:
            print(
                f"epoch {epoch:>4}: train masked acc="
                f"{metric_sums['masked_accuracy'] / len(train_loader):.2%}, "
                f"full acc={metric_sums['full_accuracy'] / len(train_loader):.2%}",
                flush=True,
            )
            snapshot(epoch, temperature)

    if args.model_output is not None:
        args.model_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "settings": report["settings"],
                "generator": trainer.generator.state_dict(),
                "extractor": trainer.extractor.state_dict(),
                "critic": trainer.critic.state_dict(),
                "predictor": trainer.predictor.state_dict(),
            },
            args.model_output,
        )
        print(f"Saved model checkpoint: {args.model_output}", flush=True)
    if args.output is not None:
        print(f"Saved report: {args.output}", flush=True)


if __name__ == "__main__":
    main()

"""Test whether the existing predictor can learn from instance-specific masks.

Run from the repository root with ``python -m diagnostics.oracle_mask``.
This experiment does not train or modify the ALTAS generator or critic.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from data import generate_synthetic_data
from masking import apply_shuffle_replacement_mask
from models import FeatureExtractor, TaskPredictor


CONDITIONS = ("oracle", "union", "full")
GROUPS = ("all", "switch_negative", "switch_nonnegative")


@dataclass(frozen=True)
class Settings:
    dataset: str
    num_samples: int
    input_dim: int
    train_ratio: float
    batch_size: int
    hidden_dim: int
    epochs: int
    learning_rate: float
    weight_decay: float
    auxiliary_full_weight: float
    eval_repeats: int
    seed: int
    device: str


def parse_args() -> tuple[Settings, Path | None, int]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("Syn4", "Syn5", "Syn6"), default="Syn4")
    parser.add_argument("--num-samples", type=int, default=20_000)
    parser.add_argument("--input-dim", type=int, default=100)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--auxiliary-full-weight", type=float, default=0.0)
    parser.add_argument("--eval-repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    args = parser.parse_args()

    if args.num_samples < 2 or args.input_dim < 11:
        parser.error("num-samples must be >= 2 and input-dim must be >= 11")
    if not 0 < args.train_ratio < 1:
        parser.error("train-ratio must lie strictly between 0 and 1")
    if int(args.num_samples * args.train_ratio) not in range(1, args.num_samples):
        parser.error("train-ratio must leave at least one sample in both splits")
    for name in ("batch_size", "hidden_dim", "epochs", "eval_repeats", "log_interval"):
        if getattr(args, name) < 1:
            parser.error(f"{name.replace('_', '-')} must be >= 1")
    if args.learning_rate <= 0 or args.weight_decay < 0 or args.auxiliary_full_weight < 0:
        parser.error("learning-rate must be positive; weights must be nonnegative")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available")

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    settings = Settings(
        dataset=args.dataset,
        num_samples=args.num_samples,
        input_dim=args.input_dim,
        train_ratio=args.train_ratio,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        auxiliary_full_weight=args.auxiliary_full_weight,
        eval_repeats=args.eval_repeats,
        seed=args.seed,
        device=device,
    )
    return settings, args.output, args.log_interval


def masked_inputs(
    inputs: torch.Tensor,
    true_masks: torch.Tensor,
    union_mask: torch.Tensor,
    condition: str,
) -> torch.Tensor:
    if condition == "full":
        return inputs
    mask = true_masks if condition == "oracle" else union_mask.expand_as(true_masks)
    return apply_shuffle_replacement_mask(inputs, mask)


def make_loaders(
    features: torch.Tensor,
    labels: torch.Tensor,
    true_masks: torch.Tensor,
    settings: Settings,
) -> tuple[DataLoader, DataLoader]:
    split = int(len(features) * settings.train_ratio)
    dataset = TensorDataset(features, labels, true_masks)
    train_data = torch.utils.data.Subset(dataset, range(split))
    test_data = torch.utils.data.Subset(dataset, range(split, len(features)))
    train_loader = DataLoader(
        train_data,
        batch_size=settings.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(settings.seed + 1),
    )
    test_loader = DataLoader(test_data, batch_size=settings.batch_size)
    return train_loader, test_loader


def make_model(settings: Settings, device: torch.device) -> nn.Module:
    return nn.Sequential(
        FeatureExtractor(settings.input_dim, settings.hidden_dim),
        TaskPredictor(settings.hidden_dim, 2),
    ).to(device)


def train_condition(
    condition: str,
    features: torch.Tensor,
    labels: torch.Tensor,
    true_masks: torch.Tensor,
    union_mask: torch.Tensor,
    settings: Settings,
    log_interval: int,
) -> dict[str, object]:
    device = torch.device(settings.device)
    torch.manual_seed(settings.seed + 2)
    model = make_model(settings, device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay
    )
    train_loader, test_loader = make_loaders(features, labels, true_masks, settings)

    start = time.perf_counter()
    for epoch in range(1, settings.epochs + 1):
        model.train()
        loss_total = 0.0
        count = 0
        for input_batch, label_batch, mask_batch in train_loader:
            inputs = input_batch.to(device)
            targets = label_batch.to(device)
            masks = mask_batch.to(device)
            selected = masked_inputs(inputs, masks, union_mask, condition)
            loss = F.cross_entropy(model(selected), targets)
            if condition != "full" and settings.auxiliary_full_weight:
                loss = loss + settings.auxiliary_full_weight * F.cross_entropy(
                    model(inputs), targets
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_total += loss.detach().item() * len(targets)
            count += len(targets)

        if epoch % log_interval == 0 or epoch == settings.epochs:
            print(f"{condition:>6} epoch {epoch:>4}/{settings.epochs}: train loss {loss_total / count:.4f}", flush=True)

    elapsed = time.perf_counter() - start
    scores = evaluate_condition(
        model, condition, test_loader, union_mask, settings.eval_repeats, settings.seed, device
    )
    return {"train_seconds": round(elapsed, 3), "test": scores}


@torch.no_grad()
def evaluate_condition(
    model: nn.Module,
    condition: str,
    test_loader: DataLoader,
    union_mask: torch.Tensor,
    repeats: int,
    seed: int,
    device: torch.device,
) -> dict[str, dict[str, float | int]]:
    model.eval()
    totals = {group: {"loss": 0.0, "correct": 0, "count": 0} for group in GROUPS}
    cuda_devices = [torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=cuda_devices):
        for repeat in range(repeats):
            torch.manual_seed(seed + 10_000 + repeat)
            for input_batch, label_batch, mask_batch in test_loader:
                inputs = input_batch.to(device)
                targets = label_batch.to(device)
                masks = mask_batch.to(device)
                selected = masked_inputs(inputs, masks, union_mask, condition)
                logits = model(selected)
                losses = F.cross_entropy(logits, targets, reduction="none")
                correct = logits.argmax(dim=1).eq(targets)
                for group, members in (
                    ("all", torch.ones_like(targets, dtype=torch.bool)),
                    ("switch_negative", inputs[:, 10] < 0),
                    ("switch_nonnegative", inputs[:, 10] >= 0),
                ):
                    totals[group]["loss"] += losses[members].sum().item()
                    totals[group]["correct"] += correct[members].sum().item()
                    totals[group]["count"] += members.sum().item()

    return {
        group: {
            "samples": totals[group]["count"] // repeats,
            "cross_entropy": round(totals[group]["loss"] / totals[group]["count"], 5),
            "accuracy": round(totals[group]["correct"] / totals[group]["count"], 5),
        }
        for group in GROUPS
    }


def main() -> None:
    settings, output_path, log_interval = parse_args()
    torch.manual_seed(settings.seed)
    features, labels, true_masks = generate_synthetic_data(
        num_samples=settings.num_samples,
        input_dim=settings.input_dim,
        dataset_type=settings.dataset,
    )
    union_mask = true_masks.any(dim=0, keepdim=True).float().to(settings.device)
    split = int(len(features) * settings.train_ratio)
    print(
        f"{settings.dataset}: train={split}, test={len(features) - split}, "
        f"batch={settings.batch_size}, device={settings.device}; "
        f"oracle mean features={true_masks.float().sum(dim=1).mean().item():.2f}, "
        f"union features={int(union_mask.sum().item())}",
        flush=True,
    )

    results = {}
    for condition in CONDITIONS:
        results[condition] = train_condition(
            condition, features, labels, true_masks, union_mask, settings, log_interval
        )
        summary = results[condition]["test"]
        print(
            f"{condition:>6}: overall CE={summary['all']['cross_entropy']:.4f}, "
            f"acc={summary['all']['accuracy']:.2%}; "
            f"negative acc={summary['switch_negative']['accuracy']:.2%}, "
            f"nonnegative acc={summary['switch_nonnegative']['accuracy']:.2%}",
            flush=True,
        )

    report = {"settings": asdict(settings), "conditions": results}
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved report: {output_path}")


if __name__ == "__main__":
    main()

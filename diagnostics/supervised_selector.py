"""Check whether the existing generator can represent conditional Syn4–Syn6 masks.

Run from the repository root: ``python -m diagnostics.supervised_selector``.
Ground-truth masks are used only in this diagnostic experiment, not in ALTAS.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from data import generate_synthetic_data
from models import FeatureMaskGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("Syn4", "Syn5", "Syn6"), default="Syn4")
    parser.add_argument("--num-samples", type=int, default=20_000)
    parser.add_argument("--input-dim", type=int, default=100)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    args = parser.parse_args()

    if args.num_samples < 2 or args.input_dim < 11:
        parser.error("num-samples must be >= 2 and input-dim must be >= 11")
    if not 0 < args.train_ratio < 1:
        parser.error("train-ratio must lie strictly between 0 and 1")
    split = int(args.num_samples * args.train_ratio)
    if not 0 < split < args.num_samples:
        parser.error("train-ratio must leave samples in both splits")
    for name in ("batch_size", "hidden_dim", "epochs", "log_interval"):
        if getattr(args, name) < 1:
            parser.error(f"{name.replace('_', '-')} must be >= 1")
    if args.learning_rate <= 0:
        parser.error("learning-rate must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available")
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return args


def selector_logits(generator: FeatureMaskGenerator, inputs: torch.Tensor) -> torch.Tensor:
    """Return keep-minus-drop logits from the unchanged generator network."""
    pair_logits = generator.network(inputs).reshape(-1, generator.input_dim, 2)
    return pair_logits[..., 1] - pair_logits[..., 0]


def evaluate(
    generator: FeatureMaskGenerator,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, object]:
    generator.eval()
    probabilities = []
    predictions = []
    targets = []
    switches = []
    with torch.no_grad():
        for input_batch, target_batch in loader:
            inputs = input_batch.to(device)
            keep_probabilities = torch.sigmoid(selector_logits(generator, inputs))
            probabilities.append(keep_probabilities.cpu())
            predictions.append((keep_probabilities >= 0.5).cpu())
            targets.append(target_batch.bool())
            switches.append(input_batch[:, 10] < 0)

    probs = torch.cat(probabilities)
    pred = torch.cat(predictions)
    truth = torch.cat(targets)
    switch_negative = torch.cat(switches)
    true_positive = (pred & truth).sum(dim=1).float()
    selected = pred.sum(dim=1).float()
    relevant = truth.sum(dim=1).float()
    tpr = (true_positive / relevant.clamp_min(1)).mean().item()
    fdr = ((selected - true_positive) / selected.clamp_min(1)).mean().item()

    by_switch = {}
    for name, members in (
        ("switch_negative", switch_negative),
        ("switch_nonnegative", ~switch_negative),
    ):
        if not members.any():
            by_switch[name] = {"samples": 0}
            continue
        by_switch[name] = {
            "samples": int(members.sum().item()),
            "exact_mask_accuracy": round(pred[members].eq(truth[members]).all(dim=1).float().mean().item(), 5),
            "selected_features_mean": round(selected[members].mean().item(), 3),
            "selection_frequency_0_to_10": [
                round(value, 5) for value in pred[members, :11].float().mean(dim=0).tolist()
            ],
            "keep_probability_0_to_10": [
                round(value, 5) for value in probs[members, :11].mean(dim=0).tolist()
            ],
            "true_frequency_0_to_10": [
                round(value, 5) for value in truth[members, :11].float().mean(dim=0).tolist()
            ],
        }

    return {
        "samples": len(pred),
        "exact_mask_accuracy": round(pred.eq(truth).all(dim=1).float().mean().item(), 5),
        "true_positive_rate": round(tpr, 5),
        "false_discovery_rate": round(fdr, 5),
        "selected_features_mean": round(selected.mean().item(), 3),
        "by_switch": by_switch,
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    features, _, true_masks = generate_synthetic_data(
        num_samples=args.num_samples,
        input_dim=args.input_dim,
        dataset_type=args.dataset,
    )
    split = int(args.num_samples * args.train_ratio)
    train_dataset = TensorDataset(features[:split], true_masks[:split])
    test_dataset = TensorDataset(features[split:], true_masks[split:])
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed + 1),
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    torch.manual_seed(args.seed + 2)
    generator = FeatureMaskGenerator(args.input_dim, args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(
        generator.parameters(), lr=args.learning_rate, betas=(0.5, 0.9)
    )
    candidate_features = true_masks[:split].any(dim=0).to(device)
    print(
        f"{args.dataset}: train={split}, test={args.num_samples - split}, "
        f"batch={args.batch_size}, device={device}, "
        f"candidate features={int(candidate_features.sum().item())}",
        flush=True,
    )

    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        generator.train()
        loss_sum = 0.0
        for input_batch, target_batch in train_loader:
            inputs = input_batch.to(device)
            targets = target_batch.to(device)
            per_entry_loss = F.binary_cross_entropy_with_logits(
                selector_logits(generator, inputs), targets, reduction="none"
            )
            # Balance feature groups, not positive labels: conditional features
            # are positive for only half the samples, while noise is always zero.
            loss = per_entry_loss[:, candidate_features].mean()
            loss = loss + per_entry_loss[:, ~candidate_features].mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += loss.detach().item() * len(inputs)
        if epoch % args.log_interval == 0 or epoch == args.epochs:
            print(
                f"epoch {epoch:>4}/{args.epochs}: grouped train BCE "
                f"{loss_sum / len(train_dataset):.5f}",
                flush=True,
            )

    result = evaluate(generator, test_loader, device)
    print(
        f"Test exact mask={result['exact_mask_accuracy']:.2%}, "
        f"TPR={result['true_positive_rate']:.2%}, "
        f"FDR={result['false_discovery_rate']:.2%}, "
        f"selected={result['selected_features_mean']:.2f}",
        flush=True,
    )
    for name, group in result["by_switch"].items():
        print(
            f"{name}: n={group['samples']}, exact={group.get('exact_mask_accuracy', 0):.2%}, "
            f"frequency[0:11]={group.get('selection_frequency_0_to_10', [])}",
            flush=True,
        )

    report = {
        "settings": {
            "dataset": args.dataset,
            "num_samples": args.num_samples,
            "input_dim": args.input_dim,
            "train_ratio": args.train_ratio,
            "batch_size": args.batch_size,
            "hidden_dim": args.hidden_dim,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
            "device": args.device,
            "candidate_features": int(candidate_features.sum().item()),
        },
        "train_seconds": round(time.perf_counter() - started, 3),
        "test": result,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved report: {args.output}")


if __name__ == "__main__":
    main()

"""Locate the ALTAS failure with an oracle-trained, frozen task model.

The oracle mask is used only to train the diagnostic predictor and to score
counterfactual masks. This is not a deployable unsupervised training method.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from config import DataConfig, ModelConfig, TrainingConfig
from data import generate_synthetic_data
from masking import apply_shuffle_replacement_mask
from models import FeatureExtractor, FeatureMaskGenerator, TaskPredictor


class SwitchVectorGenerator(nn.Module):
    """Diagnostic upper bound: known switch, learned masks, no oracle-mask loss."""

    def __init__(self, input_dim: int, union: torch.Tensor, keep_probability: float) -> None:
        super().__init__()
        keep_logit = math.log(keep_probability / (1 - keep_probability))
        logits = torch.zeros(2, input_dim, 2, device=union.device)
        logits[:, :, 1] = torch.where(
            union[0].bool(),
            torch.full_like(union[0], keep_logit),
            torch.full_like(union[0], -keep_logit),
        )
        self.logits = nn.Parameter(logits)

    def selected_logits(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.logits[(inputs[:, 10] >= 0).long()]

    def forward(self, inputs: torch.Tensor, temperature: float = 1.0, hard: bool = True):
        logits = self.selected_logits(inputs)
        probabilities = logits.softmax(-1)[..., 1]
        mask = F.gumbel_softmax(logits, tau=temperature, hard=hard, dim=-1)[..., 1]
        return probabilities, mask


def keep_probabilities(generator: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    if isinstance(generator, SwitchVectorGenerator):
        logits = generator.selected_logits(inputs)
    else:
        logits = generator.network(inputs).reshape(-1, inputs.size(1), 2)
    return logits.softmax(-1)[..., 1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-samples", type=int, default=DataConfig().num_samples)
    parser.add_argument("--input-dim", type=int, default=DataConfig().input_dim)
    parser.add_argument("--train-ratio", type=float, default=DataConfig().train_ratio)
    parser.add_argument("--batch-size", type=int, default=DataConfig().batch_size)
    parser.add_argument("--hidden-dim", type=int, default=ModelConfig().hidden_dim)
    parser.add_argument("--predictor-epochs", type=int, default=200)
    parser.add_argument("--selector-epochs", type=int, default=200)
    parser.add_argument("--checkpoints", default="0,50,100,200")
    parser.add_argument("--selector-init", choices=("union", "random"), default="union")
    parser.add_argument("--selector-kind", choices=("mlp", "switch_vectors"), default="mlp")
    parser.add_argument("--selector-learning-rate", type=float, default=TrainingConfig().generator_learning_rate)
    parser.add_argument("--union-keep-prob", type=float, default=0.9)
    parser.add_argument("--eval-repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--output", type=Path, default=Path("diagnostics/results/syn4_frozen_oracle_selector.json")
    )
    args = parser.parse_args()
    if args.num_samples < 2 or args.input_dim < 11 or args.batch_size < 1 or args.hidden_dim < 1:
        parser.error("num-samples >= 2, input-dim >= 11, batch-size and hidden-dim >= 1 required")
    split = int(args.num_samples * args.train_ratio)
    if not 0 < args.train_ratio < 1 or not 0 < split < args.num_samples:
        parser.error("train-ratio must leave both training and test samples")
    if min(args.predictor_epochs, args.selector_epochs, args.eval_repeats) < 1:
        parser.error("epoch counts and eval-repeats must be positive")
    if args.selector_learning_rate <= 0 or not 0.5 < args.union_keep_prob < 1:
        parser.error("selector-learning-rate > 0 and 0.5 < union-keep-prob < 1 required")
    try:
        checkpoints = {int(value.strip()) for value in args.checkpoints.split(",")}
    except ValueError:
        parser.error("checkpoints must be comma-separated integers")
    if any(value < 0 for value in checkpoints):
        parser.error("checkpoints must be nonnegative")
    args.checkpoints = sorted({value for value in checkpoints if value <= args.selector_epochs} | {args.selector_epochs})
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    return args


def evaluate(
    extractor: FeatureExtractor,
    predictor: TaskPredictor,
    generator: nn.Module,
    loader: DataLoader,
    union: torch.Tensor,
    sparsity_weight: float,
    repeats: int,
    seed: int,
    device: torch.device,
) -> dict[str, object]:
    groups = ("all", "negative", "nonnegative")
    ablations = {
        "drop_0": (0,), "drop_1": (1,), "drop_01": (0, 1),
        "drop_2": (2,), "drop_3": (3,), "drop_4": (4,), "drop_5": (5,),
        "drop_2345": (2, 3, 4, 5),
    }
    conditions = ("oracle", "union", "learned", *ablations)
    totals = {
        condition: {
            group: {"count": 0, "ce": 0.0, "correct": 0, "selected": 0.0,
                    "exact": 0, "feature_counts": torch.zeros(11, device=device)}
            for group in groups
        }
        for condition in conditions
    }
    probability_totals = {group: torch.zeros(11, device=device) for group in groups}
    cuda_devices = [torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.no_grad(), torch.random.fork_rng(devices=cuda_devices):
        for repeat in range(repeats):
            torch.manual_seed(seed + 10_000 + repeat)
            for batch_x, batch_y, batch_mask in loader:
                x, y, oracle = batch_x.to(device), batch_y.to(device), batch_mask.to(device)
                p = keep_probabilities(generator, x)
                masks = {"oracle": oracle, "union": union.expand_as(oracle),
                         "learned": (p >= 0.5).float()}
                for name, indices in ablations.items():
                    mask = union.expand_as(oracle).clone()
                    mask[:, indices] = 0
                    masks[name] = mask
                shuffled = x[torch.randperm(len(x), device=device)]
                members = {"all": torch.ones_like(y, dtype=torch.bool),
                           "negative": x[:, 10] < 0, "nonnegative": x[:, 10] >= 0}
                for group, take in members.items():
                    probability_totals[group] += p[take, :11].sum(0)
                for condition, mask in masks.items():
                    logits = predictor(extractor(x * mask + shuffled * (1 - mask)))
                    losses = F.cross_entropy(logits, y, reduction="none")
                    correct = logits.argmax(1).eq(y)
                    for group, take in members.items():
                        item = totals[condition][group]
                        item["count"] += int(take.sum().item())
                        item["ce"] += losses[take].sum().item()
                        item["correct"] += int(correct[take].sum().item())
                        item["selected"] += mask[take].sum().item()
                        item["exact"] += int(mask[take].bool().eq(oracle[take].bool()).all(1).sum().item())
                        item["feature_counts"] += mask[take, :11].sum(0)
    result: dict[str, object] = {}
    for condition in conditions:
        result[condition] = {}
        for group in groups:
            item = totals[condition][group]
            n = item["count"]
            ce = item["ce"] / n
            selected = item["selected"] / n
            result[condition][group] = {
                "samples": n // repeats,
                "ce": round(ce, 6),
                "accuracy": round(item["correct"] / n, 6),
                "selected": round(selected, 4),
                "objective": round(ce + sparsity_weight * selected / union.size(1), 6),
                "exact_mask_rate": round(item["exact"] / n, 6),
                "frequency_0_to_10": [round(v / n, 5) for v in item["feature_counts"].tolist()],
            }
    result["oracle_minus_union"] = {
        group: round(result["oracle"][group]["objective"] - result["union"][group]["objective"], 6)
        for group in groups
    }
    result["learned_probabilities_0_to_10"] = {
        group: [round(v / totals["learned"][group]["count"], 6)
                for v in probability_totals[group].tolist()]
        for group in groups
    }
    return result


def main() -> None:
    args = parse_args()
    config = TrainingConfig()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    features, labels, true_masks = generate_synthetic_data(
        num_samples=args.num_samples, input_dim=args.input_dim, dataset_type="Syn4"
    )
    split = int(args.num_samples * args.train_ratio)
    train_data = TensorDataset(features[:split], labels[:split], true_masks[:split])
    test_loader = DataLoader(
        TensorDataset(features[split:], labels[split:], true_masks[split:]),
        batch_size=args.batch_size, shuffle=False,
    )
    union = true_masks[:split].any(0, keepdim=True).float().to(device)
    torch.manual_seed(args.seed + 2)
    extractor = FeatureExtractor(args.input_dim, args.hidden_dim).to(device)
    predictor = TaskPredictor(args.hidden_dim, 2).to(device)
    optimizer = torch.optim.Adam(
        list(extractor.parameters()) + list(predictor.parameters()), lr=config.task_learning_rate,
        weight_decay=1e-4,
    )
    predictor_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(args.seed + 1),
    )
    print(f"Syn4 train={split}, test={len(features)-split}, device={device}", flush=True)
    for epoch in range(1, args.predictor_epochs + 1):
        extractor.train()
        predictor.train()
        total = 0.0
        for batch_x, batch_y, batch_mask in predictor_loader:
            x, y, mask = batch_x.to(device), batch_y.to(device), batch_mask.to(device)
            logits = predictor(extractor(apply_shuffle_replacement_mask(x, mask)))
            loss = F.cross_entropy(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item()
        if epoch == 1 or epoch % 20 == 0 or epoch == args.predictor_epochs:
            print(f"predictor epoch {epoch}: loss={total/len(predictor_loader):.4f}", flush=True)

    extractor.eval()
    predictor.eval()
    for module in (extractor, predictor):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    torch.manual_seed(args.seed + 3)
    if args.selector_kind == "switch_vectors":
        generator = SwitchVectorGenerator(args.input_dim, union, args.union_keep_prob).to(device)
    else:
        generator = FeatureMaskGenerator(args.input_dim, args.hidden_dim).to(device)
    if args.selector_kind == "mlp" and args.selector_init == "union":
        with torch.no_grad():
            final_layer = generator.network[-1]
            final_layer.bias[0::2] = 0.0
            keep_logit = math.log(args.union_keep_prob / (1 - args.union_keep_prob))
            final_layer.bias[1::2] = torch.where(
                union[0].bool(),
                torch.full_like(union[0], keep_logit),
                torch.full_like(union[0], -keep_logit),
            )
    selector_optimizer = torch.optim.Adam(
        generator.parameters(), lr=args.selector_learning_rate, betas=(0.5, 0.9)
    )
    selector_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(args.seed + 4),
    )
    report = {
        "settings": {**vars(args), "output": str(args.output), "training": asdict(config)},
        "notes": "Oracle masks supervise only the frozen diagnostic predictor; selector sees labels but not true masks. switch_vectors is given the known Syn4 switch feature as a diagnostic upper bound.",
        "snapshots": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def snapshot(epoch: int) -> None:
        scores = evaluate(
            extractor, predictor, generator, test_loader, union,
            config.sparsity_weight, args.eval_repeats, args.seed, device,
        )
        report["snapshots"][str(epoch)] = scores
        negative = scores["learned"]["negative"]
        positive = scores["learned"]["nonnegative"]
        print(
            f"selector epoch {epoch}: oracle-union objective={scores['oracle_minus_union']['all']:+.5f}; "
            f"negative exact={negative['exact_mask_rate']:.3f}, "
            f"positive exact={positive['exact_mask_rate']:.3f}, "
            f"selected={scores['learned']['all']['selected']:.2f}, "
            f"learned objective={scores['learned']['all']['objective']:.5f}",
            flush=True,
        )
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if 0 in args.checkpoints:
        snapshot(0)
    temperature = config.temperature_start
    for epoch in range(1, args.selector_epochs + 1):
        generator.train()
        loss_total = 0.0
        for batch_x, batch_y, _ in selector_loader:
            x, y = batch_x.to(device), batch_y.to(device)
            probabilities, mask = generator(x, temperature=temperature, hard=True)
            logits = predictor(extractor(apply_shuffle_replacement_mask(x, mask)))
            loss = F.cross_entropy(logits, y) + config.sparsity_weight * probabilities.mean()
            selector_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            selector_optimizer.step()
            loss_total += loss.item()
        temperature = max(config.temperature_min, temperature * config.temperature_decay)
        if epoch in args.checkpoints:
            print(f"selector epoch {epoch}: train loss={loss_total/len(selector_loader):.4f}", flush=True)
            generator.eval()
            snapshot(epoch)
    print(f"Saved report: {args.output}", flush=True)


if __name__ == "__main__":
    main()

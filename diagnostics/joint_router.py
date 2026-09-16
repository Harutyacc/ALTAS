"""Oracle-free joint predictor/learned-router diagnostic for Syn4.

Predictor updates exactly mirror ALTAS's full + 0.5 masked task loss, using
the router's current sampled masks. Selector updates omit only the Critic and
use two mask experts with optional high-confidence expert updates.
True masks are loaded only for held-out evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from config import DataConfig, ModelConfig, TrainingConfig
from data import generate_synthetic_data
from diagnostics.learned_router import LearnedRouterMasks, evaluate
from masking import apply_shuffle_replacement_mask
from models import FeatureExtractor, TaskPredictor


@contextmanager
def frozen_parameters(*modules: nn.Module):
    parameters = [parameter for module in modules for parameter in module.parameters()]
    original = [parameter.requires_grad for parameter in parameters]
    try:
        for parameter in parameters:
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, requires_grad in zip(parameters, original):
            parameter.requires_grad_(requires_grad)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-samples", type=int, default=DataConfig().num_samples)
    parser.add_argument("--input-dim", type=int, default=DataConfig().input_dim)
    parser.add_argument("--train-ratio", type=float, default=DataConfig().train_ratio)
    parser.add_argument("--batch-size", type=int, default=DataConfig().batch_size)
    parser.add_argument("--hidden-dim", type=int, default=ModelConfig().hidden_dim)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--checkpoints", default="0,50,100,200,300,400")
    parser.add_argument("--router-kind", choices=("linear", "mlp"), default="linear")
    parser.add_argument("--expert-keep-prob", type=float, default=0.5)
    parser.add_argument("--selector-learning-rate", type=float, default=1e-3)
    parser.add_argument("--masked-prediction-weight", type=float,
                        default=TrainingConfig().masked_prediction_weight)
    parser.add_argument("--balance-weight", type=float, default=0.1)
    parser.add_argument("--confidence-start-epoch", type=int, default=100)
    parser.add_argument("--expert-confidence-fraction", type=float, default=0.5)
    parser.add_argument("--eval-repeats", type=int, default=3)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path,
                        default=Path("diagnostics/results/syn4_joint_router.json"))
    args = parser.parse_args()
    split = int(args.num_samples * args.train_ratio)
    if args.seed < 0 or args.num_samples < 2 or args.input_dim < 11 or args.batch_size < 1:
        parser.error("seed >= 0, num-samples >= 2, input-dim >= 11 and batch-size >= 1 required")
    if args.hidden_dim < 1 or not 0 < args.train_ratio < 1 or not 0 < split < args.num_samples:
        parser.error("hidden-dim >= 1 and train-ratio must leave both splits")
    if args.epochs < 1 or args.eval_repeats < 1 or args.selector_learning_rate <= 0:
        parser.error("epochs and eval-repeats >= 1; selector-learning-rate > 0 required")
    if not 0.5 <= args.expert_keep_prob < 1 or args.masked_prediction_weight < 0:
        parser.error("0.5 <= expert-keep-prob < 1 and masked-prediction-weight >= 0 required")
    if args.balance_weight < 0 or args.confidence_start_epoch < 0:
        parser.error("balance-weight and confidence-start-epoch must be nonnegative")
    if not 0 < args.expert_confidence_fraction <= 1:
        parser.error("0 < expert-confidence-fraction <= 1 required")
    try:
        checkpoints = {int(value.strip()) for value in args.checkpoints.split(",")}
    except ValueError:
        parser.error("checkpoints must be comma-separated integers")
    if any(epoch < 0 for epoch in checkpoints):
        parser.error("checkpoints must be nonnegative")
    args.checkpoints = sorted({epoch for epoch in checkpoints if epoch <= args.epochs} | {args.epochs})
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    return args


def predictor_step(
    x: torch.Tensor, y: torch.Tensor, extractor: FeatureExtractor,
    predictor: TaskPredictor, router: LearnedRouterMasks,
    optimizer: torch.optim.Optimizer, temperature: float, masked_weight: float,
) -> tuple[float, float, float]:
    with torch.no_grad():
        route_one_hot = F.gumbel_softmax(router.router(x), tau=temperature, hard=True, dim=-1)
        expert_masks = router.sampled_masks(len(x), temperature)
        selected_mask = (route_one_hot[:, :, None] * expert_masks).sum(dim=1)
    masked_x = apply_shuffle_replacement_mask(x, selected_mask)
    full_loss = F.cross_entropy(predictor(extractor(x)), y)
    masked_loss = F.cross_entropy(predictor(extractor(masked_x)), y)
    loss = full_loss + masked_weight * masked_loss
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return loss.item(), full_loss.item(), masked_loss.item()


def selector_step(
    x: torch.Tensor, y: torch.Tensor, extractor: FeatureExtractor,
    predictor: TaskPredictor, router: LearnedRouterMasks,
    optimizer: torch.optim.Optimizer, temperature: float,
    sparsity_weight: float, balance_weight: float,
    confidence_fraction: float, confident: bool,
) -> float:
    with frozen_parameters(extractor, predictor):
        route, keep = router.probabilities(x)
        sampled = router.sampled_masks(len(x), temperature)
        shuffled = x[torch.randperm(len(x), device=x.device)]
        masked = x[:, None, :] * sampled + shuffled[:, None, :] * (1 - sampled)
        logits = predictor(extractor(masked.reshape(-1, x.size(1)))).reshape(len(x), 2, 2)
        per_expert_ce = F.cross_entropy(
            logits.reshape(-1, 2), y[:, None].expand(-1, 2).reshape(-1), reduction="none"
        ).reshape(len(x), 2)
        expert_sparsity = keep.mean(dim=1)
        balance_loss = (route.mean(dim=0) - 0.5).square().sum()
        if confident:
            objective = (per_expert_ce + sparsity_weight * expert_sparsity).detach()
            router_loss = (route * objective).sum(dim=1).mean()
            chosen = route.detach().argmax(dim=1)
            expert_losses = []
            for expert in range(2):
                assigned = torch.nonzero(chosen == expert).flatten()
                if len(assigned) == 0:
                    continue
                retain = max(1, math.ceil(len(assigned) * confidence_fraction))
                confidence = route.detach()[assigned, expert]
                selected = assigned[confidence.topk(retain).indices]
                expert_losses.append(
                    per_expert_ce[selected, expert].mean()
                    + sparsity_weight * expert_sparsity[expert]
                )
            loss = router_loss + sum(expert_losses) / len(expert_losses) + balance_weight * balance_loss
        else:
            task_loss = (route * per_expert_ce).sum(dim=1).mean()
            sparse_loss = (route * expert_sparsity).sum(dim=1).mean()
            loss = task_loss + sparsity_weight * sparse_loss + balance_weight * balance_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
    optimizer.step()
    return loss.item()


def main() -> None:
    args = parse_args()
    config = TrainingConfig()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    features, labels, true_masks = generate_synthetic_data(
        num_samples=args.num_samples, input_dim=args.input_dim, dataset_type="Syn4"
    )
    split = int(args.num_samples * args.train_ratio)
    train_loader = DataLoader(
        TensorDataset(features[:split], labels[:split]),
        batch_size=args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(args.seed + 4),
    )
    test_loader = DataLoader(
        TensorDataset(features[split:], labels[split:], true_masks[split:]),
        batch_size=args.batch_size, shuffle=False,
    )
    union_for_evaluation = true_masks[:split].any(dim=0, keepdim=True).float().to(device)
    torch.manual_seed(args.seed + 2)
    extractor = FeatureExtractor(args.input_dim, args.hidden_dim).to(device)
    predictor = TaskPredictor(args.hidden_dim, 2).to(device)
    task_optimizer = torch.optim.Adam(
        list(extractor.parameters()) + list(predictor.parameters()),
        lr=config.task_learning_rate, weight_decay=1e-4,
    )
    torch.manual_seed(args.seed + 3)
    router = LearnedRouterMasks(
        args.input_dim, args.hidden_dim,
        torch.zeros(1, args.input_dim, device=device),
        args.expert_keep_prob, args.router_kind, "uniform",
    ).to(device)
    selector_optimizer = torch.optim.Adam(
        router.parameters(), lr=args.selector_learning_rate, betas=(0.5, 0.9)
    )
    report = {
        "settings": {**vars(args), "output": str(args.output), "training": asdict(config)},
        "notes": "No true masks or known switch coordinate enter training. Predictor uses ALTAS full + masked generated-input update; Critic is omitted. True masks and union are only for evaluation.",
        "snapshots": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Syn4 train={split}, test={len(features)-split}, device={device}", flush=True)

    def snapshot(epoch: int, temperature: float, train_metrics: dict[str, float] | None) -> None:
        scores = evaluate(
            extractor, predictor, router, test_loader, union_for_evaluation,
            args.eval_repeats, args.seed, config.sparsity_weight, device,
        )
        report["snapshots"][str(epoch)] = {
            "temperature": temperature, "train": train_metrics, "test": scores,
        }
        negative, positive = scores["learned"]["negative"], scores["learned"]["nonnegative"]
        print(
            f"epoch {epoch}: negative/positive exact="
            f"{negative['exact_mask_rate']:.3f}/{positive['exact_mask_rate']:.3f}, "
            f"selected={scores['learned']['all']['selected']:.2f}, "
            f"oracle-mask negative CE={scores['oracle']['negative']['ce']:.4f}, "
            f"test acc={scores['learned']['all']['accuracy']:.3f}",
            flush=True,
        )
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    temperature = config.temperature_start
    if 0 in args.checkpoints:
        snapshot(0, temperature, None)
    for epoch in range(1, args.epochs + 1):
        extractor.train()
        predictor.train()
        router.train()
        metric_sums = {"predictor": 0.0, "full": 0.0, "masked": 0.0, "selector": 0.0}
        for batch_x, batch_y in train_loader:
            x, y = batch_x.to(device), batch_y.to(device)
            task, full, masked = predictor_step(
                x, y, extractor, predictor, router, task_optimizer,
                temperature, args.masked_prediction_weight,
            )
            selection = selector_step(
                x, y, extractor, predictor, router, selector_optimizer,
                temperature, config.sparsity_weight, args.balance_weight,
                args.expert_confidence_fraction,
                bool(args.confidence_start_epoch and epoch > args.confidence_start_epoch),
            )
            metric_sums["predictor"] += task
            metric_sums["full"] += full
            metric_sums["masked"] += masked
            metric_sums["selector"] += selection
        temperature = max(config.temperature_min, temperature * config.temperature_decay)
        if epoch in args.checkpoints:
            train_metrics = {name: round(value / len(train_loader), 6)
                             for name, value in metric_sums.items()}
            snapshot(epoch, temperature, train_metrics)
    print(f"Saved report: {args.output}", flush=True)


if __name__ == "__main__":
    main()

"""Test two mask experts with an input-learned router on Syn4.

The router sees all input dimensions without a designated switch coordinate.
Flags independently remove oracle predictor training and union initialization
to test which privileged information the diagnostic still needs.
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
from models import FeatureExtractor, TaskPredictor


class LearnedRouterMasks(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, union: torch.Tensor,
                 keep_probability: float, router_kind: str = "mlp",
                 expert_init: str = "union") -> None:
        super().__init__()
        if router_kind == "linear":
            self.router = nn.Linear(input_dim, 2)
        else:
            self.router = nn.Sequential(
                nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 2),
            )
        keep_logit = math.log(keep_probability / (1 - keep_probability))
        logits = torch.zeros(2, input_dim, 2, device=union.device)
        if expert_init == "union":
            logits[:, :, 1] = torch.where(
                union[0].bool(), torch.full_like(union[0], keep_logit),
                torch.full_like(union[0], -keep_logit),
            )
        else:
            logits[:, :, 1] = keep_logit
        logits[:, :, 1] += 0.05 * torch.randn_like(logits[:, :, 1])
        self.mask_logits = nn.Parameter(logits)

    def probabilities(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        route = self.router(inputs).softmax(dim=-1)
        keep = self.mask_logits.softmax(dim=-1)[..., 1]
        return route, keep

    def sampled_masks(self, batch_size: int, temperature: float) -> torch.Tensor:
        logits = self.mask_logits.unsqueeze(0).expand(batch_size, -1, -1, -1)
        return F.gumbel_softmax(logits, tau=temperature, hard=True, dim=-1)[..., 1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-samples", type=int, default=DataConfig().num_samples)
    parser.add_argument("--input-dim", type=int, default=DataConfig().input_dim)
    parser.add_argument("--train-ratio", type=float, default=DataConfig().train_ratio)
    parser.add_argument("--batch-size", type=int, default=DataConfig().batch_size)
    parser.add_argument("--hidden-dim", type=int, default=ModelConfig().hidden_dim)
    parser.add_argument("--predictor-epochs", type=int, default=200)
    parser.add_argument("--predictor-mask-source", choices=("oracle", "random"), default="oracle")
    parser.add_argument("--random-mask-min-keep", type=float, default=0.1)
    parser.add_argument("--random-mask-max-keep", type=float, default=0.9)
    parser.add_argument("--selector-epochs", type=int, default=200)
    parser.add_argument("--checkpoints", default="0,50,100,200")
    parser.add_argument("--selector-learning-rate", type=float, default=1e-3)
    parser.add_argument("--router-kind", choices=("mlp", "linear"), default="mlp")
    parser.add_argument("--union-keep-prob", type=float, default=0.7)
    parser.add_argument("--expert-init", choices=("union", "uniform"), default="union")
    parser.add_argument("--balance-weight", type=float, default=0.1)
    parser.add_argument("--confidence-start-epoch", type=int, default=0)
    parser.add_argument("--expert-confidence-fraction", type=float, default=0.5)
    parser.add_argument("--eval-repeats", type=int, default=3)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path,
                        default=Path("diagnostics/results/syn4_learned_router.json"))
    args = parser.parse_args()
    split = int(args.num_samples * args.train_ratio)
    if args.seed < 0 or args.num_samples < 2 or args.input_dim < 11 or args.batch_size < 1:
        parser.error("seed >= 0, num-samples >= 2, input-dim >= 11, batch-size >= 1 required")
    if args.hidden_dim < 1 or not 0 < args.train_ratio < 1 or not 0 < split < args.num_samples:
        parser.error("hidden-dim >= 1 and train-ratio must leave both splits")
    if min(args.predictor_epochs, args.selector_epochs, args.eval_repeats) < 1:
        parser.error("epoch counts and eval-repeats must be positive")
    if args.selector_learning_rate <= 0 or not 0.5 <= args.union_keep_prob < 1 or args.balance_weight < 0:
        parser.error("selector-learning-rate > 0, 0.5 <= union-keep-prob < 1, balance-weight >= 0")
    if not 0 < args.random_mask_min_keep <= args.random_mask_max_keep <= 1:
        parser.error("random-mask keep probabilities must satisfy 0 < min <= max <= 1")
    if args.confidence_start_epoch < 0 or not 0 < args.expert_confidence_fraction <= 1:
        parser.error("confidence-start-epoch >= 0 and 0 < expert-confidence-fraction <= 1 required")
    try:
        checkpoints = {int(value.strip()) for value in args.checkpoints.split(",")}
    except ValueError:
        parser.error("checkpoints must be comma-separated integers")
    if any(epoch < 0 for epoch in checkpoints):
        parser.error("checkpoints must be nonnegative")
    args.checkpoints = sorted({epoch for epoch in checkpoints if epoch <= args.selector_epochs} | {args.selector_epochs})
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    return args


def evaluate(
    extractor: FeatureExtractor, predictor: TaskPredictor, model: LearnedRouterMasks,
    loader: DataLoader, union: torch.Tensor, repeats: int, seed: int,
    sparsity_weight: float, device: torch.device,
) -> dict[str, object]:
    groups = ("all", "negative", "nonnegative")
    conditions = ("oracle", "union", "learned")
    totals = {
        condition: {group: {"count": 0, "ce": 0.0, "correct": 0,
                            "selected": 0.0, "exact": 0,
                            "feature_counts": torch.zeros(11, device=device)}
                    for group in groups}
        for condition in conditions
    }
    route_counts = {group: torch.zeros(2, device=device) for group in groups}
    route_probability_sums = {group: torch.zeros(2, device=device) for group in groups}
    original_modes = (extractor.training, predictor.training, model.training)
    cuda_devices = [torch.cuda.current_device()] if device.type == "cuda" else []
    try:
        extractor.eval()
        predictor.eval()
        model.eval()
        with torch.no_grad(), torch.random.fork_rng(devices=cuda_devices):
            for repeat in range(repeats):
                torch.manual_seed(seed + 10_000 + repeat)
                for batch_x, batch_y, batch_true in loader:
                    x, y, true = batch_x.to(device), batch_y.to(device), batch_true.to(device)
                    route_prob, keep = model.probabilities(x)
                    chosen = route_prob.argmax(dim=-1)
                    expert_masks = (keep >= 0.5).float()
                    learned = expert_masks[chosen]
                    masks = {"oracle": true, "union": union.expand_as(true), "learned": learned}
                    shuffled = x[torch.randperm(len(x), device=device)]
                    members = {"all": torch.ones_like(y, dtype=torch.bool),
                               "negative": x[:, 10] < 0, "nonnegative": x[:, 10] >= 0}
                    for group, take in members.items():
                        route_counts[group] += F.one_hot(chosen[take], 2).sum(0)
                        route_probability_sums[group] += route_prob[take].sum(0)
                    for condition, mask in masks.items():
                        logits = predictor(extractor(x * mask + shuffled * (1 - mask)))
                        losses = F.cross_entropy(logits, y, reduction="none")
                        correct = logits.argmax(dim=-1).eq(y)
                        exact = mask.bool().eq(true.bool()).all(dim=-1)
                        for group, take in members.items():
                            item = totals[condition][group]
                            item["count"] += int(take.sum().item())
                            item["ce"] += losses[take].sum().item()
                            item["correct"] += int(correct[take].sum().item())
                            item["selected"] += mask[take].sum().item()
                            item["exact"] += int(exact[take].sum().item())
                            item["feature_counts"] += mask[take, :11].sum(0)
    finally:
        extractor.train(original_modes[0])
        predictor.train(original_modes[1])
        model.train(original_modes[2])
    result: dict[str, object] = {condition: {} for condition in conditions}
    for condition in conditions:
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
    result["routing"] = {
        group: {"hard_expert_fraction": [round(v / totals["learned"][group]["count"], 6)
                                        for v in route_counts[group].tolist()],
                "mean_expert_probability": [round(v / totals["learned"][group]["count"], 6)
                                            for v in route_probability_sums[group].tolist()]}
        for group in groups
    }
    result["expert_keep_probabilities_0_to_10"] = [
        [round(v, 6) for v in row] for row in model.mask_logits.softmax(-1)[..., 1][:, :11].detach().tolist()
    ]
    result["expert_deterministic_masks_0_to_10"] = [
        [int(v >= 0.5) for v in row] for row in model.mask_logits.softmax(-1)[..., 1][:, :11].detach().tolist()
    ]
    result["oracle_minus_union_objective"] = round(
        result["oracle"]["all"]["objective"] - result["union"]["all"]["objective"], 6
    )
    if isinstance(model.router, nn.Linear):
        differences = (model.router.weight[0] - model.router.weight[1]).detach().abs()
        top = differences.topk(min(10, len(differences))).indices.tolist()
        result["linear_router_top_absolute_weight_dimensions"] = top
        result["linear_router_feature_10_absolute_weight_rank"] = int(
            (differences > differences[10]).sum().item() + 1
        )
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
    predictor_data = (
        TensorDataset(features[:split], labels[:split], true_masks[:split])
        if args.predictor_mask_source == "oracle"
        else TensorDataset(features[:split], labels[:split])
    )
    selector_data = TensorDataset(features[:split], labels[:split])
    test_loader = DataLoader(
        TensorDataset(features[split:], labels[split:], true_masks[split:]),
        batch_size=args.batch_size, shuffle=False,
    )
    union = true_masks[:split].any(dim=0, keepdim=True).float().to(device)
    torch.manual_seed(args.seed + 2)
    extractor = FeatureExtractor(args.input_dim, args.hidden_dim).to(device)
    predictor = TaskPredictor(args.hidden_dim, 2).to(device)
    predictor_optimizer = torch.optim.Adam(
        list(extractor.parameters()) + list(predictor.parameters()),
        lr=config.task_learning_rate, weight_decay=1e-4,
    )
    predictor_loader = DataLoader(
        predictor_data, batch_size=args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(args.seed + 1),
    )
    print(f"Syn4 train={split}, test={len(features)-split}, device={device}", flush=True)
    for epoch in range(1, args.predictor_epochs + 1):
        extractor.train()
        predictor.train()
        loss_total = 0.0
        for batch in predictor_loader:
            x, y = batch[0].to(device), batch[1].to(device)
            if args.predictor_mask_source == "random":
                keep_rates = args.random_mask_min_keep + (
                    args.random_mask_max_keep - args.random_mask_min_keep
                ) * torch.rand(len(x), 1, device=device)
                mask = (torch.rand_like(x) < keep_rates).float()
            else:
                mask = batch[2].to(device)
            logits = predictor(extractor(apply_shuffle_replacement_mask(x, mask)))
            loss = F.cross_entropy(logits, y)
            predictor_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            predictor_optimizer.step()
            loss_total += loss.item()
        if epoch == 1 or epoch % 20 == 0 or epoch == args.predictor_epochs:
            print(f"predictor epoch {epoch}: loss={loss_total/len(predictor_loader):.4f}", flush=True)

    extractor.eval()
    predictor.eval()
    for module in (extractor, predictor):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    torch.manual_seed(args.seed + 3)
    initialization_reference = (
        union if args.expert_init == "union"
        else torch.zeros(1, args.input_dim, device=device)
    )
    model = LearnedRouterMasks(
        args.input_dim, args.hidden_dim, initialization_reference, args.union_keep_prob,
        args.router_kind, args.expert_init,
    ).to(device)
    selector_optimizer = torch.optim.Adam(model.parameters(), lr=args.selector_learning_rate,
                                          betas=(0.5, 0.9))
    selector_loader = DataLoader(
        selector_data, batch_size=args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(args.seed + 4),
    )
    report = {
        "settings": {**vars(args), "output": str(args.output), "training": asdict(config)},
        "notes": "The router sees x only and no designated switch coordinate. Oracle masks enter training only if predictor-mask-source=oracle or expert-init=union. Routing groups and true masks otherwise serve evaluation only.",
        "snapshots": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def snapshot(epoch: int) -> None:
        scores = evaluate(extractor, predictor, model, test_loader, union,
                          args.eval_repeats, args.seed, config.sparsity_weight, device)
        report["snapshots"][str(epoch)] = scores
        n, p = scores["learned"]["negative"], scores["learned"]["nonnegative"]
        rn = scores["routing"]["negative"]["hard_expert_fraction"]
        rp = scores["routing"]["nonnegative"]["hard_expert_fraction"]
        print(
            f"selector epoch {epoch}: negative/positive exact="
            f"{n['exact_mask_rate']:.3f}/{p['exact_mask_rate']:.3f}, "
            f"selected={scores['learned']['all']['selected']:.2f}, "
            f"router expert0 negative/positive={rn[0]:.3f}/{rp[0]:.3f}, "
            f"test objective={scores['learned']['all']['objective']:.5f}",
            flush=True,
        )
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if 0 in args.checkpoints:
        snapshot(0)
    temperature = config.temperature_start
    for epoch in range(1, args.selector_epochs + 1):
        model.train()
        loss_total = 0.0
        for batch_x, batch_y in selector_loader:
            x, y = batch_x.to(device), batch_y.to(device)
            route, keep = model.probabilities(x)
            sampled = model.sampled_masks(len(x), temperature)
            shuffled = x[torch.randperm(len(x), device=device)]
            masked = x[:, None, :] * sampled + shuffled[:, None, :] * (1 - sampled)
            logits = predictor(extractor(masked.reshape(-1, args.input_dim))).reshape(len(x), 2, 2)
            per_expert_ce = F.cross_entropy(
                logits.reshape(-1, 2), y[:, None].expand(-1, 2).reshape(-1), reduction="none"
            ).reshape(len(x), 2)
            expert_sparsity = keep.mean(dim=1)
            balance_loss = (route.mean(dim=0) - 0.5).square().sum()
            if args.confidence_start_epoch and epoch > args.confidence_start_epoch:
                # Router still learns from every example. Expert updates use only
                # high-confidence, x-only assignments, limiting cross-branch leakage.
                detached_objectives = (per_expert_ce + config.sparsity_weight * expert_sparsity).detach()
                router_loss = (route * detached_objectives).sum(dim=1).mean()
                chosen = route.detach().argmax(dim=1)
                expert_losses = []
                for expert in range(2):
                    assigned = torch.nonzero(chosen == expert).flatten()
                    if len(assigned) == 0:
                        continue
                    retain = max(1, math.ceil(len(assigned) * args.expert_confidence_fraction))
                    confidence = route.detach()[assigned, expert]
                    selected = assigned[confidence.topk(retain).indices]
                    expert_losses.append(
                        per_expert_ce[selected, expert].mean()
                        + config.sparsity_weight * expert_sparsity[expert]
                    )
                loss = router_loss + sum(expert_losses) / len(expert_losses) + args.balance_weight * balance_loss
            else:
                task_loss = (route * per_expert_ce).sum(dim=1).mean()
                sparsity_loss = (route * expert_sparsity).sum(dim=1).mean()
                loss = task_loss + config.sparsity_weight * sparsity_loss + args.balance_weight * balance_loss
            selector_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            selector_optimizer.step()
            loss_total += loss.item()
        temperature = max(config.temperature_min, temperature * config.temperature_decay)
        if epoch in args.checkpoints:
            print(f"selector epoch {epoch}: train loss={loss_total/len(selector_loader):.4f}", flush=True)
            snapshot(epoch)
    print(f"Saved report: {args.output}", flush=True)


if __name__ == "__main__":
    main()

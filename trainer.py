"""ALTAS 对抗训练器。

训练器封装特征掩码生成器、共享特征提取器、隐空间判别器和任务预测器的
三条更新路径，但不负责跨 epoch 的循环或结果持久化。
"""

from collections.abc import Iterable

import torch
from torch import autograd, nn

from masking import apply_shuffle_replacement_mask
from models import FeatureExtractor, FeatureMaskGenerator, LatentCritic, TaskPredictor


class ALTASTrainer:
    """协调 ALTAS 四个组件的优化过程。

    每个训练步依次执行若干次判别器更新、一次提取器/预测器联合更新，以及
    一次掩码生成器更新。各分支的梯度边界与原始实现保持一致。
    """

    METRIC_NAMES = (
        "critic_loss",
        "predictor_loss",
        "generator_loss",
        "generator_adversarial_loss",
        "generator_sparsity_loss",
        "generator_prediction_loss",
        "wasserstein_distance",
        "retention_rate",
        "masked_accuracy",
        "full_accuracy",
    )

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        device: torch.device,
        hidden_dim: int = 64,
        generator_learning_rate: float = 1e-5,
        task_learning_rate: float = 1e-4,
        gradient_penalty_weight: float = 10.0,
        adversarial_weight: float = 0.1,
        prediction_weight: float = 1.0,
        sparsity_weight: float = 1.0,
        critic_steps: int = 3,
        masked_prediction_weight: float = 1.0,
    ) -> None:
        if critic_steps < 1:
            raise ValueError("critic_steps 必须大于或等于 1")

        self.device = device
        self.adversarial_weight = adversarial_weight
        self.prediction_weight = prediction_weight
        self.sparsity_weight = sparsity_weight
        self.gradient_penalty_weight = gradient_penalty_weight
        self.critic_steps = critic_steps
        self.masked_prediction_weight = masked_prediction_weight

        self.generator = FeatureMaskGenerator(input_dim, hidden_dim).to(device)
        self.extractor = FeatureExtractor(input_dim, hidden_dim).to(device)
        self.critic = LatentCritic(hidden_dim).to(device)
        self.predictor = TaskPredictor(hidden_dim, num_classes).to(device)

        self.generator_optimizer = torch.optim.Adam(
            self.generator.parameters(),
            lr=generator_learning_rate,
            betas=(0.5, 0.9),
        )
        task_parameters = list(self.extractor.parameters()) + list(self.predictor.parameters())
        self.task_optimizer = torch.optim.Adam(
            task_parameters,
            lr=task_learning_rate,
            weight_decay=1e-4,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=task_learning_rate
        )
        self.classification_loss = nn.CrossEntropyLoss()

    def modules(self) -> Iterable[nn.Module]:
        """返回由训练器管理的全部网络模块。"""
        return (self.generator, self.extractor, self.critic, self.predictor)

    def train(self) -> None:
        """将全部网络切换到训练模式。"""
        for module in self.modules():
            module.train()

    def eval(self) -> None:
        """将全部网络切换到评估模式。"""
        for module in self.modules():
            module.eval()

    def _compute_gradient_penalty(
        self,
        real_latent: torch.Tensor,
        masked_latent: torch.Tensor,
    ) -> torch.Tensor:
        """计算 WGAN-GP 隐空间插值样本的梯度惩罚。"""
        interpolation_weight = torch.rand(
            real_latent.size(0), 1, device=self.device
        ).expand_as(real_latent)
        interpolated_latent = (
            interpolation_weight * real_latent
            + (1.0 - interpolation_weight) * masked_latent
        ).requires_grad_(True)
        critic_scores = self.critic(interpolated_latent)
        gradient_outputs = torch.ones(
            real_latent.size(0), 1, device=self.device
        )
        gradients = autograd.grad(
            outputs=critic_scores,
            inputs=interpolated_latent,
            grad_outputs=gradient_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradient_norm = gradients.flatten(start_dim=1).norm(2, dim=1)
        return ((gradient_norm - 1.0) ** 2).mean()

    def _update_predictor(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        detached_mask: torch.Tensor,
    ) -> float:
        """使用完整输入和掩码输入联合更新特征提取器与预测器。"""
        masked_inputs = apply_shuffle_replacement_mask(inputs, detached_mask)
        full_latent = self.extractor(inputs)
        masked_latent = self.extractor(masked_inputs)
        full_loss = self.classification_loss(self.predictor(full_latent), labels)
        masked_loss = self.classification_loss(self.predictor(masked_latent), labels)
        predictor_loss = full_loss + self.masked_prediction_weight * masked_loss

        self.task_optimizer.zero_grad()
        predictor_loss.backward()
        self.task_optimizer.step()
        return predictor_loss.item()

    def _update_critic(
        self,
        inputs: torch.Tensor,
        detached_mask: torch.Tensor,
    ) -> tuple[float, float]:
        """以完整和掩码输入的隐表示更新 WGAN-GP 判别器。"""
        masked_inputs = apply_shuffle_replacement_mask(inputs, detached_mask)
        real_latent = self.extractor(inputs).detach()
        masked_latent = self.extractor(masked_inputs).detach()

        real_scores = self.critic(real_latent)
        masked_scores = self.critic(masked_latent)
        gradient_penalty = self._compute_gradient_penalty(real_latent, masked_latent)
        critic_loss = (
            masked_scores.mean()
            - real_scores.mean()
            + self.gradient_penalty_weight * gradient_penalty
        )
        wasserstein_distance = (real_scores.mean() - masked_scores.mean()).item()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        return critic_loss.item(), wasserstein_distance

    def _update_generator(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        temperature: float,
    ) -> dict[str, float]:
        """联合对抗、预测和稀疏目标更新特征掩码生成器。"""
        retention_probabilities, mask = self.generator(
            inputs, temperature=temperature, hard=True
        )
        masked_inputs = apply_shuffle_replacement_mask(inputs, mask)
        masked_latent = self.extractor(masked_inputs)

        adversarial_loss = -self.critic(masked_latent).mean()
        prediction_loss = self.classification_loss(self.predictor(masked_latent), labels)
        sparsity_loss = retention_probabilities.mean()
        generator_loss = (
            self.adversarial_weight * adversarial_loss
            + self.prediction_weight * prediction_loss
            + self.sparsity_weight * sparsity_loss
        )

        self.generator_optimizer.zero_grad()
        generator_loss.backward()
        self.generator_optimizer.step()

        return {
            "generator_loss": generator_loss.item(),
            "generator_adversarial_loss": adversarial_loss.item(),
            "generator_prediction_loss": prediction_loss.item(),
            "generator_sparsity_loss": sparsity_loss.item(),
            "retention_rate": sparsity_loss.item(),
        }

    def _compute_accuracy(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
        use_full_input: bool,
    ) -> float:
        """计算完整输入或掩码输入上的批准确率。"""
        model_inputs = (
            inputs
            if use_full_input
            else apply_shuffle_replacement_mask(inputs, mask)
        )
        predictions = self.predictor(self.extractor(model_inputs)).argmax(dim=1)
        return (predictions == labels).float().mean().item()

    def train_step(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        temperature: float = 1.0,
    ) -> dict[str, float]:
        """执行一个批次的完整三阶段更新并返回训练指标。"""
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)

        critic_loss_sum = 0.0
        wasserstein_distance_sum = 0.0
        for _ in range(self.critic_steps):
            with torch.no_grad():
                _, critic_mask = self.generator(
                    inputs, temperature=temperature, hard=True
                )
            critic_loss, wasserstein_distance = self._update_critic(
                inputs, critic_mask
            )
            critic_loss_sum += critic_loss
            wasserstein_distance_sum += wasserstein_distance

        with torch.no_grad():
            _, predictor_mask = self.generator(
                inputs, temperature=temperature, hard=True
            )
        predictor_loss = self._update_predictor(inputs, labels, predictor_mask)
        generator_metrics = self._update_generator(inputs, labels, temperature)

        with torch.no_grad():
            _, evaluation_mask = self.generator(
                inputs, temperature=temperature, hard=True
            )
            masked_accuracy = self._compute_accuracy(
                inputs, labels, evaluation_mask, use_full_input=False
            )
            full_accuracy = self._compute_accuracy(
                inputs, labels, evaluation_mask, use_full_input=True
            )

        return {
            "critic_loss": critic_loss_sum / self.critic_steps,
            "predictor_loss": predictor_loss,
            "wasserstein_distance": wasserstein_distance_sum / self.critic_steps,
            **generator_metrics,
            "masked_accuracy": masked_accuracy,
            "full_accuracy": full_accuracy,
        }

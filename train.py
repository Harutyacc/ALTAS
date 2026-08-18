"""ALTAS 训练器：Generator(Gen) / Extractor(Ext) / Critic(C) / predictor(P) 的博弈封装。"""

from collections import defaultdict
from typing import Dict

import torch
import torch.nn as nn
from torch import autograd

from model import Critic, Extractor, Generator, predictor


class ALTASTrainer:
    """Gen / C / P 三方博弈训练器。每次 train_step 包含 n_critic 次 C 更新和 1 次 Gen 更新。"""
    HISTORY_KEYS = (
        "loss_c",
        "loss_p",
        "loss_gen",
        "loss_gen_adv",
        "loss_gen_l1",
        "loss_gen_p",
        "w_distance",
        "retention_rate",
        "mask_x_acc",
        "full_x_acc",
    )

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        device: torch.device,
        lr_gen: float = 1e-5,
        lr_ep: float = 1e-4,
        lambda_gp: float = 10.0,
        alpha: float = 0.1,
        beta: float = 1.0,
        gamma: float = 1.0,
        n_critic: int = 3,
        p_mask_weight: float = 1.0,
    ):
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.lambda_gp = lambda_gp
        self.n_critic = n_critic
        self.p_mask_weight = p_mask_weight

        self.gen = Generator(input_dim).to(device)
        self.ext = Extractor(input_dim).to(device)
        self.critic = Critic().to(device)
        self.predictor = predictor(hidden_dim=64, num_classes=num_classes).to(device)

        self.opt_gen = torch.optim.Adam(self.gen.parameters(), lr=lr_gen, betas=(0.5, 0.9))
        ep_params = list(self.ext.parameters()) + list(self.predictor.parameters())
        self.opt_ep = torch.optim.Adam(ep_params, lr=lr_ep, weight_decay=1e-4)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=lr_ep)

        self.criterion_p = nn.CrossEntropyLoss()

    def modules(self):
        return [self.gen, self.ext, self.critic, self.predictor]

    def set_train(self):
        for m in self.modules():
            m.train()

    def set_eval(self):
        for m in self.modules():
            m.eval()

    def compute_latent_gradient_penalty(self, h_real, h_fake):
        """WGAN-GP：在 h_real / h_fake 之间线性插值并惩罚梯度范数偏离 1。"""
        alpha = torch.rand(h_real.size(0), 1).to(self.device).expand_as(h_real)
        interpolates = (alpha * h_real + (1 - alpha) * h_fake).requires_grad_(True)
        d_interpolates = self.critic(interpolates)
        grad_outputs = torch.ones(h_real.size(0), 1).to(self.device)
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        return ((gradients.view(gradients.size(0), -1).norm(2, dim=1) - 1) ** 2).mean()

    def _train_predictor_branch(self, x, y, mask_detached):
        """轨迹 A: 用完整特征 + 掩码特征共同监督 Ext 和 P。

        loss_p = loss_p_full + p_mask_weight * loss_p_mask
        ``p_mask_weight`` 控制稀疏输入下 CE 的权重: 越大越强调"被稀疏后还能学"。
        掩码输入直接 ``x * mask``, 被掩位置退化为 0。
        """
        x_mask = x * mask_detached
        h_full = self.ext(x)
        h_mask = self.ext(x_mask)
        loss_p_full = self.criterion_p(self.predictor(h_full), y)
        loss_p_mask = self.criterion_p(self.predictor(h_mask), y)
        loss_p = loss_p_full + self.p_mask_weight * loss_p_mask

        self.opt_ep.zero_grad()
        loss_p.backward()
        self.opt_ep.step()
        return loss_p.item()

    def _train_critic_branch(self, x, mask_detached):
        """轨迹 B: 训练 Critic。Critic 看到的"h_fake"由被掩输入得到 (x * mask)。"""
        x_mask = x * mask_detached
        h_real = self.ext(x).detach()
        h_fake = self.ext(x_mask).detach()

        d_real = self.critic(h_real)
        d_fake = self.critic(h_fake)
        gp = self.compute_latent_gradient_penalty(h_real, h_fake)
        loss_c = d_fake.mean() - d_real.mean() + self.lambda_gp * gp

        w_distance = (d_real.mean() - d_fake.mean()).item()
        self.opt_critic.zero_grad()
        loss_c.backward()
        self.opt_critic.step()

        return loss_c.item(), w_distance

    def _train_generator_branch(self, x, y, tau):
        """轨迹 C: 训练 Gen, 使 h_fake 既能骗过 Critic 又能用于 P, 并对 pi 加 L1 期望惩罚。"""
        pi, mask = self.gen(x, tau=tau, hard=True)
        x_mask = x * mask
        h_g = self.ext(x_mask)

        gen_loss_adv = -self.critic(h_g).mean()
        gen_loss_p = self.criterion_p(self.predictor(h_g), y)
        gen_loss_l1 = pi.mean()
        loss_gen = self.alpha * gen_loss_adv + self.beta * gen_loss_p + self.gamma * gen_loss_l1

        self.opt_gen.zero_grad()
        loss_gen.backward()
        self.opt_gen.step()

        return {
            "loss_gen": loss_gen.item(),
            "loss_gen_adv": gen_loss_adv.item(),
            "loss_gen_p": gen_loss_p.item(),
            "loss_gen_l1": gen_loss_l1.item(),
            "retention_rate": gen_loss_l1.item(),
        }

    def _eval_acc(self, x, y, mask, use_full=False):
        h = self.ext(x) if use_full else self.ext(x * mask)
        return (self.predictor(h).argmax(dim=1) == y).float().mean().item()

    def train_step(self, x, y, tau: float = 1.0) -> Dict[str, float]:
        """完整的一次 train_step：包含 n_critic 次 Critic 更新和 1 次 Gen 更新。"""
        x, y = x.to(self.device), y.to(self.device)

        loss_c_acc, loss_p_acc, w_dist_acc = 0.0, 0.0, 0.0
        
        # 1. 独立更新 Critic，迭代 n_critic 次
        for _ in range(self.n_critic):
            with torch.no_grad():
                _, mask = self.gen(x, tau=tau, hard=True)
            
            loss_c, w_dist = self._train_critic_branch(x, mask)
            loss_c_acc += loss_c
            w_dist_acc += w_dist
            
        # 2. 更新 Predictor & Extractor (EP 链路)
        with torch.no_grad():
            _, mask_ep = self.gen(x, tau=tau, hard=True)
        loss_p_acc += self._train_predictor_branch(x, y, mask_ep)
        
        # 3. 更新 Generator
        gen_metrics = self._train_generator_branch(x, y, tau)
        
        with torch.no_grad():
            _, mask_eval = self.gen(x, tau=tau, hard=True)
            mask_x_acc = self._eval_acc(x, y, mask_eval, use_full=False)
            full_x_acc = self._eval_acc(x, y, mask_eval, use_full=True)

        return {
            "loss_c": loss_c_acc / self.n_critic,
            "loss_p": loss_p_acc / self.n_critic,
            "w_distance": w_dist_acc / self.n_critic,
            **gen_metrics,
            "mask_x_acc": mask_x_acc,
            "full_x_acc": full_x_acc,
        }

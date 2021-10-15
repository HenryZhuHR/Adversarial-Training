from typing import List
import torch
import numpy as np
from torch import nn
from torch import optim
from torch import Tensor
from .base_attack import BaseAttack


class CW(BaseAttack):
    """
    CW attack
    ===

    Paper: [Towards evaluating the robustness of neural networks](https://arxiv.org/pdf/1608.04644.pdf) Carlini, N., & Wagner, D. (2017, May).
    Code : [carlini/nn_robust_attacks](https://github.com/carlini/nn_robust_attacks)

    github torchattacks
    """

    def __init__(self,
                 model,
                 device='cpu',
                 attack_params: List[int] = []
                 ) -> None:
        super().__init__(model, device=device)
        self.attack_name = 'CW'
        self.ATTACK_PARAMETERS.update(
            c=float(1e-4),  # confidence
            kappa=float(),  # thresd
            max_steps=int(1000),
            lr=float(1e-3),
            attack_mode=str('no_target')  # ['no_target', 'target']

        )
        super().parse_params(attack_params)

        self.c = float(self.ATTACK_PARAMETERS['c'])
        self.kappa = float(self.ATTACK_PARAMETERS['kappa'])
        self.max_steps = int(self.ATTACK_PARAMETERS['max_steps'])
        self.lr = float(self.ATTACK_PARAMETERS['lr'])
        self.attack_mode = str(self.ATTACK_PARAMETERS['attack_mode'])

    def attack(self, images: Tensor, labels: Tensor) -> Tensor:
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)


        # w = torch.zeros_like(images).detach() # Requires 2x times
        w = torch.atanh(images).detach()
        w.requires_grad = True

        best_adv_images = images.clone().detach()
        best_L2 = 1e10*torch.ones((len(images))).to(self.device)
        prev_cost = 1e10
        dim = len(images.shape)

        mse_loss = nn.MSELoss(reduction='none')
        flatten = nn.Flatten()

        optimizer = optim.Adam([w], lr=self.lr)

        for step in range(self.max_steps):
            # x+\delta = 1/2*(tanh(\omega)+1) 
            adv_images = 1/2*(torch.tanh(w)+1)

            # in paper: page-9 VI. OUR THREE ATTACKS A. Our L2 Attack
            current_L2:Tensor = mse_loss(flatten(adv_images),
                                  flatten(images)).sum(dim=1)
            L2_loss = current_L2.sum()

            outputs:Tensor = self.model(adv_images)
            f_loss = self.f(outputs, labels).sum()

            cost: Tensor = L2_loss + self.c*f_loss  # minimize term in paper

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            # Update adversarial images
            _, pre = torch.max(outputs.detach(), 1)
            correct = (pre == labels).float()

            mask = (1-correct)*(best_L2 > current_L2.detach())
            best_L2 = mask*current_L2.detach() + (1-mask)*best_L2

            mask = mask.view([-1]+[1]*(dim-1))
            best_adv_images = mask*adv_images.detach() + (1-mask)*best_adv_images

            # Early stop when loss does not converge.
            if step % (self.max_steps//10) == 0:
                if cost.item() > prev_cost:
                    return best_adv_images
                prev_cost = cost.item()

        return best_adv_images


    def attack(self, images: Tensor, labels: Tensor) -> Tensor:
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)


        # w = torch.zeros_like(images).detach() # Requires 2x times
        w = torch.atanh(images).detach()
        w.requires_grad = True

        best_adv_images = images.clone().detach()
        best_L2 = 1e10*torch.ones((len(images))).to(self.device)
        prev_cost = 1e10
        dim = len(images.shape)

        mse_loss = nn.MSELoss(reduction='none')
        flatten = nn.Flatten()

        optimizer = optim.Adam([w], lr=self.lr)

        for step in range(self.max_steps):
            # x+\delta = 1/2*(tanh(\omega)+1) 
            adv_images = 1/2*(torch.tanh(w)+1)

            # in paper: page-9 VI. OUR THREE ATTACKS A. Our L2 Attack
            current_L2:Tensor = mse_loss(flatten(adv_images),
                                  flatten(images)).sum(dim=1)
            L2_loss = current_L2.sum()

            outputs:Tensor = self.model(adv_images)
            f_loss = self.f(outputs, labels).sum()

            cost: Tensor = L2_loss + self.c*f_loss  # minimize term in paper

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            # Update adversarial images
            _, pre = torch.max(outputs.detach(), 1)
            correct = (pre == labels).float()

            mask = (1-correct)*(best_L2 > current_L2.detach())
            best_L2 = mask*current_L2.detach() + (1-mask)*best_L2

            mask = mask.view([-1]+[1]*(dim-1))
            best_adv_images = mask*adv_images.detach() + (1-mask)*best_adv_images

            # Early stop when loss does not converge.
            if step % (self.max_steps//10) == 0:
                if cost.item() > prev_cost:
                    return best_adv_images
                prev_cost = cost.item()

        return best_adv_images

        
    # f-function in the paper
    def f(self, outputs, labels):
        one_hot_labels = torch.eye(len(outputs[0]))[labels].to(self.device)

        i, _ = torch.max((1-one_hot_labels)*outputs, dim=1)
        j = torch.masked_select(outputs, one_hot_labels.bool())

        return torch.clamp((j-i), min=-self.kappa)
        

        if self._targeted:
            return torch.clamp((i-j), min=-self.kappa)
        else:
            return torch.clamp((j-i), min=-self.kappa)

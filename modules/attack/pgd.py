from typing import List
import torch
import numpy as np
from torch import nn
from torch import optim
from torch import Tensor

from .base_attack import BaseAttack

class PGD(BaseAttack):
    """
    PGD attack
    ===

    Paper: 
    Code : 
    """

    def __init__(self,
                 model,
                 device='cpu',
                 attack_params: List[int] = []
                 ) -> None:
        super().__init__(model, device=device)
        self.ATTACK_PARAMETERS.update(
            epsilon=float(1/255),  # confidence
            alpha=float(2/255),  # thresd
            num_steps=int(20)
        )
        super().parse_params(attack_params)

        self.epsilon = float(self.ATTACK_PARAMETERS['epsilon'])
        self.alpha = float(self.ATTACK_PARAMETERS['alpha'])
        self.num_steps = int(self.ATTACK_PARAMETERS['num_steps'])

    def attack(self, images: Tensor, labels: Tensor) -> Tensor:
        X=Tensor(images.detach().cpu().numpy()).to(self.device)
        x_adv=X
        x_adv.requires_grad = True

        loss_function = nn.CrossEntropyLoss()

        for i in range(self.num_steps):
            outputs = self.model(x_adv)
            self.model.zero_grad()
            loss: Tensor = loss_function(outputs, labels)
            loss.backward()

            # compute
            eta = self.alpha*x_adv.grad.detach().sign()
            x_adv = X+eta
            eta = torch.clamp(x_adv.detach()-X.detach(), min=-self.epsilon, max=self.epsilon)
            x_adv = torch.clamp(X+eta, min=0, max=1).detach()
            x_adv.requires_grad_()
            x_adv.retain_grad()

        return x_adv


import torch
import numpy as np
from torch import nn
from torch import optim
from torch import Tensor
from .base_attack import BaseAttack

class FGSM(BaseAttack):
    attack_name = 'FGSM'
    ATTACK_PARAMETERS = [
        'epsilon'
    ]

    def __init__(self,
                 model,
                 device='cpu',
                 **kwargs  # {key:word} like attack argurement
                 ) -> None:
        super().__init__(model, device=device, **kwargs)
        self.attack_params = super().parse_params(kwargs=kwargs)
        # Init Attack Parameters
        self.epsilon=self.attack_params['epsilon']

    def attack(self, input: Tensor, label: Tensor,
               clip_min: float = -1.0, clip_max: float = 1.0) -> Tensor:
        inputArray = input.detach().cpu().numpy()
        X_fgsm=Tensor(inputArray).to(self.device)

        X_fgsm.requires_grad = True

        optimizer = optim.SGD([X_fgsm], lr=1e-3)
        optimizer.zero_grad()

        loss_function = nn.CrossEntropyLoss()
        loss: Tensor = loss_function(self.model(X_fgsm), label)
        loss.backward()

        d = self.epsilon * X_fgsm.grad.data.sign()

        # 生成对抗样本
        x_adv = X_fgsm + d

        if clip_max == None and clip_min == None:
            clip_max = np.inf
            clip_min = -np.inf

        x_adv = torch.clamp(x_adv, clip_min, clip_max)
        x_adv.to(self.device)

        return x_adv


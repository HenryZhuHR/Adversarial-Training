import torch
import numpy as np
from torch import nn
from torch import optim
from torch import Tensor
from .base_attack import BaseAttack

class FGM(BaseAttack):
    attack_name = 'FGM'
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
        X_fgm=Tensor(inputArray).to(self.device)

        X_fgm.requires_grad = True

        optimizer = optim.SGD([X_fgm], lr=1e-3)
        optimizer.zero_grad()

        loss_function = nn.CrossEntropyLoss()
        loss: Tensor = loss_function(self.model(X_fgm), label)
        loss.backward()

        gradient = X_fgm.grad
        d = torch.zeros(gradient.shape, device=self.device)
        for i in range(gradient.shape[0]):
            d_ = gradient[i].data
            norm_grad = d_ / np.linalg.norm(d_.cpu().numpy())
            d[i] = norm_grad * self.epsilon

        # 生成对抗样本
        x_adv = X_fgm + d

        if clip_max == None and clip_min == None:
            clip_max = np.inf
            clip_min = -np.inf

        x_adv = torch.clamp(x_adv, clip_min, clip_max)
        x_adv.to(self.device)

        return x_adv


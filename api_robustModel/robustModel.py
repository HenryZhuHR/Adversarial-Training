import os
import torch
from torch import nn
from torch import Tensor
from resnet34 import ResNet34


class RobustResnet34():
    def __init__(self, device='cpu'):
        self.device = device
        self.num_class = 10

        # Load model
        model_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'models')
        # - robust model
        self.model = ResNet34()
        self.model.linear = nn.Linear(self.model.linear.in_features, self.num_class)
        self.model.load_state_dict(torch.load(os.path.join(model_dir, 'resnet34-robust.pt')))
        self.model.to(self.device)

    def inference(self, x: Tensor) -> Tensor:
        x = x.to(self.device)
        self.mode.eval()
        with torch.no_grad():
            x = self.model(x)
        x = torch.squeeze(x)
        x = torch.softmax(x, dim=0)
        return x

    def top_k(self, x: Tensor, k: int = 5) -> Tensor:
        pass

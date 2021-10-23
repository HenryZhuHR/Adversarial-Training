import os
import json
import torch
from torch import nn
from torch import Tensor
from api_robustModel.resnet34 import ResNet34

DEFAULT_MODEL_PATH = os.path.join(
    os.path.split(os.path.realpath(__file__))[0],
    'models', 'resnet34.pt')

with open(os.path.join(os.path.split(os.path.realpath(__file__))[0], 'models', 'class_indices.json'), 'r') as f:
    CLASS_INDICES = list(dict(json.load(f)).values())


class RobustResnet34():
    """Robust ResNet-34 Model Encapsulation.

    Attributes:
        device: A string to indicating model inference device, such as cpu, cuda or cuda:0.
        num_class: An integer count of class.
        model: A torch.nn.Module.

    Example:
        >>> model = RobustResnet34(\
            model_weight_path='<your_model_weight_path>',\
            device=DEVICE)
    """

    def __init__(self,
                 model_weight_path=DEFAULT_MODEL_PATH,
                 device='cpu'  # default device is cpu
                 ):
        self.device = device
        self.num_class = 10

        # Load model
        self.model = ResNet34()
        self.model.linear = nn.Linear(
            self.model.linear.in_features, self.num_class)

        self.model.load_state_dict(
            torch.load(model_weight_path, map_location=torch.device(device)))
        self.model.to(self.device)

    def inference(self, x: Tensor) -> Tensor:
        """
            - x: Tensor [1,3,224,224]
        """
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = x.to(self.device)
        self.model.eval()
        with torch.no_grad():
            x = self.model(x)
        x = torch.squeeze(x)
        x = torch.softmax(x, dim=0)
        return x

    def top_k(self, x: Tensor, k: int = 5):
        """
            - x: Tensor [1,3,224,224]
            - k: sort tensor and select top-k
        """
        values, indices = torch.topk(self.inference(x), k, dim=0)
        class_name = []
        for i in indices.tolist():
            class_name.append(CLASS_INDICES[i])
        return values.tolist(), indices.tolist(), class_name

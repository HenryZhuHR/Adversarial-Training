import os
<<<<<<< HEAD
import json
import torch
from torch import nn
from torch import Tensor
from api_robustModel.resnet34 import ResNet34
=======
import torch
from torch import nn
from torch import Tensor
from resnet34 import ResNet34
>>>>>>> f9ed4aca33e27478e8f2824767c1d04193fcb3e9

DEFAULT_MODEL_PATH = os.path.join(
    os.path.split(os.path.realpath(__file__))[0],
    'models', 'resnet34.pt')
with open(os.path.join(os.path.split(os.path.realpath(__file__))[0],'models', 'class_indices.json'),'r') as f :
    CLASS_INDICES=list(dict(json.load(f)).values())

class RobustResnet34():
<<<<<<< HEAD
    def __init__(self,
                 model_weight_path=DEFAULT_MODEL_PATH,
                 device='cpu'  # default device is cpu
                 ):
=======
    def __init__(self, device='cpu'):
>>>>>>> f9ed4aca33e27478e8f2824767c1d04193fcb3e9
        self.device = device
        self.num_class = 10

        # Load model
<<<<<<< HEAD
        self.model = ResNet34()
        self.model.linear = nn.Linear(
            self.model.linear.in_features, self.num_class)

        self.model.load_state_dict(
            torch.load(model_weight_path, map_location=torch.device(device)))
=======
        model_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'models')
        # - robust model
        self.model = ResNet34()
        self.model.linear = nn.Linear(self.model.linear.in_features, self.num_class)
        self.model.load_state_dict(torch.load(os.path.join(model_dir, 'resnet34-robust.pt')))
>>>>>>> f9ed4aca33e27478e8f2824767c1d04193fcb3e9
        self.model.to(self.device)

    def inference(self, x: Tensor) -> Tensor:
        '''
            - x: Tensor [1,3,224,224]
        '''
        if (x.dim())==3:
            x=x.unsqueeze(0)
        x = x.to(self.device)
<<<<<<< HEAD
        self.model.eval()
=======
        self.mode.eval()
>>>>>>> f9ed4aca33e27478e8f2824767c1d04193fcb3e9
        with torch.no_grad():
            x = self.model(x)
        x = torch.squeeze(x)
        x = torch.softmax(x, dim=0)
        return x

    def top_k(self, x: Tensor, k: int = 5):
        '''
            - x: Tensor [1,3,224,224]
            - k: 
        '''
        values,indices=torch.topk(self.inference(x),k,dim=0)
        class_name=[]
        for i in indices.tolist():
            class_name.append(CLASS_INDICES[i])
        return values.tolist(),indices.tolist(),class_name

import os
import torch
from torch import nn
from torch import Tensor
from torchvision.models import resnet34,resnet50

"""
    Dual Attention Network for Scene Segmentation
    ---
    - scale attention, object detection
    - arXiv(Paper):     https://arxiv.org/abs/1809.02983
    - code(official):   https://github.com/junfu1115/DANet

"""

class PAM(nn.Module):
    """ Position attention module"""

    def __init__(self, in_channels, **kwargs):
        super(PAM, self).__init__()
        self.conv_b = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_c = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_d = nn.Conv2d(in_channels, in_channels, 1)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: Tensor):
        b, c, h, w = x.size()
        # B: reshape & transpose
        feature_b: Tensor = self.conv_b(x)
        feature_b = feature_b.view(b, -1, h*w)
        feature_b = feature_b.permute(0, 2, 1)

        # C: reshape
        feature_c: Tensor = self.conv_c(x)
        feature_c = feature_c.view(b, -1, h*w)

        # S: B^TxC + softmax
        attention_S: Tensor = torch.bmm(feature_b, feature_c)
        attention_S: Tensor = self.softmax(attention_S)

        # D: reshape
        feature_d: Tensor = self.conv_d(x)
        feature_d = feature_d.view(b, -1, h*w)

        # E: reshape       
        feature_e = torch.bmm(feature_d, attention_S)
        # feature_e = feature_e.permute(0,2,1)
        feature_e = feature_e.view(b, -1, h,w)

        out = self.alpha*feature_e+x
        return out

class ResNet_Attention(nn.Module):
    def __init__(self,in_channels=512,num_class=10):
        super(ResNet_Attention, self).__init__()
        backbone_resnet= resnet34(pretrained=True)
        self.backbone = nn.Sequential(*list(backbone_resnet.children())[:-1])

        self.pam=PAM(in_channels)

        self.fc=nn.Linear(512,num_class)
        self.relu=nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        b,c,h,w=x.size()
        
        backbone_feature_map: Tensor = self.backbone(x)        
        
        # print(backbone_feature_map.size())  # torch.Size([1, 512, 1, 1]) decide in_channels
        attention_pam:Tensor=self.pam(backbone_feature_map)
        
        

        # x=torch.cat([backbone_feature_map,attention_pam],dim=1)
        x=torch.mul(backbone_feature_map,attention_pam)

        # print("backbone_feature_map",backbone_feature_map.size())
        # print("attention_pam",attention_pam.size())
        # print("x",x.size())

        x=x.view(b,-1)
        x:Tensor=self.relu(self.fc(x))
        return x


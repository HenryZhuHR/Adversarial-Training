import os
from torch import nn



# === ResNetX with non-local adding in different stage ===
from .resnet34 import (
    ResNet34,
)

model_zoo = {
    'resnet34': ResNet34,
}


def GetModelByName(model_name: str) -> nn.Module:
    try:
        model = model_zoo[model_name]
        return model
    except KeyError as e:
        print('\033[31m[ERROR] No such model name: %s in model zoo:%s %s\033[0m' % (
            e, os.linesep, list(model_zoo.keys())))
        exit()

import os
from torch import nn

from .cnn import (
    MNIST_CNN,
)


# === ResNetX with non-local adding in different stage ===
from .resnet import (
    Resnet34,
    Resnet50,
)

model_zoo = {
    'resnet34': Resnet34,
    'resnet50': Resnet50,
}


def GetModelByName(model_name: str) -> nn.Module:
    try:
        model = model_zoo[model_name]
        return model
    except KeyError as e:
        print('\033[31m[ERROR] No such model name: %s in model zoo:%s %s\033[0m' % (
            e, os.linesep, list(model_zoo.keys())))
        exit()

import os
from torch import nn

from .voc import VOC
from .caltech import Caltech101


dataset_zoo={
    
}


def GetModelByName(model_name: str) -> nn.Module:
    try:
        model = dataset_zoo[model_name]
        return model
    except KeyError as e:
        print('\033[31m[ERROR] No such model name: %s in model zoo:%s %s\033[0m' % (
            e, os.linesep, list(dataset_zoo.keys())))
        exit()
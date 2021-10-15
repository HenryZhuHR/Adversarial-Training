import os

from .utils import *

from .base_attack import BaseAttack
from .fgsm import FGSM 
from .pgd import PGD 
from .cw import CW
from .deepfool import DeepFool


attack_method_zoo = {
    'fgsm': FGSM,
    'FGSM': FGSM,
    'pgd':PGD,
    'PGD':PGD,
    'cw':CW,
    'CW':CW,
    'deepfool':DeepFool,
    'DeepFool':DeepFool,
}


def GetAttackByName(model_name: str):
    try:
        model:BaseAttack = attack_method_zoo[model_name]
        return model
    except KeyError as e:
        print('\033[31m[ERROR] No such attack method "%s" in attack method zoo:%s %s\033[0m' % (
            e, os.linesep, list(attack_method_zoo.keys())))
        exit()

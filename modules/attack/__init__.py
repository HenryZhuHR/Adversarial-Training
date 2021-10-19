import os

from .utils import *

from .base_attack import BaseAttack

from .pgd import PGD 



attack_method_zoo = {
    'pgd':PGD,
    'PGD':PGD,
}


def GetAttackByName(model_name: str):
    try:
        model:BaseAttack = attack_method_zoo[model_name]
        return model
    except KeyError as e:
        print('\033[31m[ERROR] No such attack method "%s" in attack method zoo:%s %s\033[0m' % (
            e, os.linesep, list(attack_method_zoo.keys())))
        exit()

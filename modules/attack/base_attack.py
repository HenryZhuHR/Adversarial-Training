import os
from abc import abstractmethod, ABCMeta
from typing import List
from torch import Tensor, nn


class BaseAttack(metaclass=ABCMeta):
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.model: nn.Module = model
        self.device: str = device
        self.attack_name = 'Base Attack'

        # all attack parameter needed for this attack method
        # in subclass:
        #   use self.ATTACK_PARAMETERS.update() to define and update parameters
        self.ATTACK_PARAMETERS = {}

    @abstractmethod
    def attack(self, images: Tensor, labels: Tensor):
        """
        attack
        ===        
        """
        adv_images = images
        return adv_images

    def parse_params(self, all_params: List[str]):
        """
        Parse Attack Parameters
        ===
        parse and filter parameters according to attack method
        """
        all_params_dict = dict()
        for param_str in all_params:
            param, value = param_str.split('=')
            all_params_dict[param] = float(value)

        print('-'*33)
        # filter param by self.ATTACK_PARAMETERS
        valid_flag = chr(128640)
        print('Get parameters(%s means valid attack parameter) :' % valid_flag)
        for param in all_params_dict.keys():
            if param in self.ATTACK_PARAMETERS.keys():
                flag = valid_flag
                self.ATTACK_PARAMETERS[param] = all_params_dict[param]
            else:
                flag = '- '
            print('  %s %-15s: %s' % (flag, param, all_params_dict[param]))

        print('Default parameters:')
        default_flag = chr(128296)
        for param in self.ATTACK_PARAMETERS.keys():
            if param not in all_params_dict.keys():
                print('  %s %-15s: %s' %
                      (default_flag, param, self.ATTACK_PARAMETERS[param]))
        

    def print_valid_params(self):
        print('-'*33)
        print('%s \033[32m%s\033[0m Attack valid parameters:' %
              (chr(128640), self.attack_name))
        
        for param, value in self.ATTACK_PARAMETERS.items():
            print('  %s %-15s: %s' % (chr(128296), param, value))

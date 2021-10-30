import os
import tqdm
import openpyxl
import cv2
import numpy
from PIL import Image
import torch
from torch import nn
from torch import Tensor
from torchvision import transforms

from api_robustModel import RobustResnet34


DEVICE = 'cpu'
TRANSFORM = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])
"""
resnet34-2_0.03_5
    - epsilon   = 2/255
    - alpha     = 0.03
    - iters     = 5
"""


def test_dataset(
        dataset_path='E:/datasets/gc10_dsets/pgd_8_255_demo',
        model_path='api_robustModel/models/resnet34.pt'
):
    error_files = list()
    model = RobustResnet34(model_weight_path=model_path, device=DEVICE)

    error_count = 0
    total_count = 0
    class_id = 0

    for class_name in os.listdir(dataset_path):
        # pbar = tqdm.tqdm(os.listdir(os.path.join(dataset_path, class_name)))
        pbar = os.listdir(os.path.join(dataset_path, class_name))
        for image_file in pbar:
            image_path = os.path.join(dataset_path, class_name, image_file)
            image: numpy.ndarray = cv2.imread(image_path)
            if image.shape[2] != 3:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_tensor: Tensor = TRANSFORM(Image.fromarray(image))

            prob_list, index_list, name_list = model.top_k(image_tensor, k=5)
            # pbar.set_description('%2d %s/%s' %(class_id, class_name, name_list[0]))
            if name_list[0] != class_name:
                error_files.append(image_path)
                error_count += 1
            total_count += 1
        class_id += 1
    return error_files


if __name__ == '__main__':
    model_names = {
        'resnet34': 'pure',
        'resnet34-adv-2': 'adv_train-2/255',
        'resnet34-adv-8': 'adv_train-8/255',
        'adv_re~lr=1e-4-best': 'adv_re~lr=1e-4-best'
    }
    datasets = {
        'test224': 'test',
        'pgd_8_255_demo': 'pgd=8/255',
        'recon_pgd_8_255_demo': 'recon-pgd=8/255',
        'recon_clean_demo': 'recon-clean'
    }

    for model_name in model_names.keys():
        for dataset in datasets.keys():
            error_files = test_dataset(
                dataset_path='E:/datasets/gc10_dsets/%s' % dataset,
                model_path='api_robustModel/models/%s.pt' % model_name)

            acc = 1-len(error_files)/2000
            print('[acc] %-2.2f%%  [model] %-15s  [data] %-15s' %
                  (acc*100, model_name, dataset))

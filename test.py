import os
import tqdm
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



def test_model(
    model_path='api_robustModel/models/resnet34.pt',
    dataset_path='E:/datasets/gc10_none_mask_divided/test',
):
    model = RobustResnet34(
        # model_weight_path='api_robustModel/models/resnet34.pt',
        model_weight_path='../server/checkpoints/resnet34-adv-2_0.03_10-best.pt',
        # model_weight_path='../server/checkpoints/resnet34-best.pt',
        device=DEVICE)

    correct_count = 0
    total_count = 0
    class_id = 0
    for class_dir in os.listdir(dataset_path):
        pbar = tqdm.tqdm(os.listdir(os.path.join(dataset_path, class_dir)))
        for image in pbar:
            pbar.set_description('%d %s: %s' % (class_id, class_dir, image))
            image_path = os.path.join(dataset_path, class_dir, image)
            image_tensor: Tensor = TRANSFORM(Image.open(image_path))
            prob_list, index_list, class_name = model.top_k(image_tensor, k=1)
            if class_name[0] == class_dir:
                correct_count += 1
            total_count += 1
        class_id += 1
    print('Test Accu = %.5f   ' % (correct_count/total_count), end='')
    print('model:', model_path)


def test_dataset(
        dataset_path='E:/datasets/gc10_dsets/pgd_8_255_demo',
        model_path='api_robustModel/models/resnet34.pt'
):
    error_files=list()
    model = RobustResnet34(model_weight_path=model_path, device=DEVICE)

    error_count = 0
    total_count = 0
    class_id = 0

    for class_name in os.listdir(dataset_path):
        pbar = tqdm.tqdm(os.listdir(os.path.join(dataset_path, class_name)))
        for image_file in pbar:
            image_path = os.path.join(dataset_path, class_name, image_file)
            image_tensor: Tensor = TRANSFORM(Image.open(image_path))
            prob_list, index_list, name_list = model.top_k(image_tensor, k=5)
            pbar.set_description('%2d %s/%s' % (class_id,class_name, name_list[0]))
            if name_list[0]!=class_name:
                error_files.append(image_path)
                error_count+=1
            total_count+=1
        class_id+=1
    return error_files


if __name__ == '__main__':


    # pgd_8_255_demo recon_pgd_8_255_demo test224
    error_files=test_dataset(dataset_path='E:/datasets/gc10_dsets/pgd_8_255_demo',
                 model_path='api_robustModel/models/resnet34.pt')
    print(len(error_files))

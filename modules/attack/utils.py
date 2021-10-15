import torch
from torchvision import utils as vutils
import cv2
import numpy as np
from torchvision import transforms

def save_image_tensor(filename: str, input_tensor: torch.Tensor):
    """
    将tensor保存为图片
    :param input_tensor: 要保存的tensor
    :param filename: 保存的文件名
    """
    # print(input_tensor.size())
    if (len(input_tensor.shape) != 4) and (len(input_tensor.shape) == 3):  # [3,a,b]->[1,3,a,b]
        input_tensor = torch.unsqueeze(input_tensor, dim=0)

    if input_tensor.shape[0] != 1:
        input_tensor = torch.cat([tensor for tensor in input_tensor], dim=2)
    
    tf = transforms.Compose([
        transforms.ToPILImage()
    ])

    # 复制一份
    input_tensor = input_tensor.clone().detach()
    # 到cpu
    # input_tensor = input_tensor.to(torch.device('cpu'))

    # print(input_tensor.size())
    input_img=tf(input_tensor.cpu())
    input_img=cv2.cvtColor(np.asarray(input_img),cv2.COLOR_RGB2BGR)

    cv2.imwrite(filename, input_img)

    # 反归一化
    # input_tensor = unnormalize(input_tensor)
    # vutils.save_image(input_tensor, filename)

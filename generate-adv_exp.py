import torch
from torch import nn
from torch import Tensor


def pgd_attack(model:nn.Module,
               images: Tensor,
               labels: Tensor,
               device='cpu',
               eps=4/255,
               alpha=0.04,
               iters=10):
    images = images.to(device)
    labels = labels.to(device)
    loss = nn.CrossEntropyLoss()

    ori_images = images.data

    for i in range(iters):
        images.requires_grad = True
        outputs = model(images)

        model.zero_grad()
        cost = loss(outputs, labels).to(device)
        cost.backward()

        adv_images = images + alpha*images.grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, min=0, max=1).detach_()

    eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
    images = torch.clamp(ori_images + 2 * eta, min=0, max=1).detach_()

    return images

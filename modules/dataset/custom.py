import os
import torch


class CustomDataset(torch.utils.data.dataset.Dataset):
    def __init__(self) -> None:
        pass

    def __getitem__(self, index:int):
        image=torch.Tensor()
        label=torch.Tensor()
        return image,label

    def __len__(self):
        return # length of your dataset 
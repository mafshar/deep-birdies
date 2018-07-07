
import os
import sys

import torch
import torch.nn as nn
import torch.nn.parallel

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torchvision import utils as vutils

BATCH_SIZE = 64
IMG_SIZE = 64

def get_transformations():
    return transforms.Compose(
        [transforms.Scale(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def load_data(data_path):
    try:
        data = dataset.CIFAR10(
            root=data_path,
            download=True,
            transform=get_transformations())
        dataloader = DataLoader(
            data,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=2)
    except:
        print 'Loading data failed. Exiting...'
        exit(-1)
    return dataloader

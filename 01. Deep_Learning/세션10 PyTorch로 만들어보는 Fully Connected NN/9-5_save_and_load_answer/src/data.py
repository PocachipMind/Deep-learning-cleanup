# Copyright 2023, Acadential, All rights reserved.
import torch
import torchvision
import numpy as np
from torchvision import transforms
from tqdm import tqdm


def get_dataloaders():
    train_data = torchvision.datasets.MNIST(
        root="../.cache",
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )

    test_data = torchvision.datasets.MNIST(
        root="../.cache",
        train=False,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_data, batch_size=64, num_workers=2, shuffle=True  # dataset
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_data,
        batch_size=64,
        num_workers=2,
        shuffle=False,  # we don't need to shuffle the test data
    )

    return train_dataloader, test_dataloader

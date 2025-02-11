import torch
import torchvision
import numpy as np
from torchvision import transforms
from tqdm import tqdm

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def get_dataloaders():
    train_data = torchvision.datasets.CIFAR10(
        root="../.cache",
        train=True,
        download=True,
        transform=transforms.Compose([torchvision.transforms.ToTensor(), normalize]),
    )

    test_data = torchvision.datasets.CIFAR10(
        root="../.cache",
        train=False,
        download=True,
        transform=transforms.Compose([torchvision.transforms.ToTensor(), normalize]),
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

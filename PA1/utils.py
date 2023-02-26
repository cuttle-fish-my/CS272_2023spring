import os

import torch
import platform
import argparse
import torch.utils.data

import torchvision

from torch.utils.data import DataLoader as DataLoader
from torchvision.datasets import CIFAR10 as CIFAR10


def get_args(train: bool = True):
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_name', type=str, default='None')
    parser.add_argument('--model_name', type=str, default='resnet18')
    parser.add_argument('--model_dir', type=str, default='None')
    parser.add_argument('--imagenet_pretrained', type=bool, default=False)
    if train:
        parser.add_argument('--train_data', type=str, default='CIFAR10')
        parser.add_argument('--val_data', type=str, default='CIFAR10')

        parser.add_argument('--start_epoch', type=int, default=0)
        parser.add_argument('--batch_size', type=int, default=1)
        parser.add_argument('--epochs', type=int, default=1000)

        parser.add_argument('--lr', type=float, default=0.001)
        parser.add_argument('--momentum', type=float, default=0.9)
        parser.add_argument('--weight_decay', type=float, default=0.0001)

        parser.add_argument('--log_interval', type=int, default=10)
        parser.add_argument('--save_interval', type=int, default=100)
        parser.add_argument('--save_dir', type=str, default='models')
    else:
        parser.add_argument('--test_data', type=str, default='CIFAR10')

    return parser.parse_args()


def create_model(opt):
    model_name = opt.model_name
    pretrained = opt.imagenet_pretrained
    model = None
    if model_name == 'resnet18':
        model = torchvision.models.resnet18(pretrained=pretrained)
    elif model_name == 'resnet34':
        model = torchvision.models.resnet34(pretrained=pretrained)
    elif model_name == 'resnet50':
        model = torchvision.models.resnet50(pretrained=pretrained)
    elif model_name == 'resnet101':
        model = torchvision.models.resnet101(pretrained=pretrained)
    elif model_name == 'resnet152':
        model = torchvision.models.resnet152(pretrained=pretrained)

    if not pretrained and os.path.exists(opt.model_dir):
        model.load_state_dict(torch.load(opt.model_dir))

    return model


def creat_data_loader(opt, train):
    if train:
        if opt.train_data == 'CIFAR10':
            transform = torchvision.transforms.Compose([
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465],
                    std=[0.2023, 0.1994, 0.2010])
            ])
            dataset = CIFAR10(root='./data/CIFAR10', train=True, download=True, transform=transform)
            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [45000, 5000])
            train_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=True, pin_memory=True)
            return train_loader, val_loader
        else:
            pass
    else:
        if opt.test_data == 'CIFAR10':
            transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465],
                    std=[0.2023, 0.1994, 0.2010])
            ])
            test_dataset = CIFAR10(root='./data/CIFAR10', train=False, download=True, transform=transform)
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True)
            return test_loader
        else:
            pass


def dev():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif "macOS" in platform.platform():
        return torch.device('mps')
    else:
        return torch.device('cpu')

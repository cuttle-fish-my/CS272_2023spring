import os

import torch
import platform
import argparse
import torch.utils.data
import torch.nn as nn
import torchvision

from torch.utils.data import DataLoader as DataLoader
from torchvision.datasets import CIFAR10 as CIFAR10
from torchvision.models import resnet


def get_args(train: bool = True):
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_name', type=str, default='None')
    parser.add_argument('--model_name', type=str, default='resnet18')
    parser.add_argument('--model_dir', type=str, default='None')
    parser.add_argument('--imagenet_pretrained', type=bool, default=False)
    parser.add_argument('--dataset_name', type=str, default='CIFAR10')
    if train:
        parser.add_argument('--start_epoch', type=int, default=0)
        parser.add_argument('--batch_size', type=int, default=1)
        parser.add_argument('--epochs', type=int, default=1000)

        parser.add_argument('--lr', type=float, default=0.001)
        parser.add_argument('--momentum', type=float, default=0.9)
        parser.add_argument('--weight_decay', type=float, default=0.0001)

        parser.add_argument('--log_interval', type=int, default=10)
        parser.add_argument('--save_interval', type=int, default=100)
        parser.add_argument('--save_dir', type=str, default='models')

    return parser.parse_args()


def create_model(opt):
    model_name = opt.model_name
    pretrained = opt.imagenet_pretrained
    model = None
    if model_name == 'resnet18':
        model = resnet.resnet18(weights=resnet.ResNet18_Weights if pretrained else None)
    elif model_name == 'resnet34':
        model = resnet.resnet34(weights=resnet.ResNet34_Weights if pretrained else None)
    elif model_name == 'resnet50':
        model = resnet.resnet50(weights=resnet.ResNet50_Weights if pretrained else None)
    elif model_name == 'resnet101':
        model = resnet.resnet101(weights=resnet.ResNet101_Weights if pretrained else None)
    elif model_name == 'resnet152':
        model = resnet.resnet152(weights=resnet.ResNet152_Weights if pretrained else None)

    if opt.dataset_name == 'CIFAR10':
        conv1_out_channels = model.conv1.out_channels
        model.conv1 = resnet.conv3x3(3, conv1_out_channels)
        fc_in_features = model.fc.in_features
        model.fc = nn.Linear(fc_in_features, 10)

    if not pretrained and os.path.exists(opt.model_dir):
        model.load_state_dict(torch.load(opt.model_dir))

    return model


def creat_data_loader(opt, train):
    if opt.dataset_name == 'CIFAR10':
        transform = []
        if train:
            transform.append(torchvision.transforms.RandomHorizontalFlip())
        transform.append(torchvision.transforms.ToTensor())
        transform.append(torchvision.transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]))
        transform = torchvision.transforms.Compose(transform)
        dataset = CIFAR10(root='./data/CIFAR10', train=train, download=True, transform=transform)
        if train:
            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [45000, 5000])
            train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=True, pin_memory=True)
            return train_loader, val_loader
        else:
            test_loader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True)
            return test_loader
    else:
        print("ShanghaiTech_Crowd_Counting_Dataset not implemented yet!")
        pass


def dev():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif "macOS" in platform.platform():
        return torch.device('mps')
    else:
        return torch.device('cpu')

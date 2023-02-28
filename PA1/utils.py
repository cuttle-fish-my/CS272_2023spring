import os

import numpy as np
import torch
import platform
import argparse
import torch.utils.data
import torch.nn as nn
import torchvision

from torch.utils.data import DataLoader as DataLoader
from torchvision.datasets import CIFAR10 as CIFAR10
from torchvision.models import resnet
from torch.nn.functional import cross_entropy as cross_entropy


def get_args(train: bool = True):
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_name', type=str, default='None')
    parser.add_argument('--model_name', type=str, default='resnet18')
    parser.add_argument('--load_dir', type=str, default='None')
    parser.add_argument('--imagenet_pretrained', action='store_true')
    parser.add_argument('--dataset_name', type=str, default='CIFAR10')
    if train:
        parser.add_argument('--batch_size', type=int, default=1)
        parser.add_argument('--epochs', type=int, default=1000)

        parser.add_argument('--lr', type=float, default=0.001)
        parser.add_argument('--momentum', type=float, default=0.9)
        parser.add_argument('--weight_decay', type=float, default=0.0001)

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

    # if pretrained:
    #     for param in model.parameters():
    #         param.requires_grad = False

    if opt.dataset_name == 'CIFAR10':
        conv1_out_channels = model.conv1.out_channels
        model.conv1 = resnet.conv3x3(3, conv1_out_channels)
        fc_in_features = model.fc.in_features
        model.fc = nn.Linear(fc_in_features, 10)

    total_train_loss = []
    total_val_loss = []
    total_train_acc = []
    total_val_acc = []
    iteration = 0

    if not pretrained and os.path.exists(opt.load_dir):
        model.load_state_dict(torch.load(os.path.join(opt.load_dir, 'model.pth')))
        total_train_loss = np.load(os.path.join(opt.load_dir, 'train_loss.npy')).tolist()
        total_val_loss = np.load(os.path.join(opt.load_dir, 'val_loss.npy')).tolist()
        total_train_acc = np.load(os.path.join(opt.load_dir, 'train_acc.npy')).tolist()
        total_val_acc = np.load(os.path.join(opt.load_dir, 'val_acc.npy')).tolist()
        iteration = np.load(os.path.join(opt.load_dir, 'iteration.npy')).item()
    return model, total_train_loss, total_val_loss, total_train_acc, total_val_acc, iteration


def creat_data_loader(opt, train):
    if opt.dataset_name == 'CIFAR10':
        transform = []
        if train:
            transform.append(torchvision.transforms.RandomCrop(32, padding=4))
            transform.append(torchvision.transforms.RandomHorizontalFlip())
        transform.append(torchvision.transforms.ToTensor())
        transform.append(torchvision.transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]))
        transform = torchvision.transforms.Compose(transform)
        dataset = CIFAR10(root='./data/CIFAR10', train=train, download=True, transform=transform)
        if train:
            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [45000, 5000],
                                                                       generator=torch.Generator().manual_seed(42))
            train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=True, pin_memory=True)
            return train_loader, val_loader
        else:
            test_loader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True)
            return test_loader
    else:
        print("ShanghaiTech_Crowd_Counting_Dataset not implemented yet!")
        pass


def run_one_epoch(model, optimizer, loader, loss_function=cross_entropy, train: bool = True, iteration: int = 0,
                  lr_scheduler=None, terminate_function: callable = None):
    if train:
        model.train()
    else:
        model.eval()
    device = dev()
    avg_loss = []
    avg_acc = []
    for data, label in loader:
        # forward
        data, label = data.to(device), label.to(device)
        output = model(data)
        label_pred = torch.argmax(output.detach().to('cpu'), dim=1)
        avg_acc.append((label_pred == label.to('cpu')).sum() / data.shape[0])
        loss = loss_function(output, label)
        avg_loss.append(loss.detach().to('cpu'))
        if train:
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iteration += 1
            if lr_scheduler is not None:
                lr_scheduler(iteration, optimizer)
            if terminate_function is not None:
                if terminate_function(iteration):
                    del data, label, output, loss
                    break
        del data, label, output, loss
    return torch.mean(torch.tensor(avg_loss)), torch.mean(torch.tensor(avg_acc)), iteration


def dev():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif "macOS" in platform.platform():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def save_model(model, train_loss, val_loss, train_acc, val_acc, iteration, root_path):
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    torch.save(model.state_dict(), os.path.join(root_path, 'model.pth'))
    np.save(os.path.join(root_path, 'train_loss.npy'), train_loss)
    np.save(os.path.join(root_path, 'val_loss.npy'), val_loss)
    np.save(os.path.join(root_path, 'train_acc.npy'), train_acc)
    np.save(os.path.join(root_path, 'val_acc.npy'), val_acc)
    np.save(os.path.join(root_path, 'iteration.npy'), iteration)


def CIFAR10_lr_scheduler(iteration: int, optimizer):
    if 32000 <= iteration < 48000:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.01
    elif iteration >= 48000:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.001


def CIFAR10_terminate(iteration: int):
    if iteration >= 64000:
        return True
    else:
        return False


def CIFAR10_FT_lr_scheduler(iteration: int, optimizer):
    if 15000 <= iteration < 35000:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.001
    elif iteration >= 35000:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0001


def creat_lr_scheduler(opt):
    if opt.dataset_name == 'CIFAR10':
        if opt.imagenet_pretrained:
            return CIFAR10_FT_lr_scheduler
        else:
            return CIFAR10_lr_scheduler
    else:
        print("ShanghaiTech_Crowd_Counting_Dataset not implemented yet!")
        pass


def creat_terminate_function(opt):
    if opt.dataset_name == 'CIFAR10':
        if opt.imagenet_pretrained:
            return None
        else:
            return CIFAR10_terminate
    else:
        print("ShanghaiTech_Crowd_Counting_Dataset not implemented yet!")
        pass
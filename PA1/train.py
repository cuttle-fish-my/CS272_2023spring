import torch
import torchvision

import utils
import os


def train(opt):
    model = utils.create_model(opt)
    train_loader, val_loader = utils.creat_data_loader(opt, train=True)
    for epoch in range(opt.start_epoch, opt.epochs):
        for data, label in train_loader:
            print(data.shape)
            print(label.shape)
            break
        break


if __name__ == "__main__":
    args = utils.get_args(train=True)
    train(args)

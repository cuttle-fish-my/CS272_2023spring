import torch
import torchvision

import utils
import os

dev = utils.dev()


def train(opt):
    model = utils.create_model(opt)
    model = model.to(device=dev)
    train_loader, val_loader = utils.creat_data_loader(opt, train=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
    for epoch in range(opt.start_epoch, opt.epochs):
        model.train()
        for data, label in train_loader:
            data, label = data.to(device=dev), label.to(device=dev)
            output = model(data)
            loss = torch.nn.functional.cross_entropy(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


if __name__ == "__main__":
    args = utils.get_args(train=True)
    train(args)

import torch
import torchvision

import utils
import os
import numpy as np
from matplotlib import pyplot as plt

dev = utils.dev()
fig = plt.figure(figsize=(20, 15))


def train(opt):
    model, total_train_loss, total_val_loss = utils.create_model(opt)
    model = model.to(device=dev)
    train_loader, val_loader = utils.creat_data_loader(opt, train=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
    best_val_loss = 1e10

    save_path = os.path.join(opt.save_dir, opt.exp_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    best_path = os.path.join(save_path, 'best')
    if not os.path.exists(best_path):
        os.makedirs(best_path)

    for epoch in range(opt.start_epoch, opt.epochs):

        avg_train_loss = utils.run_one_epoch(model, optimizer, train_loader, train=True)
        total_train_loss.append(avg_train_loss)

        avg_val_loss = utils.run_one_epoch(model, optimizer, val_loader, train=False)
        total_val_loss.append(avg_val_loss)

        print("epoch {}: training_loss {:.2f} val_loss {:.2f}".format(epoch, avg_train_loss, avg_val_loss))

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            utils.save_model(model, total_train_loss, total_val_loss, os.path.join(save_path, 'best'))

        if epoch % opt.save_interval == 0:
            utils.save_model(model, total_train_loss, total_val_loss, os.path.join(save_path, f'{epoch}'))

        plt.clf()
        plt.plot(total_train_loss, label='training_loss', color='lime')
        plt.plot(total_val_loss, label='validation_loss', color='magenta')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig(os.path.join(save_path, 'loss_curve.png'))


if __name__ == "__main__":
    args = utils.get_args(train=True)
    train(args)

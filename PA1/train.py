import torch
import torchvision

import utils
import os
import numpy as np
from matplotlib import pyplot as plt

dev = utils.dev()
fig = plt.figure(figsize=(20, 15))


def train(opt):
    model, total_train_loss, total_val_loss, total_train_acc, total_val_acc = utils.create_model(opt)
    model = model.to(device=dev)
    train_loader, val_loader = utils.creat_data_loader(opt, train=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
    best_val_loss = 1e10

    save_path = os.path.join(opt.save_dir, opt.exp_name)

    iteration = len(total_train_acc) * int(len(train_loader) / opt.batch_size)
    utils.CIFAR10_lr_scheduler(iteration, optimizer)

    for epoch in range(len(total_train_acc), opt.epochs):

        avg_train_loss, avg_train_acc = utils.run_one_epoch(model, optimizer, train_loader, train=True,
                                                            iteration=iteration,
                                                            lr_scheduler=utils.CIFAR10_lr_scheduler)
        total_train_loss.append(avg_train_loss)
        total_train_acc.append(avg_train_acc)

        avg_val_loss, avg_val_acc = utils.run_one_epoch(model, optimizer, val_loader, train=False)
        total_val_loss.append(avg_val_loss)
        total_val_acc.append(avg_val_acc)

        print("epoch {}, iter {}: lr={:.2f} training_loss {:.2f}, val_loss {:.2f}, training_acc {:.2%}, val_acc {:.2%}"
              .format(epoch, iteration, optimizer.param_groups[0]['lr'], avg_train_loss, avg_val_loss, avg_train_acc,
                      avg_val_acc))

        # save model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            utils.save_model(model, total_train_loss, total_val_loss, total_train_acc, total_val_acc,
                             os.path.join(save_path, 'best'))

        if epoch % opt.save_interval == 0:
            utils.save_model(model, total_train_loss, total_val_loss, total_train_acc, total_val_acc,
                             os.path.join(save_path, f'{epoch}'))

        if utils.CIFAR10_terminate(iteration):
            utils.save_model(model, total_train_loss, total_val_loss, total_train_acc, total_val_acc,
                             os.path.join(save_path, 'last'))

        # draw loss curve
        plt.clf()
        ax1 = fig.add_subplot()
        ax1.plot(total_train_loss, label='training_loss', color='lime')
        ax1.plot(total_val_loss, label='validation_loss', color='magenta')
        ax1.set_xlabel('epoch')
        ax1.set_ylabel('Loss')
        ax1.legend(loc='upper right')

        ax2 = ax1.twinx()
        ax2.set_ylabel('accuracy')
        ax2.plot(total_train_acc, label='training_accuracy')
        ax2.plot(total_val_acc, label='validation_accuracy')
        ax2.legend(loc='upper left')

        plt.savefig(os.path.join(save_path, 'loss_curve.png'))


if __name__ == "__main__":
    args = utils.get_args(train=True)
    train(args)

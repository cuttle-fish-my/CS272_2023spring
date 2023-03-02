import os

import torch
import torchvision

import utils

dev = utils.dev()


def test(opt):
    if not os.path.exists('results'):
        os.makedirs('results')
    model, _, _, _, _, _ = utils.create_model(opt)
    model = model.to(dev)
    test_loader = utils.creat_data_loader(opt, train=False)
    loss_function = utils.creat_loss_function(opt)
    loss, acc, _ = utils.run_one_epoch(model, optimizer=None, loader=test_loader, loss_function=loss_function,
                                       train=False)
    print("test loss = {:.3e}, test accuracy = {:.2%}".format(loss, acc))


if __name__ == '__main__':
    args = utils.get_args(train=False)
    test(args)

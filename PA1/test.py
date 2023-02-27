import torch
import torchvision

import utils

dev = utils.dev()


def test(opt):
    model, _, _, _, _, _ = utils.create_model(opt)
    model = model.to(dev)
    test_loader = utils.creat_data_loader(opt, train=False)
    loss, acc, _ = utils.run_one_epoch(model, optimizer=None, loader=test_loader, train=False)
    print("test loss = {:.3e}, test accuracy = {:.2%}".format(loss, acc))


if __name__ == '__main__':
    args = utils.get_args(train=False)
    test(args)

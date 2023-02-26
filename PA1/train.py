import torch
import torchvision

import utils
import os


def train(opt):
    model = utils.create_model(args)


if __name__ == "__main__":
    args = utils.get_args(train=True)
    train(args)

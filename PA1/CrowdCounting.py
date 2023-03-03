import os

import numpy as np
import torch
from PIL import Image as Image
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.transforms import functional as F


class CrowdCountingDataset(Dataset):
    def __init__(self, root, train: bool = True, transform=None):
        self.train = train
        data_dir = None
        if self.train:
            data_dir = os.path.join(root, 'train')
        else:
            data_dir = os.path.join(root, 'test')
        self.transform = transform
        self.img_dir = os.path.join(data_dir, 'images')
        self.gt_dir = os.path.join(data_dir, 'ground_truth')

    def __len__(self):
        return len(os.listdir(self.img_dir))

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, f"{idx + 1}.jpg")
        gt_path = os.path.join(self.gt_dir, f"{idx + 1}.npy")
        img = Image.open(img_path).convert('RGB')
        gt = np.load(gt_path)
        if self.transform:
            img, gt = self.transform(img, gt, self.train)
        return img, gt


def CrowdCountingTransform(img, gt, train: bool = True):
    mean = [110.0426287, 113.97032411, 115.16945588]
    std = [54.32304769, 55.43994135, 57.3708831]
    img = torch.Tensor(np.array(img))
    gt = torch.Tensor(gt)
    img = img.permute(2, 0, 1)
    img = F.normalize(img, mean=mean, std=std)
    if train:
        if np.random.random() > 0.5:
            img = F.hflip(img)
            gt = F.hflip(gt)
        if np.random.random() > 0.5:
            params = T.RandomCrop(gt.shape, padding='reflect').get_params(img, gt.shape)
            img = F.crop(img, *params)
            gt = F.crop(gt, *params)
    return img, gt


def CrowdCountingLoss(pred, gt):
    # return 1e-3 * torch.square(pred - gt[:, None, :, :]).sum() / (2 * pred.shape[0])

    z_pred = pred[:, 0, :, :].sum(axis=(1, 2))
    z_label = gt.sum(axis=(1, 2))
    MSE = torch.sqrt(torch.mean((z_pred - z_label) ** 2))
    MAE = torch.mean(torch.abs(z_pred - z_label))
    return 1e-3 * (MSE + MAE)
    # return torch.square(pred - gt[:, None, :, :]).mean()
    # return torch.nn.BCELoss()(pred, gt[:, None, :, :]) + torch.square(pred - gt[:, None, :, :]).mean()

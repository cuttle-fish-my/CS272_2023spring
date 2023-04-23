import argparse
import csv
import os
import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from PoseDataset import PoseDataset
from model import PoseRAC

torch.multiprocessing.set_sharing_strategy('file_system')


def dev():
    if torch.cuda.is_available():
        return torch.device('cuda')
    # elif "macOS" in platform.platform():
    #     return torch.device('mps')
    else:
        return torch.device('cpu')


# Normalization to improve training robustness.
def normalize_landmarks(all_landmarks):
    x_max = np.expand_dims(np.max(all_landmarks[:, :, 0], axis=1), 1)
    x_min = np.expand_dims(np.min(all_landmarks[:, :, 0], axis=1), 1)

    y_max = np.expand_dims(np.max(all_landmarks[:, :, 1], axis=1), 1)
    y_min = np.expand_dims(np.min(all_landmarks[:, :, 1], axis=1), 1)

    z_max = np.expand_dims(np.max(all_landmarks[:, :, 2], axis=1), 1)
    z_min = np.expand_dims(np.min(all_landmarks[:, :, 2], axis=1), 1)

    all_landmarks[:, :, 0] = (all_landmarks[:, :, 0] - x_min) / (x_max - x_min)
    all_landmarks[:, :, 1] = (all_landmarks[:, :, 1] - y_min) / (y_max - y_min)
    all_landmarks[:, :, 2] = (all_landmarks[:, :, 2] - z_min) / (z_max - z_min)

    all_landmarks = all_landmarks.reshape(len(all_landmarks), 99)
    return all_landmarks


# For each pose, we use 33 key points to represent it, and each key point has 3 dimensions.
# Here we obtain the pose information (33*3=99) of each key frame, and set up the label (1 for salient pose I and 0 for salient pose II).
def obtain_landmark_label(csv_path, all_landmarks, all_labels, label2index, num_classes):
    file_separator = ','
    n_landmarks = 33
    n_dimensions = 3
    with open(csv_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=file_separator)
        for row in csv_reader:
            assert len(row) == n_landmarks * n_dimensions + 2, 'Wrong number of values: {}'.format(len(row))
            landmarks = np.array(row[2:], np.float32).reshape([n_landmarks, n_dimensions])
            all_landmarks.append(landmarks)
            label = label2index[row[1]]

            start_str = row[0].split('/')[-3]
            label_np = np.zeros(num_classes)
            if start_str == 'salient1':
                label_np[label] = 1
            all_labels.append(label_np)
    return all_landmarks, all_labels


def csv2data(train_csv, action2index, num_classes):
    train_landmarks = []
    train_labels = []
    train_landmarks, train_labels = obtain_landmark_label(train_csv, train_landmarks, train_labels, action2index,
                                                          num_classes)

    train_landmarks = np.array(train_landmarks)
    train_labels = np.array(train_labels)
    train_landmarks = normalize_landmarks(train_landmarks)

    return train_landmarks, train_labels


def main(args):
    old_time = time.time()
    if os.path.isfile(args.config):
        with open(args.config, "r") as fd:
            config = yaml.load(fd, Loader=yaml.FullLoader)
    else:
        raise ValueError("Config file does not exist.")

    csv_label_path = config['dataset']['csv_label_path']
    root_dir = config['dataset']['dataset_root_dir']

    if args.heads != -1:
        config['PoseRAC']['heads'] = args.heads
    if args.enc_layer != -1:
        config['PoseRAC']['enc_layer'] = args.dim
    if args.accelerator is not None:
        config['trainer']['accelerator'] = args.accelerator


    train_csv = os.path.join(root_dir, 'annotation_pose', 'train.csv')

    label_pd = pd.read_csv(csv_label_path)
    index_label_dict = {}
    length_label = len(label_pd.index)
    for label_i in range(length_label):
        one_data = label_pd.iloc[label_i]
        action = one_data['action']
        label = one_data['label']
        index_label_dict[label] = action
    num_classes = len(index_label_dict)
    action2index = {v: k for k, v in index_label_dict.items()}

    train_landmarks, train_labels = csv2data(train_csv, action2index, num_classes)
    valid_landmarks, valid_labels = csv2data(train_csv, action2index, num_classes)

    train_dataset = PoseDataset(torch.from_numpy(train_landmarks).float(), torch.from_numpy(train_labels).float())
    valid_dataset = PoseDataset(torch.from_numpy(valid_landmarks).float(), torch.from_numpy(valid_labels).float())

    train_loader = DataLoader(train_dataset, batch_size=config['dataset']['batch_size'], shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config['dataset']['batch_size'], shuffle=False)

    model = PoseRAC(dim=config['PoseRAC']['dim'],
                    heads=config['PoseRAC']['heads'],
                    enc_layer=config['PoseRAC']['enc_layer'],
                    num_classes=num_classes,
                    alpha=config['PoseRAC']['alpha'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['PoseRAC']['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.75, patience=6, verbose=1,
                                                           mode='min', cooldown=0, min_lr=10e-7)
    model.to(dev())

    save_path = os.path.join(config['save_dir'],
                             f"head_{config['PoseRAC']['heads']}_layer_{config['PoseRAC']['enc_layer']}")

    pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
    yaml.dump(config, open(os.path.join(save_path, 'config.yaml'), 'w'))

    plt.Figure(figsize=(10, 10))
    train_loss_list = []
    valid_loss_list = []
    best_loss = 1e8

    for epoch in range(config['trainer']['max_epochs']):
        model.train()
        avg_train_loss = 0
        for i, (landmarks, labels) in enumerate(train_loader):
            landmarks = landmarks.to(dev())
            labels = labels.to(dev())
            _, loss = model(landmarks, labels)
            loss = loss["loss"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_train_loss += loss.item()
        scheduler.step(avg_train_loss)
        avg_train_loss /= len(train_loader)
        train_loss_list.append(avg_train_loss)

        avg_val_loss = 0
        model.eval()
        with torch.no_grad():
            for i, (landmarks, labels) in enumerate(valid_loader):
                landmarks = landmarks.to(dev())
                labels = labels.to(dev())
                _, loss = model(landmarks, labels)
                loss = loss["bce_loss"]
                avg_val_loss += loss.item()
            avg_val_loss /= len(valid_loader)
            valid_loss_list.append(avg_val_loss)
        print("epoch: {}, train_loss: {}, valid_loss: {}".format(epoch, avg_train_loss, avg_val_loss))
        scheduler.step(avg_val_loss)
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(save_path, 'best_model.pth'))
        plt.clf()
        plt.plot(train_loss_list, label='train_loss')
        plt.plot(valid_loss_list, label='valid_loss')
        plt.legend()
        plt.savefig(os.path.join(save_path, 'progress.png'))

    torch.save(model.state_dict(), os.path.join(save_path, "final.pth"))

    current_time = time.time()
    print('time: ' + str(current_time - old_time) + 's')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate our PoseRAC')
    parser.add_argument('--config', type=str, metavar='DIR',
                        help='path to a config file')
    parser.add_argument('--heads', type=int, default=-1)
    parser.add_argument('--enc_layer', type=int, default=-1)
    parser.add_argument('--accelerator', type=str, default=None)
    args = parser.parse_args()
    main(args)

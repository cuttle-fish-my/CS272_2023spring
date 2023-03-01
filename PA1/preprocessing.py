import os
import cv2
from tqdm import tqdm
import numpy as np
import scipy.io as sio
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors


def get_mean_std(data_dir):
    img_dir = os.path.join(data_dir, 'images')
    num_data = len(os.listdir(img_dir))
    mean = np.zeros(3)
    std = np.zeros(3)
    for i in tqdm(range(num_data)):
        img_path = os.path.join(img_dir, f"IMG_{i + 1}.jpg")
        img = cv2.imread(img_path)
        mean += np.mean(img, axis=(0, 1))
        std += np.std(img, axis=(0, 1))
    mean /= num_data
    std /= num_data
    print(f"mean = {mean}, std = {std}")


def gen_dense_map(data_dir, save_dir, k, beta):
    img_dir = os.path.join(data_dir, 'images')
    gt_dir = os.path.join(data_dir, 'ground_truth')
    save_img_dir = os.path.join(save_dir, 'images')
    save_gt_dir = os.path.join(save_dir, 'ground_truth')
    save_gt_jpg_dir = os.path.join(save_dir, 'ground_truth_jpg')

    if not os.path.exists(save_img_dir):
        os.makedirs(save_img_dir)
    if not os.path.exists(save_gt_dir):
        os.makedirs(save_gt_dir)
    if not os.path.exists(save_gt_jpg_dir):
        os.makedirs(save_gt_jpg_dir)

    num_data = len(os.listdir(img_dir))
    X = np.repeat(np.arange(768), 1024).reshape(768, 1024)
    Y = np.repeat(np.arange(1024), 768).reshape(1024, 768).T
    for i in tqdm(range(num_data)):
        img = cv2.imread(os.path.join(img_dir, f"IMG_{i + 1}.jpg"))
        cv2.imwrite(os.path.join(save_img_dir, f"{i + 1}.jpg"), img)

        gt = sio.loadmat(os.path.join(gt_dir, f"GT_IMG_{i + 1}.mat"))['image_info'][0][0][0][0][0]

        KNN = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(gt)
        d, _ = KNN.kneighbors(gt)
        std = beta * d.mean(axis=1)[:, None, None]

        D = np.square(X[None] - gt[:, 1, None, None]) + np.square(Y[None] - gt[:, 0, None, None])

        gaussian_kernels = np.exp(-D / (2 * std ** 2)) / (2 * np.pi * std)
        dense_map = gaussian_kernels.sum(axis=0)

        plt.imshow(dense_map)
        plt.imsave(os.path.join(save_gt_jpg_dir, f"{i + 1}.jpg"), dense_map, cmap='jet')
        np.save(os.path.join(save_gt_dir, f"{i + 1}.npy"), dense_map)


if __name__ == '__main__':
    train_dir = 'data/ShanghaiTech_Crowd_Counting_Dataset/part_B_final/train_data'
    test_dir = 'data/ShanghaiTech_Crowd_Counting_Dataset/part_B_final/test_data'
    # gen_dense_map(train_dir, 'data/Crowd_Counting/train', 5, 0.3)
    # gen_dense_map(test_dir, 'data/Crowd_Counting/test', 5, 0.3)
    get_mean_std(train_dir)
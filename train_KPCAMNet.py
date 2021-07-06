import os
import pickle
import argparse

import gdal
import numpy as np
from sklearn.cluster import KMeans
import sys
from Methodology.Deep.KPCAMNet.KPCANet import KernelPCANet
from Methodology.Deep.KPCAMNet.cluster_util import get_binary_change_map
from Methodology.Deep.KPCAMNet.acc_ass import assess_accuracy
from Methodology.Deep.KPCAMNet.data_pro_util import stad_img, norm_img, random_select_samples
import imageio

parser = argparse.ArgumentParser()

parser.add_argument('--net_depth', type=int, default=3, help='network depth of KPCAMNet[default: 3]')
parser.add_argument('--patch_size', type=int, default=5, help='convolution kernel size [default: 5]')
parser.add_argument('--kernel_func', type=str, default='rbf', help='kernel function[default: rbf]')
parser.add_argument('--gamma_list', type=list, default=[5e-4, 5e-4, 5e-4], help='parameters of kernel function')
parser.add_argument('--save_path', default='result', help='model param path')
parser.add_argument('--data_path', default='data/HY', help='dataset path')
parser.add_argument('--epoch', type=int, default=1, help='epoch to run[default: 20]')
parser.add_argument('--filter_num', type=list, default=[8, 8, 8], help='the number of KPCA convolution kernel')
parser.add_argument('--sample_num', type=int, default=100, help='training sample number[default: 100]')

# basic params
FLAGS = parser.parse_args()

PATCH_SZ = FLAGS.patch_size
NET_DEPTH = FLAGS.net_depth
SAVE_PATH = FLAGS.save_path
DATA_PATH = FLAGS.data_path
KERNEL_FUNC = FLAGS.kernel_func
GAMMA = FLAGS.gamma_list
EPOCH = FLAGS.epoch
FILTER_NUM = FLAGS.filter_num
SAMPLE_NUM = FLAGS.sample_num

if len(FILTER_NUM) != NET_DEPTH:
    print('filter number doesn\'t match network depth! Please check it')
    sys.exit(1)

if len(GAMMA) != len(FILTER_NUM):
    print('gamma number doesn\' match filter number! Please check it')
    sys.exit(1)


def load_data(data_path):
    '''
        load dataset, you should modify this function according to your own dataset
    '''
    data_set_X = gdal.Open(os.path.join(data_path, 'T1'))  # data set X
    data_set_Y = gdal.Open(os.path.join(data_path, 'T2'))  # data set Y

    img_width = data_set_X.RasterXSize  # image width
    img_height = data_set_X.RasterYSize  # image height

    img_X = data_set_X.ReadAsArray(0, 0, img_width, img_height)
    img_Y = data_set_Y.ReadAsArray(0, 0, img_width, img_height)

    img_X = stad_img(img_X, channel_first=True)  # (C, H, W)
    img_Y = stad_img(img_Y, channel_first=True)
    img_X = np.transpose(img_X, [1, 2, 0])  # (H, W, C)
    img_Y = np.transpose(img_Y, [1, 2, 0])  # (H, W, C)
    return img_X, img_Y


def train_net():
    img_X, img_Y = load_data(DATA_PATH)
    height, width, channel = img_X.shape

    print(f'sample number is {2 * SAMPLE_NUM}')

    train_X, train_Y = random_select_samples(img_X, img_Y, n_train=SAMPLE_NUM, patch_sz=PATCH_SZ)
    train_data = np.concatenate([train_X, train_Y], axis=0)

    # limited by the memory size, we have to slide the dataset into 500*500 patch, you can define this according
    # to your own dataset
    step_1 = 500
    step_2 = 500
    before_img = np.reshape(img_X, (1, img_X.shape[0], img_X.shape[1], img_X.shape[2]))
    after_img = np.reshape(img_Y, (1, img_Y.shape[0], img_Y.shape[1], img_Y.shape[2]))
    pred_img = np.concatenate([before_img, after_img], axis=0)  # (2, H, W, C)
    PCANet_model = KernelPCANet(num_stages=NET_DEPTH, patch_size=[PATCH_SZ, PATCH_SZ],
                                num_filters=FILTER_NUM,
                                gamma=GAMMA)
    for s in range(NET_DEPTH):
        trans_img = np.ones((2, height, width, FILTER_NUM[s]))
        PCANet_model.train_net(train_data, stage=s, is_mean_removal=False, kernel=KERNEL_FUNC)

        for i in range(0, height, step_1):
            for j in range(0, width, step_2):
                pred_data = pred_img[:, i:(i + step_1), j:(j + step_2), :]
                net_output = PCANet_model.infer_data(pred_data, stage=s, is_mean_removal=False)
                proj_before_img = net_output[0]
                proj_after_img = net_output[1]
                trans_img[0, i:(i + step_1), j:(j + step_2)] = proj_before_img
                trans_img[1, i:(i + step_1), j:(j + step_2)] = proj_after_img

        pred_img = np.copy(trans_img)  # feature images will be treated as input in the next stage

        # select new training samples for the next KPCA convolutional layer
        change_sample_X, change_sample_Y = random_select_samples(trans_img[0], trans_img[1],
                                                                 n_train=SAMPLE_NUM,
                                                                 patch_sz=PATCH_SZ)

        train_data = np.concatenate([change_sample_X, change_sample_Y], axis=0)

    #############################
    # mapping data into a 2-D polar domain
    #############################
    diff_img = (trans_img[0] - trans_img[1])
    rou = np.sqrt(np.sum(np.square(diff_img), axis=-1))
    eig_value = PCANet_model.filters[-1].lambdas_
    eig_value = np.reshape(eig_value, (1, 1, -1))
    theta = np.arccos(1 / np.sqrt(FILTER_NUM[-1]) * (np.sum(eig_value * diff_img, -1) / np.sqrt(
        np.sum(np.square(eig_value)) * np.sum(np.square(diff_img), axis=-1))))

    ###############################################
    # binary CD
    ###############################################
    print('-------Perform Binary Change Detection-------')
    rou = np.reshape(rou, (-1, 1))
    binary_change_map = get_binary_change_map(rou, method='otsu')
    binary_change_map = np.reshape(binary_change_map, (height, width))
    imageio.imwrite(os.path.join(SAVE_PATH, 'KPCAMNet_BCM.png'), binary_change_map)

    ###############################################
    # multi-class CD
    ###############################################
    print('-------Perform Multi-class Change Detection-------')
    changed_idx = (binary_change_map == 255)
    changed_theta = theta[changed_idx]
    changed_theta = np.reshape(changed_theta, (-1, 1))
    KMeans_cls = KMeans(n_clusters=3, max_iter=1500)
    print('-------Clustering algorithm is running')
    KMeans_cls.fit(changed_theta)
    label_pred = KMeans_cls.labels_  # get cluster label

    multi_change_map = np.zeros((height, width, 3))
    k = 0
    for h in range(height):
        for w in range(width):
            if changed_idx[h, w]:
                if label_pred[k] == 0:
                    multi_change_map[h, w] = np.array([255, 255, 0])
                elif label_pred[k] == 1:
                    multi_change_map[h, w] = np.array([255, 0, 0])
                elif label_pred[k] == 2:
                    multi_change_map[h, w] = np.array([0, 0, 255])
                k += 1
    imageio.imwrite(os.path.join(SAVE_PATH, 'KPCAMNet_MCM.png'), multi_change_map)


if __name__ == '__main__':
    train_net()

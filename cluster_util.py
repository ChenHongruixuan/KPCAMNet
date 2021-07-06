import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eig
from scipy.stats import chi2
from sklearn.cluster import KMeans


def get_binary_change_map(data, method='k_means'):
    """
    get binary change map
    :param data:
    :param method: cluster method
    :return: binary change map
    """
    if method == 'k_means':
        cluster_center = KMeans(n_clusters=2, max_iter=1500).fit(data.T).cluster_centers_.T  # shape: (1, 2)
        # cluster_center = k_means_cluster(weight, cluster_num=2)
        print('k-means cluster is done, the cluster center is ', cluster_center)
        dis_1 = np.linalg.norm(data - cluster_center[0, 0], axis=0, keepdims=True)
        dis_2 = np.linalg.norm(data - cluster_center[0, 1], axis=0, keepdims=True)

        bcm = np.copy(data)  # binary change map
        if cluster_center[0, 0] > cluster_center[0, 1]:
            bcm[dis_1 > dis_2] = 0
            bcm[dis_1 <= dis_2] = 255
        else:
            bcm[dis_1 > dis_2] = 255
            bcm[dis_1 <= dis_2] = 0
    elif method == 'otsu':
        bcm, threshold = otsu(data, num=200)
        print('otsu is done, the threshold is ', threshold)

    return bcm


def otsu(data, num=400):
    """
    generate binary change map based on otsu
    :param data: cluster data
    :param num: intensity number
    :return:
        binary change map
        selected threshold
    """
    max_value = np.max(data)
    min_value = np.min(data)

    total_num = data.shape[1]
    step_value = (max_value - min_value) / num
    value = min_value
    best_threshold = min_value
    best_inter_class_var = 0
    while value <= max_value:
        data_1 = data[data <= value]
        data_2 = data[data > value]
        w1 = data_1.shape[0] / total_num
        w2 = data_2.shape[0] / total_num

        mean_1 = data_1.mean()
        mean_2 = data_2.mean()

        inter_class_var = w1 * w2 * np.power((mean_1 - mean_2), 2)
        if best_inter_class_var < inter_class_var:
            best_inter_class_var = inter_class_var
            best_threshold = value
        value += step_value

    bwp = np.zeros(data.shape)
    bwp[data <= best_threshold] = 0
    bwp[data > best_threshold] = 255

    return bwp, best_threshold

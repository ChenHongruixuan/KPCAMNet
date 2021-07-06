import imageio

import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import accuracy_score


def assess_accuracy(gt_changed, gt_unchanged, changed_map, multi_class=False):
    """
    assess accuracy of changed map based on ground truth
    :param gt_changed: changed ground truth
    :param gt_unchanged: unchanged ground truth
    :param changed_map: changed map
    :return: confusion matrix and overall accuracy
    """
    cm = []
    gt = []

    if multi_class:
        height, width, channel = gt_changed.shape
        for i in range(0, height):
            for j in range(0, width):
                if (changed_map[i, j] == np.array([255, 255, 0])).all():
                    cm.append('soil')
                # elif (changed_map[i, j] == np.array([0, 0, 255])).all():
                #     cm.append('water')
                elif (changed_map[i, j] == np.array([255, 0, 0])).all():
                    cm.append('city')
                else:
                    cm.append('unchanged')
                if (gt_changed[i, j] == np.array([255, 255, 0])).all():
                    gt.append('soil')
                elif (gt_changed[i, j] == np.array([255, 0, 0])).all():
                    gt.append('city')
                # elif (gt_changed[i, j] == np.array([0, 0, 255])).all():
                #     gt.append('water')
                elif (gt_unchanged[i, j] == np.array([255, 255, 255])).all():
                    gt.append('unchanged')
                else:
                    gt.append('undefined')
        conf_mat = confusion_matrix(y_true=gt, y_pred=cm,
                                    labels=['soil', 'city', 'unchanged'])
        kappa_co = cohen_kappa_score(y1=gt, y2=cm,
                                     labels=['soil', 'city', 'unchanged'])
        aa = conf_mat.diagonal() / np.sum(conf_mat, axis=1)
        oa = np.sum(conf_mat.diagonal()) / np.sum(conf_mat)

        return conf_mat, oa, aa, kappa_co

    else:
        height, width = changed_map.shape
        changed_map = np.reshape(changed_map, (-1,))
        gt_changed = np.reshape(gt_changed, (-1,))
        gt_unchanged = np.reshape(gt_unchanged, (-1,))

        cm = np.ones((height * width,))
        cm[changed_map == 255] = 2

        gt = np.zeros((height * width,))
        gt[gt_changed == 255] = 2
        gt[gt_unchanged == 255] = 1


        conf_mat = confusion_matrix(y_true=gt, y_pred=cm,
                                    labels=[1, 2])  # ['soil', 'water', 'city', 'unchanged'])
        kappa_co = cohen_kappa_score(y1=gt, y2=cm,
                                     labels=[1, 2])  # ['soil', 'water', 'city', 'unchanged'])

        oa = np.sum(conf_mat.diagonal()) / np.sum(conf_mat)

        return conf_mat, oa, kappa_co



if __name__ == '__main__':
    # val_func()
    ground_truth_changed = imageio.imread('./Adata/GF_2_2/change.bmp')[:, :, 0]
    ground_truth_unchanged = imageio.imread('./Adata/GF_2_2/unchanged.bmp')  # [:, :, 1]

    cm_path = 'PCANet/compare/SAE_binary.bmp'
    changed_map = imageio.imread(cm_path)


    conf_mat, oa, kappa_co = assess_accuracy(ground_truth_changed, ground_truth_unchanged, changed_map,
                                                 multi_class=False)
    conf_mat_2 = conf_mat.copy()
    conf_mat_2[1, 1] = conf_mat[0, 0]
    conf_mat_2[0, 0] = conf_mat[1, 1]
    conf_mat_2[1, 0] = conf_mat[0, 1]
    conf_mat_2[0, 1] = conf_mat[1, 0]

    print(conf_mat)
    print(conf_mat_2[1, 0] + conf_mat_2[0, 1])
    print(oa)
    print(kappa_co)

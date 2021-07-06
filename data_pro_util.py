import numpy as np


def norm_img(img, channel_first=True):
    if channel_first:
        channel, img_height, img_width = img.shape
        img = np.reshape(img, (channel, img_height * img_width))  # (channel, height * width)
        max_value = np.max(img, axis=1, keepdims=True)  # (channel, 1)
        min_value = np.min(img, axis=1, keepdims=True)  # (channel, 1)
        diff_value = max_value - min_value
        nm_img = (img - min_value) / diff_value
        nm_img = np.reshape(nm_img, (channel, img_height, img_width))
    else:
        img_height, img_width, channel = img.shape
        img = np.reshape(img, (img_height * img_width, channel))  # (channel, height * width)
        max_value = np.max(img, axis=0, keepdims=True)  # (channel, 1)
        min_value = np.min(img, axis=0, keepdims=True)  # (channel, 1)
        diff_value = max_value - min_value
        nm_img = (img - min_value) / diff_value
        nm_img = np.reshape(nm_img, (img_height, img_width, channel))
    return nm_img


def norm_img_2(img, channel_first=True):
    if channel_first:
        channel, img_height, img_width = img.shape
        img = np.reshape(img, (channel, img_height * img_width))  # (channel, height * width)
        max_value = np.max(img, axis=1, keepdims=True)  # (channel, 1)
        min_value = np.min(img, axis=1, keepdims=True)  # (channel, 1)
        diff_value = max_value - min_value
        nm_img = 2 * ((img - min_value) / diff_value - 0.5)
        nm_img = np.reshape(nm_img, (channel, img_height, img_width))
    else:
        img_height, img_width, channel = img.shape
        img = np.reshape(img, (img_height * img_width, channel))  # (channel, height * width)
        max_value = np.max(img, axis=0, keepdims=True)  # (channel, 1)
        min_value = np.min(img, axis=0, keepdims=True)  # (channel, 1)
        diff_value = max_value - min_value
        nm_img = 2 * ((img - min_value) / diff_value - 0.5)
        nm_img = np.reshape(nm_img, (img_height, img_width, channel))
    return nm_img


def stad_img(img, channel_first=True):
    """
    normalization image
    :param channel_first:
    :param img: (C, H, W)
    :return:
        norm_img: (C, H, W)
    """
    if channel_first:
        channel, img_height, img_width = img.shape
        img = np.reshape(img, (channel, img_height * img_width))  # (channel, height * width)
        mean = np.mean(img, axis=1, keepdims=True)  # (channel, 1)
        center = img - mean  # (channel, height * width)
        var = np.sum(np.power(center, 2), axis=1, keepdims=True) / (img_height * img_width)  # (channel, 1)
        std = np.sqrt(var)  # (channel, 1)
        nm_img = center / std  # (channel, height * width)
        nm_img = np.reshape(nm_img, (channel, img_height, img_width))
    else:
        img_height, img_width, channel = img.shape
        img = np.reshape(img, (img_height * img_width, channel))  # (height * width, channel)
        mean = np.mean(img, axis=0, keepdims=True)  # (1, channel)
        center = img - mean  # (height * width, channel)
        var = np.sum(np.power(center, 2), axis=0, keepdims=True) / (img_height * img_width)  # (1, channel)
        std = np.sqrt(var)  # (channel, 1)
        nm_img = center / std  # (channel, height * width)
        nm_img = np.reshape(nm_img, (img_height, img_width, channel))
    return nm_img


def random_select_samples(img_X, img_Y, n_train, patch_sz):
    '''
        randomly selecting patches as training sample for KPCA convolution training
    '''
    height, width, channel = img_X.shape
    edge = patch_sz // 2

    img_X = np.pad(img_X, ((edge, edge), (edge, edge), (0, 0)), 'constant')
    img_Y = np.pad(img_Y, ((edge, edge), (edge, edge), (0, 0)), 'constant')

    patch_X = []
    patch_Y = []

    for i in range(0, height):
        for j in range(0, width):
            patch_X.append(img_X[i:i + patch_sz, j:j + patch_sz])
            patch_Y.append(img_Y[i:i + patch_sz, j:j + patch_sz])
    patch_X = np.array(patch_X, np.float32)
    patch_Y = np.array(patch_Y, np.float32)

    train_idx = np.arange(0, height * width)
    np.random.seed()
    np.random.shuffle(train_idx)
    train_idx = train_idx[0:n_train]

    train_X = patch_X[train_idx]
    train_Y = patch_Y[train_idx]
    return train_X, train_Y

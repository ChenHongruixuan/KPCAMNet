import numpy as np
from sklearn.decomposition import KernelPCA


class KernelPCANet(object):
    def __init__(self, num_stages, patch_size, num_filters, gamma):
        self.num_stages = num_stages
        self.patch_size = patch_size
        self.num_filters = num_filters

        self.filters = []
        self.gamma = gamma

    def train_net(self, input_data, stage, is_mean_removal, kernel='rbf'):
        return self.train_filters(input_data, stage, is_mean_removal, kernel)

    def train_filters(self, input_data, stage, is_mean_removal, kernel):
        input_shape = input_data.shape  # (B, m, n, c)
        # generate overlap patch
        print('-------patch generation-------')
        overlap_patch = self._generate_over_patch(input_data)

        overlap_patch = np.reshape(overlap_patch,  # (m * n, c * k1 * k2)
                                   (-1, self.patch_size[0] * self.patch_size[1] * input_shape[-1]))

        # overlap_patch = input_data
        # mean removal
        #  print('-------mean removal-------')
        if is_mean_removal:
            patch_mean = np.mean(overlap_patch, axis=1, keepdims=True)  # (m * n * c, 1)
            mean_overlap_patch = overlap_patch - patch_mean  # (m * n * c, k1 * k2)
        else:
            mean_overlap_patch = overlap_patch

        print('-------solve KPCA problem-------')
        KPCA_trans = KernelPCA(n_components=self.num_filters[stage], kernel=kernel, degree=3, gamma=self.gamma[stage])
        output = KPCA_trans.fit_transform(mean_overlap_patch)

        self.filters.append(KPCA_trans)
        return output

    def infer_data(self, input_data, stage, is_mean_removal):
        output_data = self.predict(input_data, stage, is_mean_removal)
        return output_data

    def predict(self, data, stage, is_mean_removal):
        input_shape = data.shape
        mar_ver = self.patch_size[0] // 2
        mar_hor = self.patch_size[1] // 2
        padding_img = np.zeros(
            (input_shape[0],
             input_shape[1] + 2 * mar_ver,
             input_shape[2] + 2 * mar_hor,
             input_shape[3]))
        for i in range(input_shape[0]):  # (B, m, n, c) --> (B, m+filter_h, n+filter_w, c)
            padding_img[i] = np.pad(data[i], ((mar_ver, mar_hor), (mar_ver, mar_hor), (0, 0)), 'constant')

        # print('-------generate overlap patch-------')
        overlap_patch = self._generate_over_patch(padding_img)

        overlap_patch = np.reshape(overlap_patch,  # (m * n, c * k1 * k2)
                                   (-1, self.patch_size[0] * self.patch_size[1] * input_shape[-1]))

        # print('-------mean removal-------')
        if is_mean_removal:
            patch_mean = np.mean(overlap_patch, axis=1, keepdims=True)  # (m * n * c, 1)
            mean_overlap_patch = overlap_patch - patch_mean  # (m * n * c, k1 * k2)
        else:
            mean_overlap_patch = overlap_patch

        KPCA_trains = self.filters[stage]
        trans_output = KPCA_trains.transform(mean_overlap_patch)

        trans_output = np.reshape(trans_output,
                                  (input_shape[0], input_shape[1], input_shape[2], self.num_filters[stage]))
        return trans_output

    def _generate_over_patch(self, data):
        input_shape = data.shape
        mar_ver = self.patch_size[0] // 2
        mar_hor = self.patch_size[1] // 2

        overlap_patch = []
        for batch_id in range(input_shape[0]):
            for i in range(mar_ver, input_shape[1] - mar_ver):
                for j in range(mar_hor, input_shape[2] - mar_hor):
                    # (B, k1, k2, c)
                    patch = data[batch_id, (i - mar_ver):(i + mar_ver + 1), (j - mar_hor):(j + mar_hor + 1)]
                    overlap_patch.append(patch)
        overlap_patch = np.reshape(overlap_patch, (-1, self.patch_size[0], self.patch_size[1], input_shape[-1]))
        return overlap_patch

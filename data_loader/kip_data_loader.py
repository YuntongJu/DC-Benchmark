import os
import torch
import numpy as np

class KIPDataLoader:

    @staticmethod
    def load_data(root_dir, dataset, ipc, data_file):
        data_path = os.path.join(root_dir, "KIP", dataset, 'IPC' + str(ipc), data_file)
        data = np.load(data_path)
        image = torch.from_numpy(data['images']).permute(0, 3, 1, 2)
        label = torch.from_numpy(data['labels'])
        return (image, label)


if __name__ == '__main__':
    
    images, labels = KIPDataLoader.load_data('/nfs/data/justincui/dc_benchmark/distilled_results', 'CIFAR10', 10, 'kip_cifar10_ConvNet_ssize100_zca_nol_noaug_ckpt1000.npz')
    print(images.shape)
    print(labels.shape)
    print(labels.max(), labels.min())
    print(images.max(), images.min())
    # print((labels + 1/10).long())
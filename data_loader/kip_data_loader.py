import os
import torch
import numpy as np

class KIPDataLoader:

    @staticmethod
    def load_data(root_dir, dataset, ipc, data_file):
        image_path = data_file[0]
        label_path = data_file[1]
        image_path = os.path.join(root_dir, "KIP", dataset, 'IPC' + str(ipc), image_path)
        label_path = os.path.join(root_dir, "KIP", dataset, 'IPC' + str(ipc), label_path)
        image = torch.from_numpy(np.load(image_path)).permute(0, 3, 1, 2)
        label = torch.from_numpy(np.load(label_path))
        return (image, label)


if __name__ == '__main__':
    
    images = KIPDataLoader.load_images('/home/justincui/dc_benchmark/dc_benchmark/distilled_results/KIP/CIFAR10/IPC10/images.npy')
    labels = KIPDataLoader.load_labels('/home/justincui/dc_benchmark/dc_benchmark/distilled_results/KIP/CIFAR10/IPC10/labels.npy')
    print(images.shape)
    print(labels.shape)
    print(labels)
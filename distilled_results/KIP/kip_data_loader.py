import torch
import numpy as np

class KIPDataLoader:

    @staticmethod
    def load_images(path_to_pt_file):
        return np.load(path_to_pt_file)

    @staticmethod
    def load_labels(path_to_pt_file):
        return torch.load(path_to_pt_file)


if __name__ == '__main__':
    
    images = KIPDataLoader.load_images('/home/justincui/dc_benchmark/dc_benchmark/distilled_results/KIP/CIFAR10/IPC10/images.npy')
    labels = KIPDataLoader.load_images('/home/justincui/dc_benchmark/dc_benchmark/distilled_results/KIP/CIFAR10/IPC10/labels.npy')
    print(images.shape)
    print(labels.shape)
    print(labels)
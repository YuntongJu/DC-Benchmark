import os
import torch

class KMeansDataLoader:

    @staticmethod
    def load_data(root_dir, dataset, ipc, data_file):
        image_path = data_file[0]
        label_path = data_file[1]
        image_path = os.path.join(root_dir, "Kmeans", dataset, 'IPC' + str(ipc), image_path)
        label_path = os.path.join(root_dir, "Kmeans", dataset, 'IPC' + str(ipc), label_path)
        training_images = torch.load(image_path)
        training_lables = torch.load(label_path)
        return training_images, training_lables
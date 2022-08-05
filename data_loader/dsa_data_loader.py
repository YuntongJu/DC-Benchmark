import os
import torch

class DSADataLoader:

    @staticmethod
    def load_data(root_dir, dataset, ipc, data_file):
        data_path = os.path.join(root_dir, "DSA", dataset, 'IPC' + str(ipc), data_file)
        dsa_data = torch.load(data_path)
        training_data = dsa_data['data']
        train_images, train_labels = training_data[-1]
        return train_images, train_labels
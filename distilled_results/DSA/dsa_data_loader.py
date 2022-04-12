import torch

class DSADataLoader:

    @staticmethod
    def load_data(path_to_pt_file):
        dm_data = torch.load(path_to_pt_file)
        training_data = dm_data['data']
        train_images, train_labels = training_data[-1]
        return train_images, train_labels


if __name__ == '__main__':
    train_images, train_labels = DSADataLoader.load_data('/home/justincui/dc_benchmark/dc_benchmark/distilled_results/DC/CIFAR10/IPC10/res_DC_CIFAR10_ConvNet_10ipc.pt')
    print(train_images.shape)
    print(train_labels.shape)
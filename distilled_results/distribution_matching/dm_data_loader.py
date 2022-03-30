import torch

class DMDataLoader:

    @staticmethod
    def load_data(path_to_pt_file):
        dm_data = torch.load(path_to_pt_file)
        print(dm_data.shape)
        training_data = dm_data['data']
        train_images, train_labels = training_data[-1]
        return train_images, train_labels


if __name__ == '__main__':
    train_images, train_labels = DMDataLoader.load_data('/home/justincui/dc_benchmark/dc_benchmark/distilled_results/distribution_matching/vis_DM_CIFAR10_ConvNet_1200ipc_exp0_iter20000.pt')
    print(train_images.shape)
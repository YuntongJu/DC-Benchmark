import torch

class KMeansDataLoader:

    @staticmethod
    def load_data(path_to_images, path_to_label):
        training_images = torch.load(path_to_images)
        training_lables = torch.load(path_to_label)
        return training_images, training_lables


if __name__ == '__main__':
    train_images, train_labels = KMeansDataLoader.load_data('/home/justincui/dc_benchmark/dc_benchmark/distilled_results/Kmeans/CIFAR10/IPC1/CIFAR10_IPC1_images.pt', '/home/justincui/dc_benchmark/dc_benchmark/distilled_results/Kmeans/CIFAR10/IPC1/CIFAR10_IPC1_labels.pt')
    print(train_images.shape)
    print(train_labels.shape)
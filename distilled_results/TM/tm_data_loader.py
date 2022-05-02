import torch

class TMDataLoader:

    @staticmethod
    def load_data(image_path, label_path):
        return torch.load(image_path), torch.load(label_path)


if __name__ == '__main__':

    data_path = '/home/justincui/dc_benchmark/dc_benchmark/distilled_results/TM/CIFAR10/IPC1/'
    images, labels = TMDataLoader.load_data(data_path + 'images_best.pt', data_path + 'labels_best.pt')
    print(images.shape)
    print(labels.shape)
    print(labels)
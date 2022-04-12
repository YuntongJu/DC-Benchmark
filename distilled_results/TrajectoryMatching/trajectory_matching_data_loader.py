import torch

class TMDataLoader:

    @staticmethod
    def load_images(path_to_pt_file):
        return torch.load(path_to_pt_file)

    @staticmethod
    def load_labels(path_to_pt_file):
        return torch.load(path_to_pt_file)


if __name__ == '__main__':
    
    images = TMDataLoader.load_images('/home/justincui/dc_benchmark/dc_benchmark/distilled_results/TrajectoryMatching/CIFAR10/IPC10/images_best.pt')
    labels = TMDataLoader.load_labels('/home/justincui/dc_benchmark/dc_benchmark/distilled_results/TrajectoryMatching/CIFAR10/IPC10/labels_best.pt')
    print(images.shape)
    print(labels.shape)
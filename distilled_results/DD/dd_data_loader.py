import torch

class TMDataLoader:

    @staticmethod
    def load_images(path_to_pt_file):
        return torch.load(path_to_pt_file)

    @staticmethod
    def load_labels(path_to_pt_file):
        return torch.load(path_to_pt_file)


if __name__ == '__main__':
    
    images = TMDataLoader.load_images('/home/justincui/dc_benchmark/dc_benchmark/distilled_results/DD/CIFAR10/IPC10/results.pth')
    for image in images:
        print(image[0].shape, image[1].shape, image[2])
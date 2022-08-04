import torch
from torchvision.utils import save_image

class RandomDataLoader:

    @staticmethod
    def load_data(path_to_images, path_to_label):
        training_images = torch.load(path_to_images)
        training_lables = torch.load(path_to_label)
        return training_images, training_lables


if __name__ == '__main__':
    # train_images, train_labels = RandomDataLoader.load_data('/home/justincui/dc_benchmark/dc_benchmark/distilled_results/random/CIFAR10/IPC1/CIFAR10_IPC1_images.pt', '/home/justincui/dc_benchmark/dc_benchmark/distilled_results/random/CIFAR10/IPC1/CIFAR10_IPC1_labels.pt')
    # print(train_images.shape)
    # print(train_labels.shape)
    # covnert to image.
    data = torch.load("/nfs/data/justincui/dc_benchmark/distilled_results/random/tinyimagenet/IPC10/tinyimagenet_IPC10_normalize_images.pt")
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image_syn_vis = data
    for ch in range(3):
        image_syn_vis[:, ch] = image_syn_vis[:, ch]  * std[ch] + mean[ch]
    print(image_syn_vis.max())
    print(image_syn_vis.min())
    save_image(image_syn_vis, "test.png", nrow=10) # Trying normalize = True/False may get better visual effects.

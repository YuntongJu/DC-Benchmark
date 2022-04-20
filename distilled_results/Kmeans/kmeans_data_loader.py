import torch
from torchvision import datasets, transforms

class KMeansDataLoader:

    @staticmethod
    def load_data():
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = datasets.CIFAR10('data', train=True, download=True, transform=transform)
        print(len(dataset))



if __name__ == '__main__':
    train_images, train_labels = KMeansDataLoader.load_data()
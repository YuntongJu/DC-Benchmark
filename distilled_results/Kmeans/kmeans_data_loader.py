import torch
from torchvision import datasets, transforms
from sklearn.cluster import KMeans
import numpy as np

class KMeansDataLoader:

    @staticmethod
    def load_data():
        transform = transforms.Compose([transforms.ToTensor()])
        # transform = transforms.Compose([])
        dataset = datasets.CIFAR10('data', train=True, download=True, transform=transform)
        data_map = {}
        for i in range(10):
            data_map[i] = []
        for data in dataset:
            data_map[data[1]].append(data[0].resize(3 * 32 * 32).numpy())
        # Find cluster centers using KMeans.
        images = []
        labels = []
        for key in data_map:
            X = np.array(data_map[key])
            print(X.shape)
            kmeans = KMeans(n_clusters=10, random_state=0).fit(X)
            for center in kmeans.cluster_centers_:
                images.append(torch.unsqueeze(torch.from_numpy(center).resize(3, 32, 32), dim=0)) 
                labels.append(key)
        images = torch.cat(images)
        labels = torch.Tensor(labels)
        return images, labels



if __name__ == '__main__':
    images, labels = KMeansDataLoader.load_data()
    print(images.shape)
    print(labels.shape)
    
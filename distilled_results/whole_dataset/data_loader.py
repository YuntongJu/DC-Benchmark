
import torch
from torchvision import datasets, transforms

class WholeDataLoader:

    @staticmethod
    def load_data(dataset_name):
        transform = transforms.Compose([transforms.ToTensor()])
        if dataset_name == 'cifar10':
            ds_train = datasets.CIFAR10('data', train=True, download=True, transform=transform)
        elif dataset_name == 'cifar100':
            ds_train = datasets.CIFAR100('data', train=True, download=True, transform=transform)
        elif dataset_name == 'tinyimagenet':
            images_all = [torch.unsqueeze(ds_train[i][0], dim=0) for i in range(len(ds_train))]
            labels_all = [ds_train[i][1] for i in range(len(ds_train))]

        images_all = torch.cat(images_all, dim=0)
        labels_all = torch.tensor(labels_all, dtype=torch.long)
        return images_all, labels_all






if __name__ == '__main__':
    
    images, labels = WholeDataLoader.load_data('cifar10')
    print(images.shape)
    print(labels.shape)
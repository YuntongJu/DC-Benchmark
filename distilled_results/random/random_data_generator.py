import sys
sys.path.append('/home/justincui/dc_benchmark/dc_benchmark')

import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import argparse
import os




class TensorDataset(Dataset):
    def __init__(self, images, labels): # images: n x c x h x w tensor
        self.images = images.detach().float()
        self.labels = labels.detach()

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return self.images.shape[0]

class RandomDataGenerator:

    @staticmethod
    def prepare_args():
        parser = argparse.ArgumentParser(description='Parameter Processing')
        parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
        parser.add_argument('--ipc', type=int, default=50, help='image(s) per class')
        parser.add_argument('--normalize_data', action="store_true", help='whether to normalize the data') # S: the same to training model, M: multi architectures,  W: net width, D: net depth, A: activation function, P: pooling layer, N: normalization layer,
        args = parser.parse_args()
        args.device = 'cuda'
        return args

    @staticmethod
    def load_data(args):
        # set random seed
        #seeting seed to 9 gets us 0.3368 accuracy.
        if args.normalize_data:
            mean = [0.4914, 0.4822, 0.4465]
            std = [0.2023, 0.1994, 0.2010]
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        else:
            transform = transforms.Compose([transforms.ToTensor()])
        dst_train = datasets.CIFAR10('data', train=True, download=True, transform=transform)

        if args.dataset == 'CIFAR10':
            num_classes = 10
        elif args.dataset == 'CIFAR100':
            num_classes = 100
        elif args.dataset == 'tinyimagenet':
            num_classes = 200

        images_all = []
        labels_all = []
        indices_class = [[] for c in range(num_classes)]

        images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
        labels_all = [dst_train[i][1] for i in range(len(dst_train))]
        for i, lab in enumerate(labels_all):
            indices_class[lab].append(i)
        images_all = torch.cat(images_all, dim=0).to(args.device)
        labels_all = torch.tensor(labels_all, dtype=torch.long, device=args.device)

        def get_images(c, n): # get random n images from class c
                idx_shuffle = np.random.RandomState(seed=42).permutation(indices_class[c])[:n]
                return images_all[idx_shuffle]

        sampled_images = []
        sampled_labels = []
        for i in range(num_classes):
            sampled_images.append(get_images(i, args.ipc))
            sampled_labels.append(torch.ones(args.ipc) * i)
        sampled_images = torch.cat(sampled_images)
        sampled_labels = torch.cat(sampled_labels)
        return sampled_images, sampled_labels




if __name__ == '__main__':
    args = RandomDataGenerator.prepare_args()
    train_image, train_label = RandomDataGenerator.load_data(args)
    output_path = os.getcwd() + "/" + args.dataset + '/IPC' + str(args.ipc) + '/' + args.dataset + '_IPC' + str(args.ipc) + '_'
    if args.normalize_data:
        output_path += 'normalize_'
    torch.save(train_image, output_path + "images.pt")
    torch.save(train_label, output_path + "labels.pt")
    print("finished generating data")
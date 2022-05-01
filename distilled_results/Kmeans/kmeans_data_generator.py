import sys
sys.path.append('/home/justincui/dc_benchmark/dc_benchmark')

import torch
from torchvision import datasets, transforms
from sklearn.cluster import KMeans
import numpy as np
import argparse
import tqdm
import os
from torch.utils.data import Dataset



from evaluator.evaluator_utils import EvaluatorUtils
from networks.network_utils import NetworkUtils


class TensorDataset(Dataset):
    def __init__(self, images, labels): # images: n x c x h x w tensor
        self.images = images.detach().float()
        self.labels = labels.detach()

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return self.images.shape[0]

class KMeansDataLoader:

    @staticmethod
    def prepare_args():
        parser = argparse.ArgumentParser(description='Parameter Processing')
        parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
        parser.add_argument('--model', type=str, default='convnet', help='model')
        parser.add_argument('--dsa', action='store_true', help='model')
        parser.add_argument('--ipc', type=int, default=10, help='image(s) per class')
        parser.add_argument('--eval_mode', type=str, default='S', help='eval_mode') # S: the same to training model, M: multi architectures,  W: net width, D: net depth, A: activation function, P: pooling layer, N: normalization layer,
        parser.add_argument('--num_eval', type=int, default=20, help='the number of evaluating randomly initialized models')
        parser.add_argument('--optimizer', type=str, default='sgd', help='the number of evaluating randomly initialized models')
        parser.add_argument('--normalize_data', type=bool, default=False, help='the number of evaluating randomly initialized models')
        parser.add_argument('--epoch_eval_train', type=int, default=300, help='epochs to train a model with synthetic data')
        parser.add_argument('--Iteration', type=int, default=1000, help='training iterations')
        parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
        parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
        parser.add_argument('--init', type=str, default='noise', help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')
        parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate', help='differentiable Siamese augmentation strategy')
        args = parser.parse_args()
        args.dc_aug_param = EvaluatorUtils.get_daparam(args.dataset, args.model, '', args.ipc) # This augmentation parameter set is only for DC method. It will be muted when args.dsa is True.
        args.device = 'cuda'
        return args

    @staticmethod
    def load_data(use_embedding=True, normalize_data = False):
        args = KMeansDataLoader.prepare_args()

        output_images_path = os.getcwd() + "/" + args.dataset + '/IPC' + str(args.ipc) + '/' + args.dataset + '_IPC' + str(args.ipc) + '_' + 'images.pt'
        output_labels_path = os.getcwd() + "/" + args.dataset + '/IPC' + str(args.ipc) + '/' + args.dataset + '_IPC' + str(args.ipc) + '_' + 'labels.pt'
        if os.path.exists(output_images_path) or os.path.exists(output_labels_path):
            exit("file already exisits")

        args.epoch_eval_train = 0

        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        if normalize_data:
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        else:
            transform = transforms.Compose([transforms.ToTensor()])
        ds_train = datasets.CIFAR10('data', train=True, download=True, transform=transform)
        ds_test = datasets.CIFAR10('data', train=False, download=True, transform=transform)

        if use_embedding:
            print("use embedding")
            if args.dsa:
                args.dsa_param = EvaluatorUtils.ParamDiffAug()
                args.epoch_eval_train = 1000
                args.dc_aug_param = None
            images_all = [torch.unsqueeze(ds_train[i][0], dim=0) for i in range(len(ds_train))]
            labels_all = [ds_train[i][1] for i in range(len(ds_train))]

            images_all = torch.cat(images_all, dim=0)
            labels_all = torch.tensor(labels_all, dtype=torch.long)
            net = NetworkUtils.create_network(args.model).to(args.device)
            testloader = torch.utils.data.DataLoader(ds_test, batch_size=256, shuffle=False, num_workers=0)
            net, _, _ = EvaluatorUtils.evaluate_synset(0, net, images_all, labels_all, testloader, args)

        feature_map = {}
        data_map = {}
        embed = net.embed

        for i in range(10):
            feature_map[i] = []
            data_map[i] = []
        for data in ds_train:
            data_map[data[1]].append(data[0])
            if use_embedding:
                feature_map[data[1]].append(embed(torch.unsqueeze(data[0].to(args.device), dim=0)).squeeze().cpu().detach().numpy())
            else:
                feature_map[data[1]].append(data[0].resize(3 * 32 * 32).numpy())

        # Find cluster centers using KMeans.
        images = []
        labels = []
        for key in feature_map:
            X = np.array(feature_map[key])
            kmeans = KMeans(n_clusters=args.ipc, random_state=0, init='k-means++').fit(X)
            dist = (kmeans.cluster_centers_[:, np.newaxis] - X)
            dist = dist ** 2
            dist = np.sum(dist, axis=2)
            dist = dist ** (0.5)
            knn_indices = np.argmin(dist, axis=1)
            print(knn_indices.shape)
            print(knn_indices)
            for index in knn_indices:
                images.append(torch.unsqueeze(data_map[key][index], dim=0))
                labels.append(key)
        images = torch.cat(images)
        labels = torch.Tensor(labels)
        with open(output_images_path, "wb") as f:
            torch.save(images, f)
        with open(output_labels_path, "wb") as f:
            torch.save(images, f)
        print("generate data ready")
        return images, labels

if __name__ == '__main__':
    images, labels = KMeansDataLoader.load_data(1)
    print(images.shape)
    print(labels.shape)

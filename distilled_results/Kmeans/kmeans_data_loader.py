import sys
sys.path.append('../../../dc_benchmark')

import torch
from torchvision import datasets, transforms
from sklearn.cluster import KMeans
import numpy as np
import argparse
import os

from evaluator.evaluator_utils import EvaluatorUtils
from networks.network_utils import NetworkUtils
from torchvision.utils import save_image


class KMeansDataLoader:

    @staticmethod
    def prepare_args():
        parser = argparse.ArgumentParser(description='Parameter Processing')
        parser.add_argument('--method', type=str, default='DC', help='DC/DSA')
        parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
        parser.add_argument('--model', type=str, default='convnet', help='model')
        parser.add_argument('--ipc', type=int, default=50, help='image(s) per class')
        parser.add_argument('--eval_mode', type=str, default='S', help='eval_mode') # S: the same to training model, M: multi architectures,  W: net width, D: net depth, A: activation function, P: pooling layer, N: normalization layer,
        parser.add_argument('--normalize_data', action="store_true", help='whether to normalize the data') # S: the same to training model, M: multi architectures,  W: net width, D: net depth, A: activation function, P: pooling layer, N: normalization layer,
        parser.add_argument('--num_eval', type=int, default=20, help='the number of evaluating randomly initialized models')
        parser.add_argument('--epoch_eval_train', type=int, default=50, help='epochs to train a model with synthetic data')
        parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
        parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
        parser.add_argument('--optimizer', type=str, default='sgd', help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')
        parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate', help='differentiable Siamese augmentation strategy')
        args = parser.parse_args()
        args.dsa = False
        args.dc_aug_param = EvaluatorUtils.get_daparam(args.dataset, args.model, '', args.ipc) # This augmentation parameter set is only for DC method. It will be muted when args.dsa is True.
        args.device = 'cuda'
        return args

    @staticmethod
    def load_data(args, use_embedding=True, normalize_data = True):
        if args.dataset == 'CIFAR10':
            num_classes = 10
            mean = [0.4914, 0.4822, 0.4465]
            std = [0.2023, 0.1994, 0.2010]
        elif args.dataset == 'CIFAR100':
            num_classes = 100
            mean = [0.5071, 0.4866, 0.4409]
            std = [0.2673, 0.2564, 0.2762]
        elif args.dataset == 'tinyimagenet':
            num_classes = 200
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]

        if normalize_data:
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        else:
            transform = transforms.Compose([transforms.ToTensor()])
        if args.dataset == 'CIFAR10':
            ds_train = datasets.CIFAR10('data', train=True, download=True, transform=transform)
            ds_test = datasets.CIFAR10('data', train=False, download=True, transform=transform)
        elif args.dataset == 'CIFAR100':
            ds_train = datasets.CIFAR100('data', train=True, download=True, transform=transform)
            ds_test = datasets.CIFAR100('data', train=False, download=True, transform=transform)
        elif args.dataset == 'tinyimagenet':
            ds_train = datasets.ImageFolder(os.path.join('/home/justincui/tiny-imagenet-200', "train"), transform=transform)
            ds_test= datasets.ImageFolder(os.path.join('/home/justincui/tiny-imagenet-200', "val", "images"), transform=transform)

        
        if use_embedding:
            print("use embedding")
            args = KMeansDataLoader.prepare_args()
            args.epoch_eval_train = 0
            args.dsa = False
            # args.model = "convnet"
            print("embedding model", args.model)
            net = NetworkUtils.create_network(args.model, args.dataset).to(args.device)
            images_all = [torch.unsqueeze(ds_train[i][0], dim=0) for i in range(len(ds_train))]
            labels_all = [ds_train[i][1] for i in range(len(ds_train))]

            images_all = torch.cat(images_all, dim=0)
            labels_all = torch.tensor(labels_all, dtype=torch.long)
            # if os.path.exists('data/' + model_name):
            #     net.load_state_dict(torch.load('data/' + model_name))
            # else:
            testloader = torch.utils.data.DataLoader(ds_test, batch_size=256, shuffle=False, num_workers=0)
            print("----------being training the embedding model: {}-------".format(args.model))
            net, _, _= EvaluatorUtils.evaluate_synset(0, net, images_all, labels_all, testloader, args)
            print("----------end training the embedding model: {}-------".format(args.model))

        feature_map = {}
        data_map = {}
        if use_embedding:
            embed = net.embed

        for i in range(num_classes):
            feature_map[i] = []
            data_map[i] = []
        for data in ds_train:
            data_map[data[1]].append(data[0])
            if use_embedding:
                feature_map[data[1]].append(embed(torch.unsqueeze(data[0].to(args.device), dim=0)).squeeze().cpu().detach().numpy())
            else:
                if args.dataset == 'tinyimagenet':
                    feature_map[data[1]].append(data[0].resize(3 * 64 * 64).numpy())
                else:
                    feature_map[data[1]].append(data[0].resize(3 * 32 * 32).numpy())

        # Find cluster centers using KMeans.
        images = []
        labels = []
        for key in feature_map:
            X = np.array(feature_map[key])
            # print(X.shape)
            # find the kmeans center
            # random state = 1, accurac = 0.0101 convnet

            kmeans = KMeans(n_clusters=args.ipc, init='k-means++', n_init=50).fit(X)
            # kmeans = KMedoids(n_clusters=ipc, random_state=0, init='k-medoids++').fit(X)
            # find the samples that are closest to the kmeans center
            # print(kmeans.cluster_centers_.shape)
            dist = (kmeans.cluster_centers_[:, np.newaxis] - X)
            # print(dist.shape)
            dist = dist ** 2
            dist = np.sum(dist, axis=2)
            # print(dist.shape)
            dist = dist ** (0.5)
            # print(dist.shape)
            # knn_indices = dist.topk(1, largest=False, sorted=False)
            knn_indices = np.argmin(dist, axis=1)
            print(knn_indices.shape)
            print(knn_indices)
            for index in knn_indices:
                images.append(torch.unsqueeze(data_map[key][index], dim=0))
                labels.append(key)
        images = torch.cat(images)
        labels = torch.Tensor(labels)
        print("load data ready")
        return images, labels, ds_test



if __name__ == '__main__':
    data = torch.load("/home/justincui/dc_benchmark/distilled_results/Kmeans/tinyimagenet_1_0.0101_images.pt")
    print(data.max())
    print(data.min())
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image_syn_vis = data
    for ch in range(3):
        image_syn_vis[:, ch] = image_syn_vis[:, ch]  * std[ch] + mean[ch]
        # image_syn_vis[image_syn_vis<0] = 0.0
        # image_syn_vis[image_syn_vis>1] = 1.0
    print(image_syn_vis.max())
    print(image_syn_vis.min())
    save_image(image_syn_vis, "test.png", nrow=1) # Trying normalize = True/False may get better visual effects.

    
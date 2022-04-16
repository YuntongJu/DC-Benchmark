import sys
sys.path.append('/home/justincui/dc_benchmark/dc_benchmark')

from evaluator.evaluator_utils import EvaluatorUtils
import torch
import numpy as np
from torchvision import datasets, transforms

class RandomDataLoader:

    @staticmethod
    def load_data(args):
        # set random seed
        #seeting seed to 9 gets us 0.3368 accuracy.

        transform = transforms.Compose([transforms.ToTensor()])
        dst_train = datasets.CIFAR10('data', train=True, download=True, transform=transform)
        # dst_train, _ = EvaluatorUtils.get_cifar10_testset(args)

        num_classes = 10
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
        for i in range(10):
            sampled_images.append(get_images(i, 10))
            sampled_labels.append(torch.ones(10) * i)
        sampled_images = torch.cat(sampled_images)
        sampled_labels = torch.cat(sampled_labels)
        return sampled_images, sampled_labels
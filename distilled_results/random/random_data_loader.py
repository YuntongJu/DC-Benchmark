import sys
sys.path.append('/home/justincui/dc_benchmark/dc_benchmark')

from evaluator.evaluator_utils import EvaluatorUtils
import torch
import numpy as np
from torchvision import datasets, transforms
import tqdm
from torch.utils.data import Dataset
import kornia as K




class TensorDataset(Dataset):
    def __init__(self, images, labels): # images: n x c x h x w tensor
        self.images = images.detach().float()
        self.labels = labels.detach()

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return self.images.shape[0]

class RandomDataLoader:

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

        if args.zca:
            images = []
            labels = []
            print("Train ZCA")
            for i in tqdm.tqdm(range(len(dst_train))):
                im, lab = dst_train[i]
                images.append(im)
                labels.append(lab)
            images = torch.stack(images, dim=0).to(args.device)
            labels = torch.tensor(labels, dtype=torch.long, device="cpu")
            zca = K.enhance.ZCAWhitening(eps=0.1, compute_inv=True)
            zca.fit(images)
            zca_images = zca(images).to("cpu")
            dst_train = TensorDataset(zca_images, labels)

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
        for i in range(num_classes):
            sampled_images.append(get_images(i, args.ipc))
            sampled_labels.append(torch.ones(args.ipc) * i)
        sampled_images = torch.cat(sampled_images)
        sampled_labels = torch.cat(sampled_labels)
        return sampled_images, sampled_labels
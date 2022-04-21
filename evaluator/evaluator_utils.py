from multiprocessing import reduction
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import tqdm
import kornia as K
from torch.utils.data import Dataset


class TensorDataset(Dataset):
    def __init__(self, images, labels): # images: n x c x h x w tensor
        self.images = images.detach().float()
        self.labels = labels.detach()

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return self.images.shape[0]

def get_time():
    return str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))


class EvaluatorUtils:

    class ParamDiffAug():
        def __init__(self):
            self.aug_mode = 'S' #'multiple or single'
            self.prob_flip = 0.5
            self.ratio_scale = 1.2
            self.ratio_rotate = 15.0
            self.ratio_crop_pad = 0.125
            self.ratio_cutout = 0.5 # the size would be 0.5x0.5
            self.brightness = 1.0
            self.saturation = 2.0
            self.contrast = 0.5
            
    @staticmethod
    def evaluate_synset(it_eval, net, images_train, labels_train, testloader, args):
        net = net.to(args.device)
        images_train = images_train.to(args.device)
        labels_train = labels_train.to(args.device)
        lr = float(args.lr_net)
        Epoch = int(args.epoch_eval_train)
        lr_schedule = [Epoch//2+1]
        if args.optimizer == 'adam':
            print("using adam optimizer")
            optimizer = torch.optim.Adam(net.parameters())
        else:
            print("using sgd optimizer")
            optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
        if hasattr(args, 'sample_weights'):
            criterion = nn.CrossEntropyLoss(reduction='none').to(args.device)
        else:
            criterion = nn.CrossEntropyLoss().to(args.device)

        dst_train = TensorDataset(images_train, labels_train)
        trainloader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_train, shuffle=True, num_workers=0)

        start = time.time()
        for ep in range(Epoch+1):
            loss_train, acc_train = EvaluatorUtils.epoch('train', trainloader, net, optimizer, criterion, args, aug = True, ep=ep)
            if ep in lr_schedule:
                lr *= 0.1
                if args.optimizer == 'adam':
                    print("using adam optimizer")
                    # optimizer = torch.optim.Adam(net.parameters(), lr=lr)
                    pass
                else:
                    print("using sgd optimizer")
                    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)

        time_train = time.time() - start
        criterion = nn.CrossEntropyLoss().to(args.device)
        loss_test, acc_test = EvaluatorUtils.epoch('test', testloader, net, optimizer, criterion, args, aug = False, ep=0)
        print('%s Evaluate_%02d: epoch = %04d train time = %d s train loss = %.6f train acc = %.4f, test acc = %.4f' % (get_time(), it_eval, Epoch, int(time_train), loss_train, acc_train, acc_test))

        return net, acc_train, acc_test

    @staticmethod
    def epoch(mode, dataloader, net, optimizer, criterion, args, aug, ep):
        loss_avg, acc_avg, num_exp = 0, 0, 0
        net = net.to(args.device)
        criterion = criterion.to(args.device)

        if mode == 'train':
            net.train()
        else:
            net.eval()

        for i_batch, datum in enumerate(dataloader):
            img = datum[0].float().to(args.device)
            if aug:
                if args.dsa:
                    img = EvaluatorUtils.DiffAugment(img, args.dsa_strategy, param=args.dsa_param)
                elif hasattr(args, 'autoaug') and args.autoaug:
                    print("using autoaug")
                    img = EvaluatorUtils.autoaug(img)
                else:
                    img = EvaluatorUtils.augment(img, args.dc_aug_param, device=args.device)
            lab = datum[1].long().to(args.device)
            n_b = lab.shape[0]
            # print("-----------", mode, img.max(), img.min())
            output = net(img)
            loss = criterion(output, lab)
            if mode =='train' and hasattr(args, 'sample_weights'):
                loss = (loss * args.sample_weights).mean()
            acc = np.sum(np.equal(np.argmax(output.cpu().data.numpy(), axis=-1), lab.cpu().data.numpy()))

            loss_avg += loss.item()*n_b
            acc_avg += acc
            num_exp += n_b

            if mode == 'train':
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        loss_avg /= num_exp
        acc_avg /= num_exp
        print(mode, " epoch:", ep, " , accuracy is:", acc_avg)
        return loss_avg, acc_avg

    @staticmethod
    def autoaug(images):
        image_syn_vis = images
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        for ch in range(3):
            image_syn_vis[:, ch] = image_syn_vis[:, ch]  * std[ch] + mean[ch]
        image_syn_vis[image_syn_vis<0] = 0.0
        image_syn_vis[image_syn_vis>1] = 1.0
        # image_syn_vis = image_syn_vis * 255
        # data_transforms = transforms.Compose([transforms.AutoAugment()])
        # image_syn_vis = data_transforms(image_syn_vis.to(torch.uint8))
        # image_syn_vis = image_syn_vis / 255.0
        for ch in range(3):
            image_syn_vis[:, ch] = (image_syn_vis[:, ch] - mean[ch])  / std[ch]
        return image_syn_vis

    @staticmethod
    def augment(images, dc_aug_param, device):
    # This can be sped up in the future.

        if dc_aug_param != None and dc_aug_param['strategy'] != 'none':
            scale = dc_aug_param['scale']
            crop = dc_aug_param['crop']
            rotate = dc_aug_param['rotate']
            noise = dc_aug_param['noise']
            strategy = dc_aug_param['strategy']

            shape = images.shape
            mean = []
            for c in range(shape[1]):
                mean.append(float(torch.mean(images[:,c])))

            def cropfun(i):
                im_ = torch.zeros(shape[1],shape[2]+crop*2,shape[3]+crop*2, dtype=torch.float, device=device)
                for c in range(shape[1]):
                    im_[c] = mean[c]
                im_[:, crop:crop+shape[2], crop:crop+shape[3]] = images[i]
                r, c = np.random.permutation(crop*2)[0], np.random.permutation(crop*2)[0]
                images[i] = im_[:, r:r+shape[2], c:c+shape[3]]

            def scalefun(i):
                h = int((np.random.uniform(1 - scale, 1 + scale)) * shape[2])
                w = int((np.random.uniform(1 - scale, 1 + scale)) * shape[2])
                tmp = F.interpolate(images[i:i + 1], [h, w], )[0]
                mhw = max(h, w, shape[2], shape[3])
                im_ = torch.zeros(shape[1], mhw, mhw, dtype=torch.float, device=device)
                r = int((mhw - h) / 2)
                c = int((mhw - w) / 2)
                im_[:, r:r + h, c:c + w] = tmp
                r = int((mhw - shape[2]) / 2)
                c = int((mhw - shape[3]) / 2)
                images[i] = im_[:, r:r + shape[2], c:c + shape[3]]

            def rotatefun(i):
                im_ = scipyrotate(images[i].cpu().data.numpy(), angle=np.random.randint(-rotate, rotate), axes=(-2, -1), cval=np.mean(mean))
                r = int((im_.shape[-2] - shape[-2]) / 2)
                c = int((im_.shape[-1] - shape[-1]) / 2)
                images[i] = torch.tensor(im_[:, r:r + shape[-2], c:c + shape[-1]], dtype=torch.float, device=device)

            def noisefun(i):
                images[i] = images[i] + noise * torch.randn(shape[1:], dtype=torch.float, device=device)


            augs = strategy.split('_')

            for i in range(shape[0]):
                choice = np.random.permutation(augs)[0] # randomly implement one augmentation
                if choice == 'crop':
                    cropfun(i)
                elif choice == 'scale':
                    scalefun(i)
                elif choice == 'rotate':
                    rotatefun(i)
                elif choice == 'noise':
                    noisefun(i)

        return images


    @staticmethod
    def get_daparam(dataset, model, model_eval, ipc):
        # We find that augmentation doesn't always benefit the performance.
        # So we do augmentation for some of the settings.

        dc_aug_param = dict()
        dc_aug_param['crop'] = 4
        dc_aug_param['scale'] = 0.2
        dc_aug_param['rotate'] = 45
        dc_aug_param['noise'] = 0.001
        dc_aug_param['strategy'] = 'none'

        if dataset == 'MNIST':
            dc_aug_param['strategy'] = 'crop_scale_rotate'

        if model_eval in ['ConvNetBN']: # Data augmentation makes model training with Batch Norm layer easier.
            dc_aug_param['strategy'] = 'crop_noise'

        return dc_aug_param





    def set_seed_DiffAug(param):
        if param.latestseed == -1:
            return
        else:
            torch.random.manual_seed(param.latestseed)
            param.latestseed += 1


    def DiffAugment(x, strategy='', seed = -1, param = None):
        AUGMENT_FNS = {
            'color': [EvaluatorUtils.rand_brightness, EvaluatorUtils.rand_saturation, EvaluatorUtils.rand_contrast],
            'crop': [EvaluatorUtils.rand_crop],
            'cutout': [EvaluatorUtils.rand_cutout],
            'flip': [EvaluatorUtils.rand_flip],
            'scale': [EvaluatorUtils.rand_scale],
            'rotate': [EvaluatorUtils.rand_rotate],
        }
        if strategy == 'None' or strategy == 'none' or strategy == '':
            return x

        if seed == -1:
            param.Siamese = False
        else:
            param.Siamese = True

        param.latestseed = seed

        if strategy:
            if param.aug_mode == 'M': # original
                for p in strategy.split('_'):
                    for f in AUGMENT_FNS[p]:
                        x = f(x, param)
            elif param.aug_mode == 'S':
                pbties = strategy.split('_')
                EvaluatorUtils.set_seed_DiffAug(param)
                p = pbties[torch.randint(0, len(pbties), size=(1,)).item()]
                for f in AUGMENT_FNS[p]:
                    x = f(x, param)
            else:
                exit('unknown augmentation mode: %s'%param.aug_mode)
            x = x.contiguous()
        return x


    # We implement the following differentiable augmentation strategies based on the code provided in https://github.com/mit-han-lab/data-efficient-gans.
    def rand_scale(x, param):
        # x>1, max scale
        # sx, sy: (0, +oo), 1: orignial size, 0.5: enlarge 2 times
        ratio = param.ratio_scale
        EvaluatorUtils.set_seed_DiffAug(param)
        sx = torch.rand(x.shape[0]) * (ratio - 1.0/ratio) + 1.0/ratio
        EvaluatorUtils.set_seed_DiffAug(param)
        sy = torch.rand(x.shape[0]) * (ratio - 1.0/ratio) + 1.0/ratio
        theta = [[[sx[i], 0,  0],
                [0,  sy[i], 0],] for i in range(x.shape[0])]
        theta = torch.tensor(theta, dtype=torch.float)
        if param.Siamese: # Siamese augmentation:
            theta[:] = theta[0]
        grid = F.affine_grid(theta, x.shape).to(x.device)
        x = F.grid_sample(x, grid)
        return x


    def rand_rotate(x, param): # [-180, 180], 90: anticlockwise 90 degree
        ratio = param.ratio_rotate
        EvaluatorUtils.set_seed_DiffAug(param)
        theta = (torch.rand(x.shape[0]) - 0.5) * 2 * ratio / 180 * float(np.pi)
        theta = [[[torch.cos(theta[i]), torch.sin(-theta[i]), 0],
            [torch.sin(theta[i]), torch.cos(theta[i]),  0],]  for i in range(x.shape[0])]
        theta = torch.tensor(theta, dtype=torch.float)
        if param.Siamese: # Siamese augmentation:
            theta[:] = theta[0]
        grid = F.affine_grid(theta, x.shape).to(x.device)
        x = F.grid_sample(x, grid)
        return x


    def rand_flip(x, param):
        prob = param.prob_flip
        EvaluatorUtils.set_seed_DiffAug(param)
        randf = torch.rand(x.size(0), 1, 1, 1, device=x.device)
        if param.Siamese: # Siamese augmentation:
            randf[:] = randf[0]
        return torch.where(randf < prob, x.flip(3), x)


    def rand_brightness(x, param):
        ratio = param.brightness
        EvaluatorUtils.set_seed_DiffAug(param)
        randb = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
        if param.Siamese:  # Siamese augmentation:
            randb[:] = randb[0]
        x = x + (randb - 0.5)*ratio
        return x


    def rand_saturation(x, param):
        ratio = param.saturation
        x_mean = x.mean(dim=1, keepdim=True)
        EvaluatorUtils.set_seed_DiffAug(param)
        rands = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
        if param.Siamese:  # Siamese augmentation:
            rands[:] = rands[0]
        x = (x - x_mean) * (rands * ratio) + x_mean
        return x


    def rand_contrast(x, param):
        ratio = param.contrast
        x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
        EvaluatorUtils.set_seed_DiffAug(param)
        randc = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
        if param.Siamese:  # Siamese augmentation:
            randc[:] = randc[0]
        x = (x - x_mean) * (randc + ratio) + x_mean
        return x


    def rand_crop(x, param):
        # The image is padded on its surrounding and then cropped.
        ratio = param.ratio_crop_pad
        shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
        EvaluatorUtils.set_seed_DiffAug(param)
        translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
        EvaluatorUtils.set_seed_DiffAug(param)
        translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
        if param.Siamese:  # Siamese augmentation:
            translation_x[:] = translation_x[0]
            translation_y[:] = translation_y[0]
        grid_batch, grid_x, grid_y = torch.meshgrid(
            torch.arange(x.size(0), dtype=torch.long, device=x.device),
            torch.arange(x.size(2), dtype=torch.long, device=x.device),
            torch.arange(x.size(3), dtype=torch.long, device=x.device),
        )
        grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
        grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
        x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
        x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
        return x


    def rand_cutout(x, param):
        ratio = param.ratio_cutout
        cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
        EvaluatorUtils.set_seed_DiffAug(param)
        offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
        EvaluatorUtils.set_seed_DiffAug(param)
        offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
        if param.Siamese:  # Siamese augmentation:
            offset_x[:] = offset_x[0]
            offset_y[:] = offset_y[0]
        grid_batch, grid_x, grid_y = torch.meshgrid(
            torch.arange(x.size(0), dtype=torch.long, device=x.device),
            torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
            torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
        )
        grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
        grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
        mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
        mask[grid_batch, grid_x, grid_y] = 0
        x = x * mask.unsqueeze(1)
        return x

    @staticmethod
    def get_cifar10_testset(args):
        channel = 3
        im_size = (32, 32)
        num_classes = 10
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        if args.zca:
            print("---------------------------used ZCA")
            transform = transforms.Compose([transforms.ToTensor()])
        elif args.normalize_data:
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        else:
            transform = transforms.Compose([transforms.ToTensor()])

        dst_train = datasets.CIFAR10('data', train=True, download=True, transform=transform)
        dst_test = datasets.CIFAR10('data', train=False, download=True, transform=transform)

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

            images = []
            labels = []
            print("Test ZCA")
            for i in tqdm.tqdm(range(len(dst_test))):
                im, lab = dst_test[i]
                images.append(im)
                labels.append(lab)
            images = torch.stack(images, dim=0).to(args.device)
            labels = torch.tensor(labels, dtype=torch.long, device="cpu")

            zca_images = zca(images).to("cpu")
            dst_test = TensorDataset(zca_images, labels)

            args.zca_trans = zca

        return dst_test
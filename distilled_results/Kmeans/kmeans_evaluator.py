import sys
sys.path.append('/home/justincui/dc_benchmark')

import torch
from evaluator.evaluator import Evaluator
from evaluator.evaluator_utils import EvaluatorUtils
from networks.network_utils import NetworkUtils
import argparse
import tqdm
import kornia as K
from torch.utils.data import Dataset
import os
import copy
from torchvision.utils import save_image


class CrossArchEvaluator(Evaluator):

    def __init__(self, input_images, input_labels, test_dataset, config):
        super().__init__(input_images, input_labels, test_dataset)
        self.config = config

    @staticmethod
    def prepare_args():
        parser = argparse.ArgumentParser(description='Parameter Processing')
        parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
        parser.add_argument('--model', type=str, default='convnet', help='model')
        parser.add_argument('--ipc', type=int, default=10, help='image(s) per class')
        parser.add_argument('--eval_mode', type=str, default='S', help='eval_mode') # S: the same to training model, M: multi architectures,  W: net width, D: net depth, A: activation function, P: pooling layer, N: normalization layer,
        parser.add_argument('--num_eval', type=int, default=5, help='the number of evaluating randomly initialized models')
        parser.add_argument('--optimizer', type=str, default='sgd', help='the number of evaluating randomly initialized models')
        parser.add_argument('--normalize_data', action="store_true", help='the number of evaluating randomly initialized models')
        parser.add_argument('--epoch_eval_train', type=int, default=300, help='epochs to train a model with synthetic data')
        parser.add_argument('--Iteration', type=int, default=1000, help='training iterations')
        parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
        parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
        parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate', help='differentiable Siamese augmentation strategy')
        args = parser.parse_args()
        args.dc_aug_param = EvaluatorUtils.get_daparam(args.dataset, args.model, '', args.ipc) # This augmentation parameter set is only for DC method. It will be muted when args.dsa is True.
        args.device = 'cuda'
        return args

    
    def evaluate(self, args):
        if args.dsa:
            args.dsa_param = EvaluatorUtils.ParamDiffAug()
            args.epoch_eval_train = 1000
            args.dc_aug_param = None
        per_arch_accuracy = {}
        for model_name in self.config['models']:
            print("using model", model_name)
            model = NetworkUtils.create_network(model_name, args.dataset)
            _, _, test_acc = EvaluatorUtils.evaluate_synset(0, model, self.input_images, self.input_labels, self.test_dataset, args)
            per_arch_accuracy[model_name] = test_acc
        return per_arch_accuracy

# Evaluation for DSA
if __name__ == '__main__':
    import sys
    sys.path.append('/home/justincui/dc_benchmark/dc_benchmark')
    from distilled_results.Kmeans.kmeans_data_loader import KMeansDataLoader

    args = CrossArchEvaluator.prepare_args()
    # dst_test = EvaluatorUtils.get_testset(args)
    current_best = 0.0
    args.normalize_data = True
    if args.dataset == 'tinyimagenet':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        channel = 3
    while True:
        train_image, train_label, dst_test = KMeansDataLoader.load_data(args, use_embedding=True, normalize_data=True)
        args.zca = False
        args.dsa = True
        args.normalize_data = True
        print(train_image.shape)
        print(train_label.shape)
        print(train_image.max())
        print(train_image.min())
        # args.optimizer = 'adam'
        testloader = torch.utils.data.DataLoader(dst_test, batch_size=256, shuffle=False, num_workers=0)
        evaluator = CrossArchEvaluator(train_image, train_label, testloader, {'models':[args.model]})
        per_arch_accuracy = evaluator.evaluate(args)
        if per_arch_accuracy[args.model] > current_best:
            prev_prefix = '{}_{}_{}_'.format(args.dataset, args.ipc, current_best)
            if os.path.exists(prev_prefix + "images.pt"):
                os.remove(prev_prefix + "images.pt")
            if os.path.exists(prev_prefix + "label.pt"):
                os.remove(prev_prefix + "label.pt")

            current_best = per_arch_accuracy[args.model]
            output_prefix = '{}_{}_{}_'.format(args.dataset, args.ipc, current_best)
            torch.save(train_image,  output_prefix+  'images.pt')
            torch.save(train_label, output_prefix + 'label.pt')
            image_syn_vis = copy.deepcopy(train_image.detach().cpu())
            for ch in range(channel):
                image_syn_vis[:, ch] = image_syn_vis[:, ch]  * std[ch] + mean[ch]
            image_syn_vis[image_syn_vis<0] = 0.0
            image_syn_vis[image_syn_vis>1] = 1.0
            save_image(image_syn_vis, "IPC_" + str(args.ipc) + "_best_image.png", nrow=args.ipc) # Trying normalize = True/False may get better visual effects.
        print("current best accuracy: ", current_best)

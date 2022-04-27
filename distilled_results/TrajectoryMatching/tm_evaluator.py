import sys
sys.path.append('/home/justincui/dc_benchmark/dc_benchmark')

import torch
from evaluator.evaluator import Evaluator
from evaluator.evaluator_utils import EvaluatorUtils
from networks.network_utils import NetworkUtils
import kornia as K
import tqdm
import argparse
from torch.utils.data import Dataset



class CrossArchEvaluator(Evaluator):

    def __init__(self, input_images, input_labels, test_dataset, config):
        super().__init__(input_images, input_labels, test_dataset)
        self.config = config

    @staticmethod
    def prepare_args():
        parser = argparse.ArgumentParser(description='Parameter Processing')
        parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')

        parser.add_argument('--subset', type=str, default='imagenette', help='ImageNet subset. This only does anything when --dataset=ImageNet')

        parser.add_argument('--model', type=str, default='ConvNet', help='model')

        parser.add_argument('--ipc', type=int, default=1, help='image(s) per class')

        parser.add_argument('--eval_mode', type=str, default='S',
                            help='eval_mode, check utils.py for more info')

        parser.add_argument('--num_eval', type=int, default=5, help='how many networks to evaluate on')

        parser.add_argument('--eval_it', type=int, default=100, help='how often to evaluate')

        parser.add_argument('--epoch_eval_train', type=int, default=1000, help='epochs to train a model with synthetic data')
        parser.add_argument('--Iteration', type=int, default=5000, help='how many distillation steps to perform')

        parser.add_argument('--lr_img', type=float, default=1000, help='learning rate for updating synthetic images')
        parser.add_argument('--lr_lr', type=float, default=1e-05, help='learning rate for updating... learning rate')
        parser.add_argument('--lr_teacher', type=float, default=0.01, help='initialization for synthetic learning rate')
        parser.add_argument('--lr_net', type=float, default=0.03327, help='initialization for synthetic learning rate')

        parser.add_argument('--lr_init', type=float, default=0.01, help='how to init lr (alpha)')

        parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
        parser.add_argument('--batch_syn', type=int, default=None, help='should only use this if you run out of VRAM')
        parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
        parser.add_argument('--normalize_data', type=bool, default=False, help='batch size for training networks')

        parser.add_argument('--pix_init', type=str, default='real', choices=["noise", "real"],
                            help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')

        parser.add_argument('--dsa', type=str, default='True', choices=['True', 'False'],
                            help='whether to use differentiable Siamese augmentation.')

        parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate',
                            help='differentiable Siamese augmentation strategy')
        parser.add_argument('--optimizer', type=str, default='sgd',
                            help='differentiable Siamese augmentation strategy')

        parser.add_argument('--data_path', type=str, default='data', help='dataset path')
        parser.add_argument('--buffer_path', type=str, default='./buffers', help='buffer path')

        parser.add_argument('--expert_epochs', type=int, default=3, help='how many expert epochs the target params are')
        parser.add_argument('--syn_steps', type=int, default=20, help='how many steps to take on synthetic data')
        parser.add_argument('--max_start_epoch', type=int, default=25, help='max epoch we can start at')

        parser.add_argument('--zca', action='store_true', help="do ZCA whitening")

        parser.add_argument('--load_all', action='store_true', help="only use if you can fit all expert trajectories into RAM")

        parser.add_argument('--no_aug', type=bool, default=False, help='this turns off diff aug during distillation')

        parser.add_argument('--texture', action='store_true', help="will distill textures instead")
        parser.add_argument('--canvas_size', type=int, default=2, help='size of synthetic canvas')
        parser.add_argument('--canvas_samples', type=int, default=1, help='number of canvas samples per iteration')


        parser.add_argument('--max_files', type=int, default=None, help='number of expert files to read (leave as None unless doing ablations)')
        parser.add_argument('--max_experts', type=int, default=None, help='number of experts to read per file (leave as None unless doing ablations)')

        parser.add_argument('--force_save', action='store_true', help='this will save images for 50ipc')
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
            model = NetworkUtils.create_network(model_name)
            per_arch_accuracy[model_name] = EvaluatorUtils.evaluate_synset(0, model, self.input_images, self.input_labels, self.test_dataset, args)
        return per_arch_accuracy
        
# Evaluation for Trajectory Matching
if __name__ == '__main__':
    import sys
    sys.path.append('/home/justincui/dc_benchmark/dc_benchmark')
    from distilled_results.TrajectoryMatching.tm_data_loader import TMDataLoader

    args = CrossArchEvaluator.prepare_args()
    # train_image = TMDataLoader.load_images('/home/justincui/dc_benchmark/dc_benchmark/distilled_results/TrajectoryMatching/CIFAR10/IPC10/ZCA/images_5000.pt')
    # train_label = TMDataLoader.load_images('/home/justincui/dc_benchmark/dc_benchmark/distilled_results/TrajectoryMatching/CIFAR10/IPC10/ZCA/labels_5000.pt')
    # train_image = TMDataLoader.load_images('/home/justincui/dc_benchmark/dc_benchmark/distilled_results/TrajectoryMatching/CIFAR10/IPC10/NO_ZCA/images_5000.pt')
    # train_label = TMDataLoader.load_images('/home/justincui/dc_benchmark/dc_benchmark/distilled_results/TrajectoryMatching/CIFAR10/IPC10/NO_ZCA/labels_5000.pt')
    # train_image = TMDataLoader.load_images('/home/justincui/dc_benchmark/dc_benchmark/distilled_results/TrajectoryMatching/CIFAR10/IPC50/images_3000.pt')
    # train_label = TMDataLoader.load_images('/home/justincui/dc_benchmark/dc_benchmark/distilled_results/TrajectoryMatching/CIFAR10/IPC50/labels_3000.pt')
    train_image = TMDataLoader.load_images('/home/justincui/dc_benchmark/dc_benchmark/distilled_results/TrajectoryMatching/CIFAR10/IPC1/ZCA/images_zca_5000.pt')
    train_label = TMDataLoader.load_images('/home/justincui/dc_benchmark/dc_benchmark/distilled_results/TrajectoryMatching/CIFAR10/IPC1/ZCA/labels_5000.pt')
    print(train_image.shape)
    print(train_label.shape)
    args.zca = True
    args.dsa = True
    args.normalize_data = False
    # args.optimizer = 'adam'
    dst_test = EvaluatorUtils.get_cifar10_testset(args)
    testloader = torch.utils.data.DataLoader(dst_test, batch_size=256, shuffle=False, num_workers=0)
    evaluator = CrossArchEvaluator(train_image, train_label, testloader, {'models':['convnet']})
    evaluator.evaluate(args)

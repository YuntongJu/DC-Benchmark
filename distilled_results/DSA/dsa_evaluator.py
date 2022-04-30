import sys
sys.path.append('/home/justincui/dc_benchmark/dc_benchmark')

import torch
from evaluator.evaluator import Evaluator
from evaluator.evaluator_utils import EvaluatorUtils
from networks.network_utils import NetworkUtils
import argparse
import tqdm
import kornia as K
from torch.utils.data import Dataset


class CrossArchEvaluator(Evaluator):

    def __init__(self, input_images, input_labels, test_dataset, config):
        super().__init__(input_images, input_labels, test_dataset)
        self.config = config

    @staticmethod
    def prepare_args():
        parser = argparse.ArgumentParser(description='Parameter Processing')
        parser.add_argument('--method', type=str, default='DC', help='DC/DSA')
        parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
        parser.add_argument('--model', type=str, default='convnet', help='model')
        parser.add_argument('--ipc', type=int, default=10, help='image(s) per class')
        parser.add_argument('--dsa', action="store_true", help='dsa')
        parser.add_argument('--aug', type=str, default='', help='augmentation method')
        parser.add_argument('--eval_mode', type=str, default='S', help='eval_mode') # S: the same to training model, M: multi architectures,  W: net width, D: net depth, A: activation function, P: pooling layer, N: normalization layer,
        parser.add_argument('--num_exp', type=int, default=5, help='the number of experiments')
        parser.add_argument('--num_eval', type=int, default=5, help='the number of evaluating randomly initialized models')
        parser.add_argument('--optimizer', type=str, default='sgd', help='the number of evaluating randomly initialized models')
        parser.add_argument('--normalize_data', type=bool, default=True, help='the number of evaluating randomly initialized models')
        parser.add_argument('--epoch_eval_train', type=int, default=300, help='epochs to train a model with synthetic data')
        parser.add_argument('--Iteration', type=int, default=1000, help='training iterations')
        parser.add_argument('--lr_img', type=float, default=0.1, help='learning rate for updating synthetic images')
        parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
        parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
        parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
        parser.add_argument('--init', type=str, default='noise', help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')
        parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate', help='differentiable Siamese augmentation strategy')
        parser.add_argument('--data_path', type=str, default='data', help='dataset path')
        parser.add_argument('--save_path', type=str, default='result', help='path to save results')
        parser.add_argument('--dis_metric', type=str, default='ours', help='distance metric')
        args = parser.parse_args()
        args.dc_aug_param = EvaluatorUtils.get_daparam(args.dataset, args.model, '', args.ipc) # This augmentation parameter set is only for DC method. It will be muted when args.dsa is True.
        args.device = 'cuda'
        return args

    
    def evaluate(self, args):
        if args.dsa:
            args.dsa_param = EvaluatorUtils.ParamDiffAug()
            args.epoch_eval_train = 1000
            args.dc_aug_param = None
        if args.aug != '':
            args.epoch_eval_train = 1000
            args.dc_aug_param = None
        if args.dc_aug_param != None and args.dc_aug_param['strategy'] != 'none':
            args.epoch_eval_train = 1000

        per_arch_accuracy = {}
        for model_name in self.config['models']:
            model = NetworkUtils.create_network(model_name)
            _, _, test_acc = EvaluatorUtils.evaluate_synset(0, model, self.input_images, self.input_labels, self.test_dataset, args)
            per_arch_accuracy[model_name]  = test_acc
        return per_arch_accuracy

# Evaluation for DSA
if __name__ == '__main__':
    import sys
    sys.path.append('/home/justincui/dc_benchmark/dc_benchmark')
    from distilled_results.DC.dc_data_loader import DCDataLoader

    args = CrossArchEvaluator.prepare_args()
    data_path = ''
    if args.dataset == 'CIFAR10':
        data_path = '/home/justincui/dc_benchmark/dc_benchmark/distilled_results/DSA/CIFAR10/IPC' + str(args.ipc) + '/res_DSA_CIFAR10_ConvNet_' + str(args.ipc) + 'ipc.pt'
    elif args.dataset == 'CIFAR100':
        data_path = '/home/justincui/dc_benchmark/dc_benchmark/distilled_results/DSA/CIFAR100/IPC' + str(args.ipc) + '/res_DSA_CIFAR100_ConvNet_' + str(args.ipc) + 'ipc.pt'

    train_image, train_label = DCDataLoader.load_data(data_path)

    # args.optimizer = 'adam'
    dst_test = EvaluatorUtils.get_cifar10_testset(args)
    testloader = torch.utils.data.DataLoader(dst_test, batch_size=256, shuffle=False, num_workers=0)
    evaluator = CrossArchEvaluator(train_image, train_label, testloader, {'models':[args.model]})
    avg_acc = 0.0
    for i in range(args.num_eval):
        print("current iteration: ", i)
        per_arch_accuracy = evaluator.evaluate(args)
        avg_acc += per_arch_accuracy[args.model]
    print("final average result is: ", avg_acc / args.num_eval, " for ", args.model, " and IPC ", args.ipc, " DSA:", args.dsa, " num eval:", args.num_eval, ' ', args.aug)

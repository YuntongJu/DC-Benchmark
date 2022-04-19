import sys
sys.path.append('/home/justincui/dc_benchmark/dc_benchmark')

import torch
from evaluator.evaluator import Evaluator
from evaluator.evaluator_utils import EvaluatorUtils
from networks.network_utils import NetworkUtils
import argparse


class CrossArchEvaluator(Evaluator):

    def __init__(self, input_images, input_labels, test_dataset, config):
        super().__init__(input_images, input_labels, test_dataset)
        self.config = config

    @staticmethod
    def prepare_args():
        parser = argparse.ArgumentParser(description='Parameter Processing')
        parser.add_argument('--method', type=str, default='DM', help='DC/DSA/DM')
        parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
        parser.add_argument('--model', type=str, default='ConvNet', help='model')
        parser.add_argument('--ipc', type=int, default=1, help='image(s) per class')
        parser.add_argument('--eval_mode', type=str, default='S', help='eval_mode') # S: the same to training model, M: multi architectures,  W: net width, D: net depth, A: activation function, P: pooling layer, N: normalization layer,
        parser.add_argument('--num_exp', type=int, default=5, help='the number of experiments')
        parser.add_argument('--num_eval', type=int, default=20, help='the number of evaluating randomly initialized models')
        parser.add_argument('--normalize_data', type=bool, default=True, help='the number of evaluating randomly initialized models')
        parser.add_argument('--optimizer', type=str, default='sgd', help='the number of evaluating randomly initialized models')
        parser.add_argument('--epoch_eval_train', type=int, default=300, help='epochs to train a model with synthetic data')
        parser.add_argument('--Iteration', type=int, default=20000, help='training iterations')
        parser.add_argument('--lr_img', type=float, default=1.0, help='learning rate for updating synthetic images')
        parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
        parser.add_argument('--zca', type=bool, default=False, help='learning rate for updating network parameters')
        parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
        parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
        parser.add_argument('--init', type=str, default='real', help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')
        parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_scale_rotate', help='differentiable Siamese augmentation strategy')
        parser.add_argument('--data_path', type=str, default='data', help='dataset path')
        parser.add_argument('--save_path', type=str, default='result', help='path to save results')
        parser.add_argument('--dis_metric', type=str, default='ours', help='distance metric')
        args = parser.parse_args()
        args.dc_aug_param = EvaluatorUtils.get_daparam(args.dataset, args.model, '', args.ipc) # This augmentation parameter set is only for DC method. It will be muted when args.dsa is True.
        args.dsa = False
        args.device = 'cuda'
        return args

    
    def evaluate(self, args):
        if args.dsa:
            print("------------using DSA augmentation-----------")
            args.dsa_param = EvaluatorUtils.ParamDiffAug()
            args.epoch_eval_train = 1000
            args.dc_aug_param = None
        per_arch_accuracy = {}
        for model_name in self.config['models']:
            model = NetworkUtils.create_network(model_name)
            net, acc_train, acc_test= EvaluatorUtils.evaluate_synset(0, model, self.input_images, self.input_labels, self.test_dataset, args)
            per_arch_accuracy[model_name] = acc_test
        return per_arch_accuracy
        

# Evaluation for DC
if __name__ == '__main__':
    import sys
    sys.path.append('/home/justincui/dc_benchmark/dc_benchmark')
    from distilled_results.DM.dm_data_loader import DMDataLoader

    # train_image, train_label = DMDataLoader.load_data('/home/justincui/dc_benchmark/dc_benchmark/distilled_results/DM/CIFAR10/IPC10/res_DM_CIFAR10_ConvNet_10ipc.pt')
    train_image, train_label = DMDataLoader.load_data('/home/justincui/dc_benchmark/dc_benchmark/distilled_results/DM/CIFAR10/IPC50/res_DM_CIFAR10_ConvNet_50ipc.pt')
    print(train_image.shape)
    print(train_label.shape)
    args = CrossArchEvaluator.prepare_args()
    args.dsa = True
    args.zca = False
    # args.optimizer = 'adam'
    dst_test = EvaluatorUtils.get_cifar10_testset(args)
    testloader = torch.utils.data.DataLoader(dst_test, batch_size=256, shuffle=False, num_workers=0)
    avg_result = 0.0
    num_eval = 1
    model_name = 'alexnet'
    for i in range(num_eval):
        evaluator = CrossArchEvaluator(train_image, train_label, testloader, {'models':[model_name]})
        result = evaluator.evaluate(args)
        avg_result += result[model_name]
    # for i in range(num_eval):
    #     evaluator = CrossArchEvaluator(train_image, train_label, testloader, {'models':['convnet']})
    #     result = evaluator.evaluate(args)
    #     avg_result += result['convnet']
    print("average result for ", num_eval, avg_result / num_eval)

    

import sys
sys.path.append('/home/justincui/dc_benchmark/dc_benchmark')

import torch
from evaluator import Evaluator
from evaluator_utils import EvaluatorUtils
from networks.network_utils import NetworkUtils
import argparse


class CrossArchEvaluator(Evaluator):

    def __init__(self, input_images, input_labels, test_dataset, config):
        super().__init__(input_images, input_labels, test_dataset)
        self.config = config

    def prepare_args(self):
        parser = argparse.ArgumentParser(description='Parameter Processing')
        parser.add_argument('--method', type=str, default='DC', help='DC/DSA')
        parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
        parser.add_argument('--model', type=str, default='ConvNet', help='model')
        parser.add_argument('--ipc', type=int, default=1, help='image(s) per class')
        parser.add_argument('--eval_mode', type=str, default='S', help='eval_mode') # S: the same to training model, M: multi architectures,  W: net width, D: net depth, A: activation function, P: pooling layer, N: normalization layer,
        parser.add_argument('--num_exp', type=int, default=5, help='the number of experiments')
        parser.add_argument('--num_eval', type=int, default=20, help='the number of evaluating randomly initialized models')
        parser.add_argument('--epoch_eval_train', type=int, default=300, help='epochs to train a model with synthetic data')
        parser.add_argument('--Iteration', type=int, default=1000, help='training iterations')
        parser.add_argument('--lr_img', type=float, default=0.1, help='learning rate for updating synthetic images')
        parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
        parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
        parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
        parser.add_argument('--init', type=str, default='noise', help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')
        parser.add_argument('--dsa_strategy', type=str, default='None', help='differentiable Siamese augmentation strategy')
        parser.add_argument('--data_path', type=str, default='data', help='dataset path')
        parser.add_argument('--save_path', type=str, default='result', help='path to save results')
        parser.add_argument('--dis_metric', type=str, default='ours', help='distance metric')
        args = parser.parse_args()
        args.dsa = False
        args.device = 'cuda'
        return args

    
    def evaluate(self):
        args = self.prepare_args()
        per_arch_accuracy = {}
        for model_name in self.config['models']:
            model = NetworkUtils.create_network(model_name)
            per_arch_accuracy[model_name] = EvaluatorUtils.evaluate_synset(0, model, self.input_images, self.input_labels, self.test_dataset, args)
        return per_arch_accuracy
        
# evaluation for DM
# if __name__ == '__main__':
#     import sys
#     sys.path.append('/home/justincui/dc_benchmark/dc_benchmark')
#     from distilled_results.DM.dm_data_loader import DMDataLoader
#     from torchvision import datasets, transforms

#     train_image, train_label = DMDataLoader.load_data('/home/justincui/dc_benchmark/dc_benchmark/distilled_results/DM/CIFAR10/IPC10/res_DM_CIFAR10_ConvNet_10ipc.pt')
#     print(train_image.shape)
#     print(train_label.shape)
#     dst_test = datasets.CIFAR10('data', train=False, download=True, transform=transforms.ToTensor())
#     testloader = torch.utils.data.DataLoader(dst_test, batch_size=256, shuffle=False, num_workers=0)
#     evaluator = CrossArchEvaluator(train_image, train_label, testloader, {'models':['convnet']})
#     evaluator.evaluate()

def get_cifar10_testset():
    channel = 3
    im_size = (32, 32)
    num_classes = 10
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    dst_test = datasets.CIFAR10('data', train=False, download=True, transform=transform)
    return dst_test

# Evaluation for DC
if __name__ == '__main__':
    import sys
    sys.path.append('/home/justincui/dc_benchmark/dc_benchmark')
    from distilled_results.DC.dc_data_loader import DCDataLoader
    from torchvision import datasets, transforms

    train_image, train_label = DCDataLoader.load_data('/home/justincui/dc_benchmark/dc_benchmark/distilled_results/DC/CIFAR10/IPC10/res_DC_CIFAR10_ConvNet_10ipc.pt')
    print(train_image.shape)
    print(train_label.shape)
    dst_test = get_cifar10_testset()
    testloader = torch.utils.data.DataLoader(dst_test, batch_size=256, shuffle=False, num_workers=0)
    evaluator = CrossArchEvaluator(train_image, train_label, testloader, {'models':['convnet']})
    evaluator.evaluate()
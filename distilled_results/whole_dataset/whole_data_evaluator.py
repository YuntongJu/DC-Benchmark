import sys
sys.path.append('/nfs/data/justincui/dc_benchmark')

import torch
from evaluator.evaluator import Evaluator
from evaluator.evaluator_utils import EvaluatorUtils
from networks.network_utils import NetworkUtils
import argparse
import logging


class CrossArchEvaluator(Evaluator):

    def __init__(self, input_images, input_labels, test_dataset, config):
        super().__init__(input_images, input_labels, test_dataset)
        self.config = config

    @staticmethod
    def prepare_args():
        parser = argparse.ArgumentParser(description='Parameter Processing')
        parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
        parser.add_argument('--model', type=str, default='convnet', help='model')
        parser.add_argument('--ipc', type=int, default=50, help='image(s) per class')
        parser.add_argument('--eval_mode', type=str, default='S', help='eval_mode') # S: the same to training model, M: multi architectures,  W: net width, D: net depth, A: activation function, P: pooling layer, N: normalization layer,
        parser.add_argument('--normalize_data', type=bool, default=False, help='whether to normalize the data') # S: the same to training model, M: multi architectures,  W: net width, D: net depth, A: activation function, P: pooling layer, N: normalization layer,
        parser.add_argument('--num_exp', type=int, default=5, help='the number of experiments')
        parser.add_argument('--num_eval', type=int, default=20, help='the number of evaluating randomly initialized models')
        parser.add_argument('--print', type=bool, default=True, help='the number of evaluating randomly initialized models')
        parser.add_argument('--epoch_eval_train', type=int, default=50, help='epochs to train a model with synthetic data')
        parser.add_argument('--Iteration', type=int, default=1000, help='training iterations')
        parser.add_argument('--lr_img', type=float, default=0.1, help='learning rate for updating synthetic images')
        parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
        parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
        parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
        parser.add_argument('--init', type=str, default='noise', help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')
        parser.add_argument('--optimizer', type=str, default='sgd', help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')
        parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate', help='differentiable Siamese augmentation strategy')
        parser.add_argument('--data_path', type=str, default='data', help='dataset path')
        parser.add_argument('--save_path', type=str, default='result', help='path to save results')
        parser.add_argument('--dis_metric', type=str, default='ours', help='distance metric')
        args = parser.parse_args()
        args.dsa = False
        args.dc_aug_param = EvaluatorUtils.get_daparam(args.dataset, args.model, '', args.ipc) # This augmentation parameter set is only for DC method. It will be muted when args.dsa is True.
        args.device = 'cuda'
        return args

    @staticmethod
    def evaluate(args, dst_train, dst_test, logging):
        testloader = torch.utils.data.DataLoader(dst_test, batch_size=256, shuffle=True, num_workers=0)

        if args.dsa:
            args.dsa_param = EvaluatorUtils.ParamDiffAug()
            args.epoch_eval_train = 150
            args.dc_aug_param = None
        per_arch_accuracy = {}
        for model_name in [args.model]:
            model = NetworkUtils.create_network(args)
            start_params = [p.detach().cpu() for p in model.parameters()]
            net, acc_train, acc_test = EvaluatorUtils.evaluate_synset_dataset(0, model, dst_train, testloader, args, logging)
            end_params = [p.detach().cpu() for p in net.parameters()]
            per_arch_accuracy[model_name] = acc_test
        return per_arch_accuracy, (start_params, end_params)
    

# Evaluation for DSA
if __name__ == '__main__':
    import sys
    sys.path.append('/home/justincui/dc_benchmark/dc_benchmark')

    args = CrossArchEvaluator.prepare_args()
    args.dsa = False
    dst_train, dst_test = EvaluatorUtils.get_dataset(args)
    print("train set length:", len(dst_train))
    print("test set length:", len(dst_test))
    testloader = torch.utils.data.DataLoader(dst_test, batch_size=256, shuffle=False, num_workers=0)
    avg_acc = []
    for i in range(args.num_eval):
        print("current iteration: ", i)
        result, (start_params, end_params) = CrossArchEvaluator.evaluate(args, dst_train, dst_test, logging)
        torch.save((start_params, end_params), '/nfs/data/justincui/model_matching/traces/' + str(i) + '_traces.pt')
        avg_acc.append(result[args.model])
    mean, std = EvaluatorUtils.compute_std_mean(avg_acc)
    logging.warning("Whole: final acc is: %.2f +- %.2f, dataset: %s, IPC: %d, DSA:%r, num_eval: %d, model: %s, optimizer: %s", 
        mean * 100, std * 100, 
        args.dataset, 
        args.ipc,
        args.dsa,
        args.num_eval,
        args.model,
        args.optimizer
    )

    print("Whole: final acc is: %.2f +- %.2f, dataset: %s, IPC: %d, DSA:%r, num_eval: %d, model: %s, optimizer: %s" % 
        (mean * 100, 
        std * 100, 
        args.dataset, 
        args.ipc,
        args.dsa,
        args.num_eval,
        args.model,
        args.optimizer)
    )

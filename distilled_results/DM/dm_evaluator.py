import sys
sys.path.append('../../../dc_benchmark')

import torch
from evaluator.evaluator import Evaluator
from evaluator.evaluator_utils import EvaluatorUtils
from networks.network_utils import NetworkUtils
import argparse
import os
import logging


class CrossArchEvaluator(Evaluator):

    def __init__(self, input_images, input_labels, test_dataset, config):
        super().__init__(input_images, input_labels, test_dataset)
        self.config = config

    @staticmethod
    def prepare_args():
        parser = argparse.ArgumentParser(description='Parameter Processing')
        parser.add_argument('--verbose', action="store_true",  help='whether to output extra logging')
        parser.add_argument('--gpu', type=str, default='auto', help='gpu ID(s)')
        parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
        parser.add_argument('--model', type=str, default='convnet', help='model')
        parser.add_argument('--ipc', type=int, default=1, help='image(s) per class')
        parser.add_argument('--dsa', action="store_true", help='dsa')
        parser.add_argument('--aug', type=str, default='', help='augmentation method')
        parser.add_argument('--eval_mode', type=str, default='S', help='eval_mode') # S: the same to training model, M: multi architectures,  W: net width, D: net depth, A: activation function, P: pooling layer, N: normalization layer,
        parser.add_argument('--num_eval', type=int, default=10, help='the number of evaluating randomly initialized models')
        parser.add_argument('--normalize_data', action="store_true", help='the number of evaluating randomly initialized models')
        parser.add_argument('--optimizer', type=str, default='sgd', help='the number of evaluating randomly initialized models')
        parser.add_argument('--epoch_eval_train', type=int, default=300, help='epochs to train a model with synthetic data')
        parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
        parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
        parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_scale_rotate', help='differentiable Siamese augmentation strategy')
        args = parser.parse_args()
        args.dc_aug_param = EvaluatorUtils.get_daparam(args.dataset, args.model, '', args.ipc) # This augmentation parameter set is only for DC method. It will be muted when args.dsa is True.
        os.environ["CUDA_VISIBLE_DEVICES"] = str(EvaluatorUtils.pick_gpu_lowest_memory()) if args.gpu == 'auto' else args.gpu
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return args

    
    def evaluate(self, args, logging):
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
            model = NetworkUtils.create_network(args)
            _, _, acc_test= EvaluatorUtils.evaluate_synset(0, model, self.input_images, self.input_labels, self.test_dataset, args, logging)
            per_arch_accuracy[model_name] = acc_test
        return per_arch_accuracy
        

# Evaluation for DC
if __name__ == '__main__':
    import sys
    sys.path.append('/home/justincui/dc_benchmark/dc_benchmark')
    from distilled_results.DM.dm_data_loader import DMDataLoader
    args = CrossArchEvaluator.prepare_args()

    logging.basicConfig(
        filename = 'dm_' + args.model + '.log',
        filemode = 'a',
        format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', 
        level=logging.WARNING
    )

    if args.ipc <= 50:
        data_path = '/nfs/data/justincui/dc_benchmark/distilled_results/DM/'+ args.dataset + '/IPC' + str(args.ipc)
    else:
        data_path = '/home/justincui/dm'
    data_path = data_path  + '/res_DM_' + args.dataset
    if args.dataset == 'tinyimagenet':
        data_path += '_ConvNetD4_'
    else:
        data_path += '_ConvNet_'
    data_path += str(args.ipc) + 'ipc.pt'
    
    train_image, train_label = DMDataLoader.load_data(data_path)
    print(train_image.shape)
    print(train_label.shape)
    print(train_image.min())
    print(train_image.max())
    # args.optimizer = 'adam'
    dst_test = EvaluatorUtils.get_testset(args)
    testloader = torch.utils.data.DataLoader(dst_test, batch_size=256, shuffle=False, num_workers=0)
    avg_acc = []
    for i in range(args.num_eval):
        evaluator = CrossArchEvaluator(train_image, train_label, testloader, {'models':[args.model]})
        result = evaluator.evaluate(args, logging)
        avg_acc.append(result[args.model])
    mean, std = EvaluatorUtils.compute_std_mean(avg_acc)
    logging.warning("DM: final acc is: %.2f +- %.2f, dataset: %s, IPC: %d, DSA:%r, num_eval: %d, aug:%s , model: %s, optimizer: %s", 
        mean * 100, std * 100, 
        args.dataset, 
        args.ipc,
        args.dsa,
        args.num_eval,
        args.aug,
        args.model,
        args.optimizer
    )

    print("DM: final acc is: %.2f +- %.2f, dataset: %s, IPC: %d, DSA:%r, num_eval: %d, aug:%s , model: %s, optimize: %s" % 
        (mean * 100, std * 100, 
        args.dataset, 
        args.ipc,
        args.dsa,
        args.num_eval,
        args.aug,
        args.model,
        args.optimizer)
    )

    

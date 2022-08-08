import sys
sys.path.append('/home/justincui/dc_benchmark/dc_benchmark')

import torch
import argparse
import numpy as np


# Evaluation for DSA
if __name__ == '__main__':
    import sys
    sys.path.append('/home/justincui/dc_benchmark/dc_benchmark')

    data = np.load('./CIFAR10/IPC10/kip_cifar10_ConvNet_ssize100_zca_nol_noaug_ckpt1000.npz')
    print(list(data.keys()))
    # print(train_label)
    # print(train_image.shape, train_image.dtype)
    # print(train_label.shape, train_label.dtype)
    # dst_test = get_cifar10_testset()
    # testloader = torch.utils.data.DataLoader(dst_test, batch_size=256, shuffle=False, num_workers=0)
    # evaluator = CrossArchEvaluator(train_image, train_label, testloader, {'models':['convnet']})
    # args.soft_label = True
    # evaluator.evaluate(args)

import sys
sys.path.append('/home/justincui/dc_benchmark/dc_benchmark')

from networks.convnet import ConvNet
from networks.alexnet import AlexNet
from networks.resnet import ResNet, ResNet18
class NetworkUtils:

    @staticmethod
    def create_network(model_name, dataset):
        if dataset == 'CIFAR10':
            channel = 3
            num_classes = 10
        elif dataset == 'CIFAR100':
            channel = 3
            num_classes = 100

        if model_name == 'convnet':
            net_width, net_depth, net_act, net_norm, net_pooling = 128, 3, 'relu', 'instancenorm', 'avgpooling'
            return ConvNet(channel, num_classes, net_width, net_depth, net_act, net_norm, net_pooling, im_size = (32,32))
        if model_name == 'convnet4':
            net_width, net_depth, net_act, net_norm, net_pooling = 128, 4, 'relu', 'instancenorm', 'avgpooling'
            return ConvNet(channel, num_classes, net_width, net_depth, net_act, net_norm, net_pooling, im_size = (32,32))
        elif model_name == 'alexnet':
            return AlexNet(channel, num_classes)
        elif model_name == 'resnet18':
            return ResNet18(channel=channel, num_classes=num_classes)
        return None
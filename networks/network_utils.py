import sys
sys.path.append('/home/justincui/dc_benchmark/dc_benchmark')

from networks.convnet import ConvNet
from networks.alexnet import AlexNet
from networks.mlp import MLP
from networks.resnet import ResNet152, ResNet18, ResNet18ImageNet, ResNet152Imagenet
class NetworkUtils:

    @staticmethod
    def create_network(args):
        channel = 3
        model_name = args.model
        if args.dataset == 'CIFAR10':
            im_size = (32, 32)
            num_classes = 10
        elif args.dataset == 'CIFAR100':
            im_size = (32, 32)
            num_classes = 100
        elif args.dataset == 'tinyimagenet':
            im_size = (64, 64)
            num_classes = 200
        if model_name == 'mlp':
            return MLP(channel, num_classes, im_size)
        if model_name == 'convnet':
            net_width, net_depth, net_act, net_norm, net_pooling = 128, 3, 'relu', 'instancenorm', 'avgpooling'
            return ConvNet(channel, num_classes, net_width, net_depth, net_act, net_norm, net_pooling, im_size = im_size)
        if model_name == 'convnet4':
            net_width, net_depth, net_act, net_norm, net_pooling = 128, 4, 'relu', 'instancenorm', 'avgpooling'
            return ConvNet(channel, num_classes, net_width, net_depth, net_act, net_norm, net_pooling, im_size = im_size)
        if model_name == 'convnet2':
            net_width, net_depth, net_act, net_norm, net_pooling = 128, 2, 'relu', 'instancenorm', 'avgpooling'
            return ConvNet(channel, num_classes, net_width, net_depth, net_act, net_norm, net_pooling, im_size = im_size)
        if model_name == 'convnet1':
            net_width, net_depth, net_act, net_norm, net_pooling = 128, 1, 'relu', 'instancenorm', 'avgpooling'
            return ConvNet(channel, num_classes, net_width, net_depth, net_act, net_norm, net_pooling, im_size = im_size)
        elif model_name == 'alexnet':
            return AlexNet(channel, num_classes)
        elif model_name == 'resnet18':
            return ResNet18(channel=channel, num_classes=num_classes)
        elif model_name == 'resnet18imagenet':
            return ResNet18ImageNet(channel=channel, num_classes=num_classes)
        elif model_name == 'resnet152':
            return ResNet152(channel=channel, num_classes=num_classes)
        elif model_name == 'resnet152imagenet':
            return ResNet152Imagenet(channel=channel, num_classes=num_classes)
        return None
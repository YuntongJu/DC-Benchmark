import sys
sys.path.append('/home/justincui/dc_benchmark/dc_benchmark')

from networks.convnet import ConvNet
class NetworkUtils:

    @staticmethod
    def create_network(model_name):
        if model_name == 'convnet':
            channel = 3
            num_classes = 10
            net_width, net_depth, net_act, net_norm, net_pooling = 128, 3, 'relu', 'instancenorm', 'avgpooling'
            return ConvNet(channel, num_classes, net_width, net_depth, net_act, net_norm, net_pooling, im_size = (32,32))
        return None
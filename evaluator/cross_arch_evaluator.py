from evaluator import Evaluator
from evaluator_utils import EvaluatorUtils


class CrossArchEvaluator(Evaluator):

    def __init__(self, input_images, input_labels, test_dataset, config):
        super().__init__(input_images, input_labels, test_dataset)
        self.config = config
    
    def evaluate(self):
        per_arch_accuracy = {}
        for model_name in self.config['models']:
            model = EvaluatorUtils.create_network(model_name)
            per_arch_accuracy[model_name] = EvaluatorUtils.evaluate_synset(model, input_images, input_labels, test_dataset)
        return per_arch_accuracy
        
if __name__ == '__main__':
    import sys
    sys.path.append('/home/justincui/dc_benchmark/dc_benchmark')
    from distilled_results.distribution_matching.dm_data_loader import DMDataLoader
    from torchvision import datasets, transforms

    train_image, train_label = DMDataLoader.load_data('/home/justincui/dc_benchmark/dc_benchmark/distilled_results/distribution_matching/res_DM_CIFAR10_ConvNet_50ipc.pt')
    print(train_image.shape)
    print(train_label.shape)
    dst_test = datasets.CIFAR10('data', train=False, download=True, transform=transforms.ToTensor())
    evaluator = CrossArchEvaluator(train_image, train_label, dst_test, {'models':['convnet']})
    evaluator.evaluate()
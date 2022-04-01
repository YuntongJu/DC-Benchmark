from evaluator import Evaluator


class CrossArchEvaluator(Evaluator):

    def __init__(input_images, input_labels, test_dataset, config):
        super().__init__(input_images, input_labels, test_dataset)
        this.config = config
    
    def evaluate():
        per_arch_accuracy = {}
        for model_name in config.models:
            model = EvaluatorUtils.create_network(model_name)
            per_arch_accuracy[model_name] = EvaluatorUtils.evaluate_synset(model, input_images, input_labels, test_dataset)
        return per_arch_accuracy
        
if __name__ == '__main__':
    import sys
    sys.path.append('/home/justincui/dc_benchmark/dc_benchmark')
    from distilled_results.distribution_matching.dm_data_loader import DMDataLoader
    train_image, train_label = DMDataLoader.load_data('/home/justincui/dc_benchmark/dc_benchmark/distilled_results/distribution_matching/res_DM_CIFAR10_ConvNet_50ipc.pt')
    print(train_image.shape)
    print(train_label.shape)
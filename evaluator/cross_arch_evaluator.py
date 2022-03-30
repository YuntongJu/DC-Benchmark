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
        

from evaluator import Evaluator


class CrossArchEvaluator(Evaluator):

    def __init__(input_images, input_labels, test_dataset, config):
        super().__init__(input_images, input_labels, test_dataset)
        this.config = config
    
    def evaluate():
        

from evaluator import Evaluator


class DirectEvaluator(Evaluator):

    def __init__(input_images, input_labels, test_dataset):
        super().__init__(input_images, input_labels, test_dataset)
    
    def evaluate():
        '''
        Use the same model architecture as training.
        '''
        pass
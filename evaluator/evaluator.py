class Evaluator:

    def __init__(self, input_images, input_labels, test_dataset):
       self.input_image = input_images
       self.input_labels = input_labels
       self.test_dataset = test_dataset

    def evaluate(self):
        '''
        return a map containing the evaluated result.
        '''
        return {'condensation_method' : 1.0}
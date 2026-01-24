import numpy as np
from Layers.Base import BaseLayer

class Dropout(BaseLayer):
    def __init__(self,probability):
        super().__init__()
        self.probability = probability # probability to keep an activation
        self.drop_mask = None

    def forward(self, input_tensor):
        # During training set activations to zero with probability 1-p
        if self.testing_phase:
            # while testing just return input tensor without modifying it
            return input_tensor
        else:
            # create binary drop mask, that is multiplied with the input tensor
            # generate drop_mask by generating random values between 0 and 1
            # if below p keep them, else drop them (1-p)
            self.drop_mask = np.where(np.random.rand(*input_tensor.shape) < self.probability, 1/self.probability, 0)
            return input_tensor * self.drop_mask
        
    def backward(self, error_tensor):
        error_prev = error_tensor * self.drop_mask
        return error_prev
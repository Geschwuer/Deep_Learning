from Layers.Base import BaseLayer
import numpy as np

class SoftMax(BaseLayer):
    def __init__(self):
        super().__init__()
        self.probabilities = None

    def forward(self, input_tensor):
        # center the input tensor to increase numeric stabilty
        input_tensor_centered = input_tensor - np.max(input_tensor, axis=1, keepdims=True)
        # calculate output probabilities
        probabilities = np.exp(input_tensor_centered)/np.sum(np.exp(input_tensor_centered), axis=1, keepdims=True)
        # save probs for backprop
        self.probabilities = probabilities 
        return probabilities
    
    def backward(self, error_tensor):
        # calculate sclar product for every batch
        s = np.sum(error_tensor * self.probabilities, axis=1, keepdims=True)
        # calculate error for backprop
        error_prev = self.probabilities * (error_tensor - s)
        return error_prev
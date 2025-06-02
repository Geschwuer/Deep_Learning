from Layers.Base import BaseLayer

import numpy as np

class SoftMax(BaseLayer):
    def __init__(self):
        super().__init__()
        self.output_tensor = None

    def forward(self, input_tensor):
        # row wise computation of softmax
        

        """        
        output_tensor = np.empty((input_tensor.shape[0], input_tensor.shape[1]))
        for i in range(input_tensor.shape[0]):
            summe = 0
            for k in range(input_tensor.shape[1]):
                summe += math.exp(input_tensor[i, k])
            
            for j in range(input_tensor.shape[1]):
                output_tensor[i][j] = math.exp(input_tensor[i][j]) / summe

        return output_tensor
        """

        # Numerical stability: subtract max from each row
        shifted_tensor = input_tensor - np.max(input_tensor, axis=1, keepdims=True)
        exp_tensor = np.exp(shifted_tensor)
        self.output_tensor = exp_tensor / np.sum(exp_tensor, axis=1, keepdims=True)
        return self.output_tensor



    def backward(self, error_tensor):
        # self.output_tensor: (batch_size, num_classes)
        dot = np.sum(error_tensor * self.output_tensor, axis=1, keepdims=True)
        # dot: shape (batch_size, 1)

        return self.output_tensor * (error_tensor - dot)
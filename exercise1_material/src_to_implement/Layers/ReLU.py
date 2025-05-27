from Layers.Base import BaseLayer

import numpy as np

class ReLU(BaseLayer):
    def __init__(self):
        super().__init__()
    
    def forward(self, input_tensor : np.ndarray) -> np.ndarray:
        self.input_tensor = input_tensor # store input for backward pass

        output_tensor = input_tensor[input_tensor < 0]  = 0
        return output_tensor
    

    def backward(self, error_tensor): 
        # in BackPropagation compute: dL/dx = dL/dy * dy/dx
        # method input: error_tensor = dL/dy
        # so we have to compute: dy/dx
        
        # Derivative (dy/dx) of ReLU: 1 for input > 0, else 0
        relu_derivative = self.input_tensor > 0

        return error_tensor * relu_derivative
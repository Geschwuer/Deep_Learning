from Layers.Base import BaseLayer
import numpy as np

class ReLU(BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        return np.maximum(0, input_tensor)
    
    def backward(self, error_tensor):
        # compute ReLU derivative
        # 0 if x <= 0
        # e else
        relu_deriv = self.input_tensor > 0 # binary mask
        return error_tensor * relu_deriv # apply mask
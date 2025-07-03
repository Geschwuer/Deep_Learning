from Layers.Base import BaseLayer
import numpy as np

class Sigmoid(BaseLayer):
    def __init__(self):
        super().__init__()
        self.trainable = False
        self.activation = None

    def forward(self, input_tensor):
        self.activation = 1 / (1 + np.exp(- input_tensor))
        return self.activation

    def backward(self, error_tensor):
        return self.activation * (1 - self.activation) * error_tensor
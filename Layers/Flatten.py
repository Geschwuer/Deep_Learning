from Layers.Base import BaseLayer
import numpy as np

class Flatten(BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        self.shape = input_tensor.shape # save original shape for backprop
        num_batches = input_tensor.shape[0]

        return input_tensor.reshape((num_batches, -1))
    
    def backward(self, error_tensor):
        return error_tensor.reshape(self.shape)

# class Flatten(BaseLayer):
#     def _init_(self):
#         super().__init__()
#         self.input_shape = None     # use in backward path

#     def forward(self, input_tensor):
#         self.input_shape = input_tensor.shape       # needed for backpropagation
#         batch_size = input_tensor.shape[0]     # keeps the first dimension (batch) intact and flattens everything else

#         return np.reshape(input_tensor, newshape=(batch_size, -1))     # argument -1 = one shape dimension: input tensor --> 2d array containing batches

#     def backward(self, error_tensor):
#         return np.reshape(error_tensor, newshape=self.input_shape)     # reshapes error_tensor to match the original input shape
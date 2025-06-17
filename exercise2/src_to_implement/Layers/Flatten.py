import numpy as np

class Flatten:
    def __init__(self):
        self.input_shape = None     # use in backward path

    def forward(self, input_tensor):
        self.input_shape = input_tensor.shape       # needed for backpropagation
        batch_size = input_tensor.shape[0]     # keeps the first dimension (batch) intact and flattens everything else

        return np.reshape(input_tensor, shape=(batch_size, -1))     # argument -1 = one shape dimension: input tensor --> 2d array containing batches

    def backward(self, error_tensor):
        return np.reshape(error_tensor, shape=self.input_shape)     # reshapes error_tensor to match the original input shape
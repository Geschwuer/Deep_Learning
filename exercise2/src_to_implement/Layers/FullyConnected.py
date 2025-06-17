from Layers.Base import BaseLayer
import numpy as np

class FullyConnected(BaseLayer):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.trainable = True
        # initialize weight matrix with uniformly distributed weights [0, 1]
        self.weights = np.random.rand(input_size + 1, output_size) # +1 for bias
        self._gradient_weights = None
        self._optimizer = None
        self.input_tensor = None


    def initialize(self, weights_initializer, bias_initializer):
        # fan_in is input size, fan_out is output size
        fan_in = self.weights.shape[0] - 1  # subtract 1 for bias row
        fan_out = self.weights.shape[1]

        weights = weights_initializer.initialize((fan_in, fan_out), fan_in, fan_out)
        bias = bias_initializer.initialize((1, fan_out), fan_in, fan_out)

        self.weights = np.vstack(weights, bias)


    def forward(self, input_tensor: np.ndarray) -> np.ndarray:
        # bias is included in the weights
        # --> augment input_tensor with a bias term
        ones = np.ones((input_tensor.shape[0], 1)) # column of ones
        input_tensor_bias = np.concatenate((input_tensor, ones), axis=1) # axis = 1 joins them side by side
        self.input_tensor = input_tensor_bias # store for backward pass
        output_tensor = input_tensor_bias @ self.weights # matrix multiplication
        return output_tensor
    

    def backward(self, error_tensor: np.ndarray) -> np.ndarray:
        # update weights
        self._gradient_weights = self.input_tensor.T @ error_tensor # (input+1 x output)

        # update weights
        if self._optimizer is not None:
            self.weights = self._optimizer.calculate_update(self.weights, self._gradient_weights)

        # calculate error_tensor for previous layer
        error_prev_bias = error_tensor @ self.weights.T # here the bias is included !
        error_prev = error_prev_bias[:,:-1] # remove bias term (last column)
        return error_prev
        
    
    @property # acess optimizer like class attribute, e.g. optimizer = fc_layer.optimizer
    def optimizer(self):
        return self._optimizer
    

    @optimizer.setter # set optimizer like class attribute, e.g. fc_layer.optimizer = optimizer
    def optimizer(self, optimizer):
        self._optimizer = optimizer


    @property
    def gradient_weights(self):
        return self._gradient_weights
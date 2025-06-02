from Layers.Base import BaseLayer

import numpy as np

class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.trainable = True

        # input_size is the number of input features or neurons feeding into this layer.
        # output_size is the number of output neurons in this layer.
        # Each output neuron has input_size weights (one for each input neuron).
        # The full weight matrix has a shape of (input_size, output_size).
        # Each column in the weight matrix corresponds to a single output neuron.
        # Each row corresponds to a single input feature or neuron.
        self.weights = np.random.uniform(0, 1, (input_size + 1, output_size))
        # +1 because of the bias

        self.input_size = input_size
        self.output_size = output_size

        self._optimizer = None  # protected
        self._gradient_weights = None # protected

    def forward(self, input_tensor):
        # insert column for bias in x^T
        batch_size = input_tensor.shape[0] # how much rows in input_tensor
        bias_column = np.ones((batch_size, 1)) # create column with 1ones with size (batch_size, 1)

        # store x^T in forward pass to compute gradients later in backward pass
        self.input_tensor_including_bias = np.hstack((input_tensor, bias_column))

        # matmul: y^^T = x^T * W^T
        # matmul dimensions: (batch_size, input_size + 1) * (input_size + 1, output_size) -> (batch_size, output_size)
        return np.matmul(self.input_tensor_including_bias, self.weights)

    # getter pythonic
    @property
    def optimizer(self):
        return self._optimizer
    

    # setter pythonic
    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value


    def backward(self, error_tensor : np.ndarray) -> np.ndarray:
        # in BackPropagation compute: dL/dx = dL/dy * dy/dx

        # Compute gradient_tensor for calculate_update(...) as it is written in "1_BasicFramework.pdf"
        # gradient_weights = dL/dw
        # input_tensor_including_bias.T = x'^T
        # error_tensor = E_n = dL/dy^
        # dL/dw = x'^T * E_n
        self._gradient_weights = np.dot(self.input_tensor_including_bias.T, error_tensor)
        # (input_size + 1, output_size) = (intput_size + 1, batch_size)*(batch_size, output_size)

        # Apply optimizer if set
        if self._optimizer is not None:
            self.weights = self._optimizer.calculate_update(self.weights, self._gradient_weights)

        # Compute error to pass to previous layer (exclude bias weights)
        # error_prev_bias = E_{n-1} = dL/dx
        # error_tensor = E_n = dL/dy^
        # weights.t = w^T
        error_prev_bias = np.dot(error_tensor, self.weights.T)  # here the bias is included !
        # (batch_size, input_size + 1) = (batch_size, output_size)*(output_size, input_size + 1)

        error_prev = error_prev_bias[:, :-1]  # remove last column (bias term), because of derivative theres no more bias included
        return error_prev
        

    @property
    def gradient_weights(self):
        return self._gradient_weights
    
    @gradient_weights.setter
    def gradient_weights(self, gradient_weights):
        self._gradient_weights = gradient_weights
    
       
        
import numpy as np
from copy import deepcopy
from Layers.Base import BaseLayer

class BatchNormalization(BaseLayer):
    def __init__(self, channels):
        super().__init__()
        self.trainable = True
        self.channels = channels

        self.weights = np.ones(channels)  # gamma
        self.bias = np.zeros(channels)    # beta

        self._gradient_weights = None
        self._gradient_bias = None

        self.original_shape = None

        self._optimizer = None
        self._optimizer_weights = None
        self._optimizer_bias = None

        self.mean = None
        self.variance = None
        self.epsilon = 1e-15  # numeric stability

        self.moving_mean = None
        self.moving_variance = None
        self.momentum = 0.8  # for moving mean

        # for backward pass
        self.input_tensor = None
        self.normalized_tensor = None


    def initialize(self, weights_initializer, bias_initializer):
        self.weights = np.ones(self.channels)
        self.bias = np.zeros(self.channels)


    def reformat(self, tensor):
        # batch norm is done for each channel
        # (num_images, num_channels, height, width) --> (num_images*height*width, num_channels)
        if tensor.ndim == 4:
            # (N, C, H, W) → (N*H*W, C)
            self.original_shape = tensor.shape
            N, C, H, W = tensor.shape
            return tensor.transpose(0, 2, 3, 1).reshape(-1, C)
        elif self.original_shape is not None:
            N, C, H, W = self.original_shape
            return tensor.reshape(N, H, W, C).transpose(0, 3, 1, 2)
        else:
            return tensor


    def forward(self, input_tensor):
        # reformat input tensor
        reformatted = self.reformat(input_tensor)
        self.input_tensor = reformatted # save input tensor for backprob

        if self.testing_phase:
            # while testing do not update mean and variance values
            mean = self.moving_mean
            variance = self.moving_variance
        else:
            mean = np.mean(reformatted, axis=0)
            variance = np.var(reformatted, axis=0)

            if self.moving_mean is None: # first iteration
                self.moving_mean = mean
                self.moving_variance = variance
            else: # every other iteration
                self.moving_mean = self.momentum * self.moving_mean + (1 - self.momentum) * mean
                self.moving_variance = self.momentum * self.moving_variance + (1 - self.momentum) * variance

        self.mean = mean
        self.variance = variance

        # calculate normalized tensor
        self.normalized_tensor = (reformatted - mean) / np.sqrt(variance + self.epsilon)
        out = self.weights * self.normalized_tensor + self.bias

        # return reformated output tensor (N, C, H, W)
        return self.reformat(out)
    

    def backward(self, error_tensor):
        from Layers.Helpers import compute_bn_gradients

        # (N, C, H, W) --> (N*H*W, C)
        error_reformatted = self.reformat(error_tensor)

        # gradient w.r.t. weights
        self._gradient_weights = np.sum(error_reformatted * self.normalized_tensor, axis=0)
        # gradient w.r.t. bias
        self._gradient_bias = np.sum(error_reformatted, axis=0)
        
        # calculate weight and bias update
        if self._optimizer_weights is not None:
            self.weights = self._optimizer_weights.calculate_update(self.weights, self._gradient_weights)
        if self._optimizer_bias is not None:
            self.bias = self._optimizer_bias.calculate_update(self.bias, self._gradient_bias)

        # gradient w.r.t. input
        error_prev = compute_bn_gradients(error_reformatted, self.input_tensor, self.weights, self.mean, self.variance, self.epsilon)

        return self.reformat(error_prev)
    

    @property
    def optimizer(self):
        return self._optimizer
    
    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
        self._optimizer_weights = deepcopy(optimizer)
        self._optimizer_bias = deepcopy(optimizer)

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, gradient_weights):
        self._gradient_weights = gradient_weights

    @property
    def gradient_bias(self):
        return self._gradient_bias

    @gradient_weights.setter
    def gradient_weights(self, gradient_bias):
        self._gradient_bias = gradient_bias
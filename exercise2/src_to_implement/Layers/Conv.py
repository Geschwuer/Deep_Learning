from Layers.Base import BaseLayer
import numpy as np
from scipy.signal import correlate, correlate2d


class Conv(BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        super().__init__()

        self.trainable = True
        self.stride_shape = stride_shape if isinstance(stride_shape, tuple) else (stride_shape,)
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels

        if len(convolution_shape) == 2:
            self.is_1d = True
            self.num_input_channels, kernel_size = convolution_shape
            self.kernel_size = (kernel_size,)
        elif len(convolution_shape) == 3:
            self.is_1d = False
            self.num_input_channels, *kernel_size = convolution_shape
            self.kernel_size = kernel_size
        else:
            raise ValueError("Invalid convolution shape!")

        weights_shape = (num_kernels, self.num_input_channels, *self.kernel_size)
        self.weights = np.random.uniform(0, 1, size=weights_shape)
        self.bias = np.random.uniform(0, 1, size=num_kernels)

        self._optimizer_weights = None
        self._optimizer_bias = None
        self._gradient_weights = None
        self._gradient_bias = None
        self.input_tensor = None

    def initialize(self, weights_initializer, bias_initializer):
        weights_shape = self.weights.shape
        bias_shape = self.bias.shape
        fan_in = np.prod(self.kernel_size) * self.num_input_channels
        fan_out = self.num_kernels * np.prod(self.kernel_size)

        self.weights = weights_initializer.initialize(weights_shape, fan_in, fan_out)
        self.bias = bias_initializer.initialize(bias_shape, fan_in, fan_out)

    def same_padding(self, input_size, kernel_size, stride):
        output_size = int(np.ceil(input_size / stride))
        pad_total = max((output_size - 1) * stride + kernel_size - input_size, 0)
        pad_before = pad_total // 2
        pad_after = pad_total - pad_before
        return pad_before, pad_after, output_size

    def forward(self, input_tensor):
        self.input_tensor = input_tensor

        if self.is_1d:
            batch_size, num_channels, input_width = input_tensor.shape
            kernel_width = self.kernel_size[0]
            stride = self.stride_shape[0]
            pad_left, pad_right, output_width = self.same_padding(input_width, kernel_width, stride)
            padded_input = np.pad(input_tensor, ((0, 0), (0, 0), (pad_left, pad_right)), mode='constant')
            output = np.zeros((batch_size, self.num_kernels, output_width))
            for b in range(batch_size):
                for k in range(self.num_kernels):
                    for c in range(num_channels):
                        corr = correlate(padded_input[b, c], self.weights[k, c], mode="valid")
                        output[b, k] += corr[::stride]
                    output[b, k] += self.bias[k]
        else:
            batch_size, num_channels, input_height, input_width = input_tensor.shape
            stride_y, stride_x = self.stride_shape
            output_height = int(np.ceil(input_height / stride_y))
            output_width = int(np.ceil(input_width / stride_x))
            output = np.zeros((batch_size, self.num_kernels, output_height, output_width))
            for b in range(batch_size):
                for k in range(self.num_kernels):
                    for c in range(num_channels):
                        corr = correlate2d(input_tensor[b, c], self.weights[k, c], mode="valid")
                        output[b, k] += corr[::stride_y, ::stride_x]
                    output[b, k] += self.bias[k]
        return output

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @property
    def gradient_bias(self):
        return self._gradient_bias

    @property
    def optimizer_weights(self):
        return self._optimizer_weights

    @property
    def optimizer_bias(self):
        return self._optimizer_bias

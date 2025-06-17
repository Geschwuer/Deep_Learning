from Layers.Base import BaseLayer
import numpy as np

class Pooling(BaseLayer):
    def __init__(self, stride_shape, pooling_shape):
        super().__init__()
        self.stride_shape = stride_shape if isinstance(stride_shape, tuple) else (stride_shape,)
        self.pooling_shape = pooling_shape
        self.input_tensor = None
        self.mask = None    # array mask to keep track of max index


    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        batch_size, num_channels, input_width, input_height = input_tensor.shape
        stride_x, stride_y = self.stride_shape
        pooling_width, pooling_height = self.pooling_shape

        output_width = (input_width - pooling_width) // stride_x + 1
        output_height = (input_height - pooling_height) // stride_y + 1
        pooled_output_tensor = np.zeros(batch_size, num_channels, output_width, output_height)

        for b in range(batch_size):
            for c in range(num_channels):
                for x in range(0, output_width - 1, stride_x):
                    for y in range(0, output_height - 1, stride_y):
                        x_start = x
                        x_end = x_start + pooling_width
                        y_start = y
                        y_end = y_start + pooling_height
                        window = input_tensor[b, c, x_start:x_end, y_start:y_end]
                        max_val = max(window)
                        pooled_output_tensor[b, c, x, y] = max_val
                        mask = (window == np.max(window)) # returns bool Array with true at max pos
                        self.mask[b, c, x_start:x_end, y_start:y_end] = mask.astype(np.float32)

        return pooled_output_tensor


    def backward(self, error_tensor):
        batch_size, num_channels, output_width, output_height = error_tensor.shape
        stride_x, stride_y = self.stride_shape
        pooling_width, pooling_height = self.pooling_shape

        error_prev = np.zeros_like(self.input_tensor)

        for b in range(batch_size):
            for c in range(num_channels):
                for x in range(0, output_width - 1, stride_x):
                    for y in range(0, output_height - 1, stride_y):
                        x_start = x
                        x_end = x_start + pooling_width
                        y_start = y
                        y_end = y_start + pooling_height
                        window_mask = self.mask[b, c, x_start:x_end, y_start:y_end]
                        error = error_tensor[b, c, x, y]
                        error_prev[b, c, x_start:x_end, y_start:y_end] += window_mask * error

        return error_prev
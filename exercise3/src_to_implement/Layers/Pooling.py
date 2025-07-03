from Layers.Base import BaseLayer
import numpy as np

class Pooling(BaseLayer):
    def __init__(self, stride_shape, pooling_shape):
        super().__init__()
        self.stride_shape = stride_shape if isinstance(stride_shape, tuple) else (stride_shape,)
        self.pooling_shape = pooling_shape
        self.input_tensor = None
        self.max_indices = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        batch_size, num_channels, input_width, input_height = input_tensor.shape
        stride_x, stride_y = self.stride_shape
        pooling_width, pooling_height = self.pooling_shape

        output_width = (input_width - pooling_width) // stride_x + 1
        output_height = (input_height - pooling_height) // stride_y + 1
        output = np.zeros((batch_size, num_channels, output_width, output_height))
        self.max_indices = {}

        for b in range(batch_size):
            for c in range(num_channels):
                for x in range(output_width):
                    for y in range(output_height):
                        x_start = x * stride_x
                        y_start = y * stride_y
                        x_end = x_start + pooling_width
                        y_end = y_start + pooling_height

                        window = input_tensor[b, c, x_start:x_end, y_start:y_end]
                        max_pos = np.unravel_index(np.argmax(window), window.shape)
                        output[b, c, x, y] = window[max_pos]
                        self.max_indices[(b, c, x, y)] = (x_start + max_pos[0], y_start + max_pos[1])

        return output

    def backward(self, error_tensor):
        batch_size, num_channels, output_width, output_height = error_tensor.shape
        error_prev = np.zeros_like(self.input_tensor)

        for b in range(batch_size):
            for c in range(num_channels):
                for x in range(output_width):
                    for y in range(output_height):
                        x_idx, y_idx = self.max_indices[(b, c, x, y)]
                        error_prev[b, c, x_idx, y_idx] += error_tensor[b, c, x, y]

        return error_prev

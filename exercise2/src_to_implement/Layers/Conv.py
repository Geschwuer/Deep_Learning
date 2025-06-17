from Layers.Base import BaseLayer
import numpy as np
from scipy.signal import correlate, correlate2d


class Conv(BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        super().__init__()
        """
        - convolution_shape: [num_channels, kernel_height, kernel_width]
        - num_kernels: number of kernels applied to input (=number ouput_feature_maps)
        - stride_shape: 
        """
        self.trainable = True
        self.stride_shape = stride_shape if isinstance(stride_shape, tuple) else (stride_shape,) #always safe as tuple
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels

        # check if conv is 1D or 2D
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
        
        # init weights
        weights_shape = (num_kernels, self.num_input_channels, *self.kernel_size)
        self.weights = np.random.uniform(0, 1, size=weights_shape)
        self.bias = np.random.uniform(0, 1, size=num_kernels) # output(h) = (input * filter) + bias(h)

        # prepare optimizers and gradients
        self._optimizer_weights = None
        self._optimizer_bias = None
        self._gradient_weights = None
        self._gradient_bias = None

        # prepare input tensor (backward pass)
        self.input_tensor = None



    

    def same_padding(self, input_size, kernel_size, stride):
        """
        Computes the amount of padding needed for 'same' padding.
        """
        output_size = int(np.ceil(input_size / stride))
        pad_total = max((output_size - 1) * stride + kernel_size - input_size, 0)
        pad_before = pad_total // 2
        pad_after = pad_total - pad_before
        return pad_before, pad_after, output_size


    def forward(self, input_tensor):
        """
        Forward pass of convolutional layer

        Parameters:
        - input_tensor: np.ndarray
            Input tensor for the layer
                - 1D Convolution: shape (batch_size, num_channels, width); e.g. Audio Signals
                - 2D Convolution: shape (batch_size, num_channels, width, height); e.g. Images

        Returns:
        - output_tensor: np.ndarray
            Input tensor for the next layer
                - 1D Convolution: shape(batch_size, num_kernels, output_width)
                - 2D Convolution: shape(batch_size, num_kenels, output_width, output_height)
        """
        self.input_tensor = input_tensor # save for backprop

        if self.is_1d:
            batch_size, num_channels, input_width = input_tensor.shape
            kernel_width = self.kernel_size[0]
            stride = self.stride_shape[0]
            # calculate padding
            pad_left, pad_right, output_width = self.same_padding(input_width, kernel_width, stride)
            padded_input = np.pad(
                input_tensor,
                pad_width=((0, 0),  # batch-dimension --> no padding
                           (0, 0),  # channels-dimension --> no padding
                           (pad_left, pad_right)),
                mode='constant',    # pad with constant values
                constant_values=0   # use 0 as padding value
            )
            # prepare output
            output = np.zeros((batch_size, self.num_kernels, output_width))
            # apply kernels to input
            for b in range(batch_size):
                for k in range(self.num_kernels):
                    for c in range(num_channels):
                        # instead of padding with np we can also use mode="same" here
                        corr = correlate(padded_input[b, c], self.weights[k, c], mode="valid") 
                        output[b, k] += corr[::stride] # stride via slicing
                    output[b, k] += self.bias[k] # sum up over all channels e.g. 3 channels for RGB

        else:
            batch_size, num_channels, input_width, input_height = input_tensor.shape
            stride_x, stride_y = self.stride_shape
            output_width = int(np.ceil(input_width / stride_x))
            output_height = int(np.ceil(input_height / stride_y))
            # here we use padding from scipy.correlate
            # prepare output
            output = np.zeros((batch_size, self.num_kernels, output_width, output_height))
            # apply kernels to input
            for b in range(batch_size):
                for k in range(self.num_kernels):
                    for c in range(num_channels):
                        # instead of padding with np we can also use mode="same" here
                        corr = correlate2d(input_tensor[b, c], self.weights[k, c], mode="valid") 
                        output[b, k] += corr[::stride_x, ::stride_y] # stride via slicing
                    output[b, k] += self.bias[k] # sum up over all channels e.g. 3 channels for RGB
        return output


    def backward(self, error_tensor):
        # handle 1D case
        if self.is_1d:
            # dL/db
            # sum over batch and width axis --> one error value for each kernel
            self._gradient_bias = np.sum(error_tensor, axis=(0, 2)) # 1D: (b, k, w)
            # dL/dw
            self._gradient_weights = np.zeros_like(self.weights)

            batch_size = error_tensor.shape[0]
            stride = self.stride_shape[0]
            kernel_width = self.kernel_size[0]
            num_input_channels = self.input_tensor.shape[1] # get number of input channels
            input_width = self.input_tensor.shape[2]

            # - Manual padding needed in backward pass
            # - Ensures correct alignment of input and error tensor
            # - Mimics "same" padding from forward pass
            # - Padding with half the kernel width on both sides
            # - Needed to get properly shaped weight gradients using correlation
            pad = self.kernel_size[0] - 1
            padded_input = np.pad(
                self.input_tensor,
                pad_width=((0, 0),  # batch-dimension --> no padding
                           (0, 0),  # channels-dimension --> no padding
                           (pad, pad)),
                mode='constant',    # pad with constant values
                constant_values=0   # use 0 as padding value
            )

            for k in range(self.num_kernels):
                for c in range(num_input_channels):
                    grad = np.zeros(kernel_width)
                    for b in range(batch_size):
                        grad += correlate(padded_input[b, c], error_tensor[b, k], mode='valid')
                    self._gradient_weights[k, c] = grad

            # calculate weight and bias update
            if self._optimizer_bias is not None:
                self.bias = self._optimizer_bias.calculate_update(self.bias, self._gradient_bias)
            if self._optimizer_weights is not None:
                self.weights = self._optimizer_weights.calculate_update(self.weights, self._gradient_weights)

            # calculate error_prev dL/dx
            padded_error = np.pad(
                error_tensor,
                pad_width=((0, 0),  # batch-dimension --> no padding
                           (0, 0),  # channels-dimension --> no padding
                           (pad, pad)),
                mode='constant',    # pad with constant values
                constant_values=0   # use 0 as padding value
            )
            
            error_prev = np.zeros_like(self.input_tensor)

            # convolve with kernel or correlate with flipped kernel
            for b in range(batch_size):
                for c in range(num_input_channels):
                    for k in range(self.num_kernels):
                        rotated_kernel = np.flip(self.weights[k, c]) # flip kernel --> conv in backward
                        corr = correlate(padded_error[b, k], rotated_kernel, mode="valid")
                        error_prev[b, c] += corr[::stride] # stride!!!

        # handle 2D case
        else:
            # dL/db
            # sum over batch, width and height axis --> one error value for each kernel
            self._gradient_bias = np.sum(error_tensor, axis=(0, 2, 3)) # 1D: (b, k, w, h)
            # dL/dw
            self._gradient_weights = np.zeros_like(self.weights)

            batch_size, num_input_channels, input_width, input_height = self.input_tensor.shape
            kernel_width, kernel_height = self.kernel_size
            stride_x, stride_y = self.stride_shape

            # padding of input_tensor
            pad_x = self.kernel_size[0] - 1
            pad_y = self.kernel_size[1] - 1

            padded_input = np.pad(
                self.input_tensor,
                pad_width=((0, 0),  # batch-dimension --> no padding
                           (0, 0),  # channels-dimension --> no padding
                           (pad_x, pad_x),
                           (pad_y, pad_y)),
                mode='constant',    # pad with constant values
                constant_values=0   # use 0 as padding value
            )

            # compute gradient w.r.t. weights
            for k in range(self.num_kernels):
                for c in range(num_input_channels):
                    grad = np.zeros((kernel_width, kernel_height))
                    for b in range(batch_size):
                        grad += correlate2d(padded_input[b, c], error_tensor[b, k], mode='valid')
                    self._gradient_weights[k, c] = grad

            # calculate weight and bias update
            if self._optimizer_bias is not None:
                self.bias = self._optimizer_bias.calculate_update(self.bias, self._gradient_bias)
            if self._optimizer_weights is not None:
                self.weights = self._optimizer_weights.calculate_update(self.weights, self._gradient_weights)

            # calculate dL/dx
            padded_error = np.pad(
                error_tensor,
                pad_width=((0, 0),  # batch-dimension --> no padding
                           (0, 0),  # channels-dimension --> no padding
                           (pad_x, pad_x),
                           (pad_y, pad_y)),
                mode='constant',    # pad with constant values
                constant_values=0   # use 0 as padding value
            )
            
            error_prev = np.zeros_like(self.input_tensor)

            # convolve with kernel or correlate with flipped kernel
            for b in range(batch_size):
                for c in range(num_input_channels):
                    for k in range(self.num_kernels):
                        rotated_kernel = np.flip(self.weights[k, c], axis=(0,1)) # flip kernel both axes --> conv in backward
                        corr = correlate2d(padded_error[b, k], rotated_kernel, mode="valid")
                        error_prev[b, c] += corr[::stride_x, ::stride_y] # stride!!!

        return error_prev


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
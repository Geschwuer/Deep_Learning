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
        self.stride_shape = stride_shape
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
        self._optimizer = None
        self._gradient_weights = None
        self._gradient_bias = None

        # prepare input tensor (backward pass)
        self.input_tensor = None


    def initialize(self, weights_initializer, bias_initializer):
            fan_in = self.num_input_channels * np.prod(self.kernel_size)
            fan_out = self.num_kernels * np.prod(self.kernel_size)
            self.weights = weights_initializer.initialize(self.weights.shape, fan_in, fan_out)
            self.bias = bias_initializer.initialize(self.bias.shape, fan_in, fan_out)


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
            output_width = int(np.ceil(input_width / stride))
            output = np.zeros((batch_size, self.num_kernels, output_width))
            # apply kernels to input
            for b in range(batch_size):
                for k in range(self.num_kernels):
                    for c in range(num_channels):
                        # instead of padding with np we can also use mode="same" here
                        corr = correlate(input_tensor[b, c], self.weights[k, c], mode="same") 
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
                        corr = correlate2d(input_tensor[b, c], self.weights[k, c], mode="same") 
                        output[b, k] += corr[::stride_x, ::stride_y] # stride via slicing
                    output[b, k] += self.bias[k] # sum up over all channels e.g. 3 channels for RGB
        return output


    def _upsampling_1D(self, tensor, stride):
        batch_size, num_kernels, tensor_width = tensor.shape
        upsampled_width = (tensor_width - 1) * stride + 1
        upsampled_tensor = np.zeros((batch_size, num_kernels, upsampled_width))
        upsampled_tensor[:, :, ::stride] = tensor
        return upsampled_tensor
    

    def _upsampling_2D(self, tensor, stride):
        batch_size, num_kernels, tensor_width, tensor_height = tensor.shape
        stride_x, stride_y = stride
        upsampled_width = (tensor_width - 1) * stride_x + 1
        upsampled_height = (tensor_height - 1) * stride_y + 1
        upsampled_tensor = np.zeros((batch_size, num_kernels, upsampled_width, upsampled_height))
        upsampled_tensor[:, :, ::stride_x, ::stride_y] = tensor

        #     # 🔍 Spezialfall: stride_x ≠ stride_y kann zu Shape-Mismatch führen
        # if stride_x != stride_y:
        #     # Prüfen, ob in der Breite (Achse 3) 1 Spalte fehlt, um korrekte correlate2d-Größe zu erhalten
        #     expected_width = upsampled_tensor.shape[2]
        #     expected_height = upsampled_tensor.shape[3]

        #     # Diese Information ist nur beim Rückwärtslauf vorhanden:
        #     # Man kann alternativ direkt im backward() prüfen, ob correlate2d falsch rauskommt.
        #     # Aber einfacher Workaround:
        #     if expected_height % stride_y != 1:
        #         # Pad rechts 1 Spalte mit 0
        #         upsampled_tensor = np.pad(
        #             upsampled_tensor,
        #             pad_width=((0, 0), (0, 0), (0, 0), (0, 1)),  # nur rechts in Achse 3
        #             mode='constant'
        #         )    
        return upsampled_tensor


    def _reconstrct_padding_1D(self, tensor, input_width, kernel_width):
        tensor_width = tensor.shape[2]
        pad_total = (input_width + kernel_width - 1) - tensor_width
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        padded_tensor = np.pad(
            tensor,
            pad_width=((0, 0), (0, 0), (pad_left, pad_right)),
            mode='constant'
        )
        return padded_tensor
    

    def _reconstruct_padding_2D(self, tensor, input_width, input_height, kernel_width, kernel_height):
        tensor_width = tensor.shape[2]
        tensor_height = tensor.shape[3]

        pad_total_x = (input_width + kernel_width - 1) - tensor_width
        pad_left = pad_total_x // 2
        pad_right = pad_total_x - pad_left

        pad_total_y = (input_height + kernel_height - 1) - tensor_height
        pad_top = pad_total_y // 2
        pad_bottom = pad_total_y - pad_top

        padded_tensor = np.pad(
            tensor,
            pad_width=(
            (0, 0),
            (0, 0),
            (pad_left, pad_right),
            (pad_top, pad_bottom)),
            mode = 'constant'
        )
        return padded_tensor
    
    # def _reconstruct_padding_2D_input(self, tensor, upsampled_error, kernel_width, kernel_height):
    #     input_width = tensor.shape[2]
    #     input_height = tensor.shape[3]

    #     target_width = upsampled_error.shape[2] + kernel_width - 1
    #     target_height = upsampled_error.shape[3] + kernel_height - 1

    #     pad_total_x = target_width - input_width
    #     pad_left = pad_total_x // 2
    #     pad_right = pad_total_x - pad_left

    #     pad_total_y = target_height - input_height
    #     pad_top = pad_total_y // 2
    #     pad_bottom = pad_total_y - pad_top

    #     padded_tensor = np.pad(
    #         tensor,
    #         pad_width=((0, 0), (0, 0), (pad_left, pad_right), (pad_top, pad_bottom)),
    #         mode='constant'
    #     )
    #     return padded_tensor





    def backward(self, error_tensor):
        if self.is_1d:
            batch_size, num_kernels, error_width = error_tensor.shape
            stride = self.stride_shape[0]
            kernel_width = self.kernel_size[0]
            num_input_channels = self.input_tensor.shape[1]
            input_width = self.input_tensor.shape[2]

            # Bias gradient
            self._gradient_bias = np.sum(error_tensor, axis=(0, 2))  # (num_kernels,)
            
            # === 1. Upsampling of error_tensor ===
            upsampled_error = self._upsampling_1D(error_tensor, stride)

            # === 2. Reconstruct padding ===D
            padded_input = self._reconstrct_padding_1D(self.input_tensor, input_width, kernel_width)
            padded_upsampled_error = self._reconstrct_padding_1D(upsampled_error, input_width, kernel_width)

            # === 3. Grad w.r.t. Weights: dL/dw ===
            self._gradient_weights = np.zeros_like(self.weights)
            
            for b in range(batch_size):
                for k in range(num_kernels):
                    for c in range(num_input_channels):
                        self._gradient_weights[k, c] += correlate(
                            padded_input[b, c],
                            upsampled_error[b, k],
                            mode='valid'
                        )

            # === 4. Optimizer step ===
            if self._optimizer is not None:
                self.weights = self._optimizer.calculate_update(self.weights, self._gradient_weights)
                self.bias = self._optimizer.calculate_update(self.bias, self._gradient_bias)
                

            # === 5. Grad w.r.t. Input: dL/dx ===
            error_prev = np.zeros_like(self.input_tensor)
            weights_flipped = np.flip(self.weights, axis=2)

            for b in range(batch_size):
                for c in range(num_input_channels):
                    for k in range(num_kernels):
                        error_prev[b, c] += correlate(
                            padded_upsampled_error[b, k],
                            weights_flipped[k, c],
                            mode='valid'
                        )

        # handle 2D case
        else:
            # dL/db
            # sum over batch, width and height axis --> one error value for each kernel
            self._gradient_bias = np.sum(error_tensor, axis=(0, 2, 3)) # 1D: (b, k, w, h)
            # dL/dw
            self._gradient_weights = np.zeros_like(self.weights)

            batch_size, num_input_channels, input_width, input_height = self.input_tensor.shape
            kernel_width, kernel_height = self.kernel_size

            # === 1. Upsampling of error_tensor ===
            upsampled_error = self._upsampling_2D(error_tensor, self.stride_shape)
            
            # === 2. Reconstruct padding ===D
            padded_input = self._reconstruct_padding_2D(
                self.input_tensor, 
                input_width,
                input_height,
                kernel_width,
                kernel_height
                )
            

            expected_upsampled_width = padded_input.shape[2] - kernel_width + 1
            expected_upsampled_height = padded_input.shape[3] - kernel_height + 1

            actual_upsampled_width = upsampled_error.shape[2]
            actual_upsampled_height = upsampled_error.shape[3]

            pad_w = expected_upsampled_width - actual_upsampled_width
            pad_h = expected_upsampled_height - actual_upsampled_height

            if pad_w > 0 or pad_h > 0:
                upsampled_error = np.pad(
                    upsampled_error,
                    pad_width=((0, 0), (0, 0), (0, pad_w), (0, pad_h)),  # rechts/unten paddieren
                    mode='constant'
                )

            padded_upsampled_error = self._reconstruct_padding_2D(upsampled_error,
                                                                 input_width,
                                                                 input_height,
                                                                 kernel_width,
                                                                 kernel_height)

            #upsampled_error_flipped = np.flip(upsampled_error, axis=(2,3))

            # compute gradient w.r.t. weights
            for k in range(self.num_kernels):
                for c in range(num_input_channels):
                    for b in range(batch_size):
                        self._gradient_weights[k, c] += correlate2d(padded_input[b, c], 
                                                                    upsampled_error[b, k],
                                                                    mode='valid')
                    
            # calculate weight and bias update
            if self._optimizer is not None:
                self.weights = self._optimizer.calculate_update(self.weights, self._gradient_weights)
                self.bias = self._optimizer.calculate_update(self.bias, self._gradient_bias)

            # calculate dL/dx           
            error_prev = np.zeros_like(self.input_tensor)
            weights_flipped = np.flip(self.weights, axis=(2,3))

            # convolve with kernel or correlate with flipped kernel
            for b in range(batch_size):
                for c in range(num_input_channels):
                    for k in range(self.num_kernels):
                        error_prev[b, c] += correlate2d(padded_upsampled_error[b, k], 
                                           weights_flipped[k, c], 
                                           mode="valid")

        return error_prev


    @property
    def gradient_weights(self):
        return self._gradient_weights


    @property
    def gradient_bias(self):
        return self._gradient_bias
    

    @property
    def optimizer(self):
        return self._optimizer
    
    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
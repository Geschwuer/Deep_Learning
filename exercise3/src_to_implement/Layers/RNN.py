from Layers.Base import BaseLayer
from Layers.FullyConnected import FullyConnected
from Layers.TanH import TanH
from Layers.Sigmoid import Sigmoid
from Layers.Flatten import Flatten
import numpy as np
# import icecream as ic
from copy import deepcopy

class RNN(BaseLayer):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.trainable = True
        self.input_size = input_size    # size of input
        self.hidden_size = hidden_size  # size of hidden states
        self.output_size = output_size  # output size

        self._optimizer = None
        self._optimizer_output = None
        self._optimizer_hidden = None

        self.T = None

        self.h_t = None
        self.h_t_prev = None
        self.y_t = None

        self._memorize = False

        self.FC_hidden = FullyConnected(input_size=self.input_size + self.hidden_size, 
                                        output_size=self.hidden_size)
        self.FC_hidden_memory = []
        self.hidden_gradient_weights = None

        self.FC_output = FullyConnected(input_size=self.hidden_size,
                                        output_size=self.output_size)
        self.FC_output_memory = []
        self.output_gradient_weights = None
        
        self.flatten = Flatten()
        
        self.tanh = TanH()
        self.sigmoid = Sigmoid()

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        self.T = input_tensor.shape[0]

        # initialize some output variables
        self.y_t = np.zeros((self.T, self.output_size))
        self.h_t = np.zeros((self.T+1, self.hidden_size))

        if self.memorize and self.h_t_prev is not None:
            self.h_t[0] = self.h_t_prev

        # free memory everytime we do a forward pass
        self.FC_hidden_memory.clear()
        self.FC_output_memory.clear()

        for t in range(self.T):
            input_concat = np.concatenate((input_tensor[t], self.h_t[t])).reshape(1, -1)
            # calculate u_t
            u_t = self.FC_hidden.forward(input_concat)
            self.FC_hidden_memory.append(self.FC_hidden.input_tensor)
            # calculate new hidden state h_t
            self.h_t[t+1] = self.tanh.forward(u_t)
            # calculate o_t
            # h_t_flat = self.flatten.forward(self.h_t[t+1])
            h_t_flat = self.h_t[t+1].reshape(1, -1)
            o_t = self.FC_output.forward(h_t_flat)
            self.FC_output_memory.append(self.FC_output.input_tensor)
            # calculate output sigmoid(o_t)
            self.y_t[t] = self.sigmoid.forward(o_t)
        
        self.h_t_prev = self.h_t[-1]
        return self.y_t


    def backward(self, error_tensor):
        error_prev = np.zeros((self.T, self.input_size))
        h_t_deriv_next = np.zeros((1, self.hidden_size)) # initial values for∇h_{t+1}

        self.hidden_gradient_weights = np.zeros_like(self.FC_hidden.weights)
        self.output_gradient_weights = np.zeros_like(self.FC_output.weights)

        for t in reversed(range(self.T)):
            self.sigmoid.activation = self.y_t[t]
            error_tensor_flat = error_tensor[t].reshape(1, -1)
            sigmoid_deriv = self.sigmoid.backward(error_tensor_flat)

            self.FC_output.input_tensor = self.FC_output_memory[t]
            o_t_deriv = self.FC_output.backward(sigmoid_deriv)
            self.output_gradient_weights += self.FC_output.gradient_weights
            error_before_tanh = h_t_deriv_next + o_t_deriv

            self.tanh.activation = self.h_t[t+1]
            h_t_deriv = self.tanh.backward(error_before_tanh)

            self.FC_hidden.input_tensor = self.FC_hidden_memory[t]
            u_t_deriv = self.FC_hidden.backward(h_t_deriv)
            self.hidden_gradient_weights += self.FC_hidden.gradient_weights

            # u_t_deriv contains error from previous hidden state and error from input
            # we need to seperate them according to conacatination in forward pass
            h_t_deriv_next = u_t_deriv[:, self.input_size:]
            x_t_deriv = u_t_deriv[:, :self.input_size]
            error_prev[t] = x_t_deriv

        if self._optimizer_hidden is not None and self._optimizer_output is not None:
            self.FC_hidden.weights = self._optimizer_hidden.calculate_update(self.FC_hidden.weights, 
                                                                             self.hidden_gradient_weights)
            
            self.FC_output.weights = self._optimizer_output.calculate_update(self.FC_output.weights,
                                                                             self.output_gradient_weights)
        return error_prev


    def initialize(self, weights_initializer, bias_initializer):
        self.FC_hidden.initialize(weights_initializer, bias_initializer)
        self.FC_output.initialize(weights_initializer, bias_initializer)
        self.weights = self.FC_hidden.weights

        
    @property
    def optimizer(self):
        return self._optimizer
    
    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = deepcopy(optimizer)
        self._optimizer_hidden = deepcopy(optimizer)
        self._optimizer_output = deepcopy(optimizer)

    @property
    def memorize(self):
        return self._memorize
    
    @memorize.setter
    def memorize(self, val: bool):
        self._memorize = val

    @property
    def weights(self):
        return self.FC_hidden.weights

    @weights.setter
    def weights(self, weights):
        self.FC_hidden.weights = weights

    @property
    def gradient_weights(self):
        return self.hidden_gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, gradient_weights):
        self.FC_hidden._gradient_weights = gradient_weights
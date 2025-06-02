from Layers import FullyConnected, ReLU, SoftMax
from Optimization import Loss

import copy

class NeuralNetwork:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.loss : list = []
        self.layers : list = []
        self.data_layer = None
        self.loss_layer = None

        self.label_tensor = None


    def forward(self):
        input_tensor, self.label_tensor = self.data_layer.next()
        # Pass input through all layers in the network
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)

        # Compute final output using the loss layer
        return self.loss_layer.forward(input_tensor, self.label_tensor)    
    

    def backward(self):
        # Start the backward pass at the loss layer
        error_tensor = self.loss_layer.backward(self.label_tensor)

        # Propagate backwards through the network layers in reverse order
        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)


    def append_layer(self, layer):
        if layer.trainable:
            layer.optimizer = copy.deepcopy(self.optimizer)
        self.layers.append(layer)

    
    def train(self, iterations):
        for _ in range(iterations):
            loss = self.forward()
            self.loss.append(loss)
            self.backward()

    def test(self, input_tensor):
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
        return input_tensor
import copy
import numpy as np

class NeuralNetwork:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None

    def forward(self):
        input_tensor, label_tensor = self.data_layer.next()
        # save current input_tensor and label_tensor for backprop
        self.input_tensor_current = input_tensor
        self.label_tensor_current = label_tensor

        # input for next layer is output of current layer
        input = input_tensor
        for layer in self.layers:
            output = layer.forward(input)
            input = output
        # calculate loss using probs from softmax and labels
        loss_output = self.loss_layer.forward(input, label_tensor)
        return loss_output

    def backward(self):
        # calculate backstep of loss layer first
        error_tensor = self.loss_layer.backward(self.label_tensor_current)

        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)
        # no return needed bc every trainable layer does the weight update by calling "layer.backward"

    def append_layer(self, layer):
        # check if layer is trainable
        if hasattr(layer, 'trainable') and layer.trainable:
            # if layer is trainable add deep copy of optimizer to layer
            layer.optimizer = copy.deepcopy(self.optimizer)
        self.layers.append(layer)

    def train(self, iterations):
        # for each training clear the loss value
        self.loss.clear()
        # do forward and backward iterations
        for _ in range(iterations):
            loss_val = self.forward()
            self.loss.append(np.mean(loss_val)) # in case loss val is batch-wise
            self.backward() # update weights
        # also no need to return here

    def test(self, input_tensor):
        input = input_tensor
        for layer in self.layers:
            output = layer.forward(input)
            input = output
        return output
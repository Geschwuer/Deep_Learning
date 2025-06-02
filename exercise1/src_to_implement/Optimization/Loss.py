import numpy as np


class CrossEntropyLoss:
    def __init__(self):
        pass

    def forward(self, prediction_tensor, label_tensor):
        self.prediction_tensor = prediction_tensor
        self.label_tensor = label_tensor # label_tensor is one-hot-encoded. so only the log of the correct class probability is picked

        # instead of hardcoding we use finfo() to create a very small epsilon
        epsilon = np.finfo(float).eps

        # keep predictions away from 0 to avoid log(0)  --> clip between epsilon and 1 
        clipped_preds = np.clip(prediction_tensor, epsilon, 1)

        # dividing by the batch_size (prediction_tensor.shape[0]) gives the average loss
        loss = -np.sum(label_tensor * np.log(clipped_preds))
        return loss # returns a number, not a matrix


    def backward(self, label_tensor):
        # don't compute: dL/dx = dL/dy * dy/dx here, because CrossEntropy is not a real layer, 
        # it's just a loss function and the start of the backpropagation chain.
        # You don’t return values that are fed into another layer

        # just start the gradient flow by computing E_n = dL/dy
        E_n = -label_tensor / (self.prediction_tensor + np.finfo(float).eps)
        return E_n

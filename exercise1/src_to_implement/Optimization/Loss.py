import numpy as np

class CrossEntropyLoss():
    def __init__(self):
        self.prediction_tensor = None

    def forward(self, prediction_tensor, label_tensor):
        eps = np.finfo(float).eps
        # clip prediction to eps and 1-eps to ensure we never get 0 or 1 as prediction
        # this may cause numerical issues during backpropagation (Loss calculation)
        prediction_tensor_clipped = np.clip(prediction_tensor, eps, 1-eps)
        self.prediction_tensor = prediction_tensor_clipped
        # calculate loss
        loss = -np.sum(label_tensor * np.log(prediction_tensor_clipped), axis=1, keepdims=True)
        return loss

    def backward(self, label_tensor):
        eps = np.finfo(float).eps
        # use clipping again
        prediction_tensor_clipped = np.clip(self.prediction_tensor, eps, 1-eps)
        error_prev = -(label_tensor/prediction_tensor_clipped)
        return error_prev
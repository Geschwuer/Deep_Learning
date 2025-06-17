import numpy as np

class Sgd:
    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor: np.ndarray, gradient_tensor: np.ndarray) -> np.ndarray:
        updated_weight_tensor = weight_tensor - self.learning_rate * gradient_tensor
        return updated_weight_tensor
    

class SgdWithMomentum:
    def __init__(self, learning_rate, momentum_rate):
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate

        self.velocity = None

    def calculate_update(self, weight_tensor, gradient_tensor):
        # first initialize self.velocity with zeros
        if self.velocity is None:
            self.velocity = np.zeros_like(weight_tensor)
        
        # calculate velocity
        self.velocity = self.momentum_rate * self.velocity - self.learning_rate * gradient_tensor
        # calculate updated weights using average information from the last gradients --> accumulate gradient information over time
        return weight_tensor + self.velocity


class Adam:
    def __init__(self, learning_rate, mu, rho):
        self.learning_rate = learning_rate
        self.mu = mu        # beta1
        self.rho = rho      # beta2
        self.epsilon = 1e-8

        self.v = None           # first moment estimate (mean of gradients)
        self.r = None           # second moment estimate (squared gradients)
        self.k = 0              # iteration step counter

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.v is None:
            self.v = np.zeros_like(weight_tensor)
            self.r = np.zeros_like(weight_tensor)

        self.k += 1 # for bias correction

        # calculate first moment
        self.v = self.mu * self.v + (1-self.mu) * gradient_tensor

        # calculate second moment
        self.r = self.rho * self.r + (1-self.rho) * gradient_tensor * gradient_tensor

        # Compute bias-corrected estimates
        v_hat = self.v / (1 - self.mu ** self.k)
        r_hat = self.r / (1 - self.rho ** self.k)
        # why bias correction?
        # v and r (first and second moment) are initialized with zero, so at the beginning they are underestimated
        # --> with bias correction early gradients are corrected upward

        # calculate weight update
        return weight_tensor - self.learning_rate * v_hat / (np.sqrt(r_hat) + self.epsilon)
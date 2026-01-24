import numpy as np


class Optimizer:
    def __init__(self):
        self.regularizer = None

    def add_regularizer(self, regularizer):
        self.regularizer = regularizer


class Sgd(Optimizer):
    def __init__(self, learning_rate: float):
        super().__init__()
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor: np.ndarray, gradient_tensor: np.ndarray) -> np.ndarray:
        if self.regularizer is not None:
            shrinkage = self.regularizer.calculate_gradient(weight_tensor)
            weight_tensor -= self.learning_rate * shrinkage

        updated_weight_tensor = weight_tensor - self.learning_rate * gradient_tensor
        return updated_weight_tensor


class SgdWithMomentum(Optimizer):
    def __init__(self, learning_rate, momentum_rate):
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.velocity = None # init during first optimization step

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.velocity is None:
            self.velocity = np.zeros_like(gradient_tensor) # first optimization step

        if self.regularizer is not None:
            shrinkage = self.regularizer.calculate_gradient(weight_tensor)
            weight_tensor -= self.learning_rate * shrinkage

        # calculate new velocity vector
        self.velocity = (
            self.momentum_rate * self.velocity  # μ * v_{t-1}
            - self.learning_rate * gradient_tensor  # - η * ∇J(θ)
        )

        # update weights
        updated_weights = weight_tensor + self.velocity
        return updated_weights

        
class Adam(Optimizer):
    def __init__(self, learning_rate, mu, rho):
        super().__init__()
        self.learning_rate = learning_rate  # learning rate (η)
        self.mu = mu                        # decay rate for the first moment estimates (β1)
        self.rho = rho                      # decay rate for the second moment estimates (β2)
        self.v = None                      # first moment vector (mean of gradients)
        self.r = None                      # second moment vector (mean of squared gradients)
        self.t = 0                        # timestep counter
        self.eps = np.finfo(float).eps    # small epsilon value to prevent division by zero

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.v is None:
            # initialize first and second moment vectors with zeros of same shape as weights
            self.r = np.zeros_like(weight_tensor)
            self.v = np.zeros_like(weight_tensor)

        self.t += 1  # increment timestep

        if self.regularizer is not None:
            shrinkage = self.regularizer.calculate_gradient(weight_tensor)
            weight_tensor -= self.learning_rate * shrinkage

        # update biased first moment estimate
        self.v = self.mu * self.v + (1 - self.mu) * gradient_tensor
        # update biased second moment estimate
        self.r = self.rho * self.r + (1 - self.rho) * (gradient_tensor ** 2)

        # why bias correction?
        # v and r (first and second moment) are initialized with zero, so at the beginning they are underestimated
        # --> with bias correction early gradients are corrected upward
        # compute bias-corrected first moment estimate
        v_hat = self.v / (1 - self.mu ** self.t)
        # compute bias-corrected second moment estimate
        r_hat = self.r / (1 - self.rho ** self.t)

        # update weights using Adam formula
        updated_weights = weight_tensor - self.learning_rate * v_hat / (np.sqrt(r_hat) + self.eps)

        return updated_weights
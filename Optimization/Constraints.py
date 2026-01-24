import numpy as np

class L2_Regularizer:
    def __init__(self, alpha: float):
        self.alpha = alpha  # regularization strength

    def calculate_gradient(self, weights: np.ndarray) -> np.ndarray:
        # Gradient of ½ * α * ||w||² is α * w
        return self.alpha * weights

    def norm(self, weights: np.ndarray) -> float:
        # Regularization term: ½ * α * ||w||²
        #return 0.5 * self.alpha * np.sum(weights ** 2)
        return self.alpha * np.sum(weights ** 2)


class L1_Regularizer:
    def __init__(self, alpha: float):
        self.alpha = alpha  # regularization strength

    def calculate_gradient(self, weights: np.ndarray) -> np.ndarray:
        # Subgradient of α * ||w||₁ is α * sign(w)
        return self.alpha * np.sign(weights)

    def norm(self, weights: np.ndarray) -> float:
        # Regularization term: α * ||w||₁
        return self.alpha * np.sum(np.abs(weights))
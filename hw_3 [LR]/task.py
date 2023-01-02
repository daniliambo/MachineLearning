import numpy as np

# Task 1

def mse(y_true: np.ndarray, y_predicted: np.ndarray):
    return 0.0


def r2(y_true: np.ndarray, y_predicted: np.ndarray):
    return 0.0


# Task 2

class NormalLR:
    def __init__(self):
        self.weights = None  # Save weights here

    def fit(self, X: np.ndarray, y: np.ndarray):
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        pass


# Task 3

class GradientLR:
    def __init__(self, alpha: float, iterations=10000, l=0.):
        self.weights = None  # Save weights here

    def fit(self, X: np.ndarray, y: np.ndarray):
        pass

    def predict(self, X: np.ndarray):
        pass


# Task 4

def get_feature_importance(linear_regression):
    return []


def get_most_important_features(linear_regression):
    return []

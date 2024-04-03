import numpy as np


def logistic_regression(x: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    return sigmoid(np.dot(w.T, x) + b)


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))

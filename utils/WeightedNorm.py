import numpy as np


def weightedNorm(X, alpha):
    beta = (3 - alpha) / 2
    weights = np.array([beta, beta, alpha])
    return np.sqrt(np.sum(weights * X**2))


if __name__ == "__main__":
    X = np.arange(6).reshape(-1, 3)

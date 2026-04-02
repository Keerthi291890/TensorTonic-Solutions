import numpy as np


def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """

    n_samples, n_features = X.shape
    W = np.zeros(n_features)
    b=  0.0
    for step in range(steps):
        weighted_x = X.dot(W)+b
        y_hat = _sigmoid(weighted_x)
        grad_w = (1/n_samples)*X.T.dot(y_hat-y)
        grad_b = (1/n_samples)*np.sum(y_hat-y)
        b-=lr*grad_b
        W-=lr*grad_w
    return (W,b)
    pass
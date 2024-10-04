import numpy as np


# CR TODO: implement more dataset types
class Dataset:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    @classmethod
    def generate(cls, n_samples, noise):
        raise NotImplementedError


class Diagonal(Dataset):
    def __init__(self, X, y):
        super().__init__(X, y)

    @classmethod
    def generate(cls, n_samples=1000, noise=0.1):
        """Generate a 2D classification dataset with one-hot encoded labels."""
        X = np.random.randn(n_samples, 2)
        y_binary = (X[:, 0] + X[:, 1] > 0).astype(int)
        y = np.zeros((n_samples, 2))
        y[np.arange(n_samples), y_binary] = 1  # One-hot encoding
        X += np.random.randn(n_samples, 2) * noise
        return cls(X, y)

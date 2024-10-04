import numpy as np
import matplotlib.pyplot as plt


def visualize_2D_classification_data(dataset):
    X = dataset.X
    y = dataset.y
    """Visualize a 2D classification dataset."""
    plt.scatter(X[:, 0], X[:, 1], c=np.argmax(y, axis=1), cmap="viridis")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("2D Classification Dataset")
    plt.colorbar(label="Class")
    plt.show()

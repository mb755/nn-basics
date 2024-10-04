import numpy as np
from src.datasets import Diagonal
from src.layers import Dense, ReLU, Softmax
from src.network import NeuralNetwork


def test_dataset_generation():
    dataset = Diagonal.generate(n_samples=100)
    assert dataset.X.shape == (100, 2)
    assert dataset.y.shape == (100, 2)
    assert np.all((dataset.y == 0) | (dataset.y == 1))


def test_dense_layer():
    dense = Dense(2, 3)
    inputs = np.array([[1, 2]])
    output = dense.forward(inputs)
    assert output.shape == (1, 3)


def test_relu_layer():
    relu = ReLU()
    inputs = np.array([[-1, 0, 1]])
    output = relu.forward(inputs)
    assert np.array_equal(output, [[0, 0, 1]])


def test_softmax_layer():
    softmax = Softmax()
    inputs = np.array([[1, 2]])
    output = softmax.forward(inputs)
    assert np.isclose(np.sum(output), 1)


def test_network_forward():
    network = NeuralNetwork([Dense(2, 3), ReLU(), Dense(3, 2), Softmax()])
    inputs = np.array([[1, 2]])
    output = network.forward(inputs)
    assert output.shape == (1, 2)
    assert np.isclose(np.sum(output), 1)

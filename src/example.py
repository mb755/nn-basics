from datasets import Diagonal
from layers import Dense, ReLU, Softmax
from network import NeuralNetwork
from visualizations import (
    visualize_2D_classification_data,
    visualize_network,
    visualize_decision_boundary,
)

# Create a dataset
training_data = Diagonal.generate(1000, 0.1)
visualize_2D_classification_data(training_data)

test_data = Diagonal.generate(500, 0.1)

# Define network architecture
network = NeuralNetwork(
    [Dense(2, 4), ReLU(), Dense(4, 2), ReLU(), Dense(2, 2), Softmax()]
)

# Train network
network.train(training_data, test_data, epochs=1000, learning_rate=0.001)

# evaluate and visualize predictions
predictions = Diagonal(test_data.X, network.predict(test_data.X))

visualize_2D_classification_data(predictions)

visualize_network(network, test_data)

visualize_decision_boundary(network, test_data)

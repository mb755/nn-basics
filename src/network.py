import numpy as np


class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, grad_output, learning_rate):
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output, learning_rate)

    def train(self, training_data, test_data, epochs, learning_rate):
        X = training_data.X
        y = training_data.y

        test_X = test_data.X
        test_y = test_data.y

        for epoch in range(epochs):
            # Forward pass
            # have to call forward on test data first, since forward saves input for backprop
            test_output = self.forward(test_X)
            output = self.forward(X)

            # Compute loss (cross-entropy)
            training_loss = -np.sum(
                y * np.log(np.clip(output, 1e-15, 1 - 1e-15))
            ) / len(y)
            test_loss = -np.sum(
                test_y * np.log(np.clip(test_output, 1e-15, 1 - 1e-15))
            ) / len(test_y)

            # Backward pass
            grad_output = output - y
            self.backward(grad_output, learning_rate)

            if epoch % 100 == 0:
                print(
                    f"Epoch {epoch}, Training loss: {training_loss}, Test loss: {test_loss}"
                )

    def predict(self, X):
        return self.forward(X)

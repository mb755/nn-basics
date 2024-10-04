import numpy as np


class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        raise NotImplementedError

    def backward(self, output_gradient, learning_rate):
        raise NotImplementedError


class Dense(Layer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.bias = np.zeros((1, output_size))
        self.input_size = input_size
        self.output_size = output_size

    def forward(self, input):
        self.input = input
        return np.dot(input, self.weights) + self.bias

    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(self.input.T, output_gradient)
        input_gradient = np.dot(output_gradient, self.weights.T)

        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * np.sum(output_gradient, axis=0, keepdims=True)

        return input_gradient


class ReLU(Layer):
    def forward(self, input):
        self.input = input
        return np.maximum(0, input)

    def backward(self, output_gradient, learning_rate):
        return output_gradient * (self.input > 0)


class Softmax(Layer):
    def forward(self, input):
        exp_values = np.exp(input - np.max(input, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return self.output

    def backward(self, output_gradient, learning_rate):
        # For simplicity, we assume this layer is used with cross-entropy loss,
        # so the gradient is simplified to the difference between predicted and true probabilities
        return output_gradient


class Sigmoid(Layer):
    def forward(self, input):
        self.output = 1 / (1 + np.exp(-input))
        return self.output

    def backward(self, output_gradient, learning_rate):
        return output_gradient * self.output * (1 - self.output)


class Tanh(Layer):
    def forward(self, input):
        self.output = np.tanh(input)
        return self.output

    def backward(self, output_gradient, learning_rate):
        return output_gradient * (1 - np.power(self.output, 2))


class Dropout(Layer):
    def __init__(self, rate):
        super().__init__()
        self.rate = rate
        self.mask = None

    def forward(self, input):
        self.mask = np.random.binomial(1, 1 - self.rate, input.shape) / (1 - self.rate)
        return input * self.mask

    def backward(self, output_gradient, learning_rate):
        return output_gradient * self.mask


class BatchNormalization(Layer):
    def __init__(self, input_shape, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon
        self.gamma = np.ones(input_shape)
        self.beta = np.zeros(input_shape)

    def forward(self, input):
        self.input = input
        self.mean = np.mean(input, axis=0)
        self.var = np.var(input, axis=0)
        self.normalized = (input - self.mean) / np.sqrt(self.var + self.epsilon)
        return self.gamma * self.normalized + self.beta

    def backward(self, output_gradient, learning_rate):
        input_size = self.input.shape[0]

        gamma_gradient = np.sum(output_gradient * self.normalized, axis=0)
        beta_gradient = np.sum(output_gradient, axis=0)

        normalized_gradient = output_gradient * self.gamma
        var_gradient = (
            np.sum(normalized_gradient * (self.input - self.mean), axis=0)
            * -0.5
            * (self.var + self.epsilon) ** (-1.5)
        )
        mean_gradient = np.sum(
            normalized_gradient * -1 / np.sqrt(self.var + self.epsilon), axis=0
        ) + var_gradient * np.mean(-2 * (self.input - self.mean), axis=0)

        input_gradient = (
            normalized_gradient / np.sqrt(self.var + self.epsilon)
            + var_gradient * 2 * (self.input - self.mean) / input_size
            + mean_gradient / input_size
        )

        self.gamma -= learning_rate * gamma_gradient
        self.beta -= learning_rate * beta_gradient

        return input_gradient

import random
import numpy as np
import _pickle as pickle
import gzip


def __sigmoid__(Z):
    return 1.0 / (1.0 + np.exp(-Z))


def __derive_sigmoid__(Z):
    return __sigmoid__(Z) * (1 - __sigmoid__(Z))


def cost_derivative(output_activations, y):
    return output_activations - y


class NonConvolutionalNeuralNetwork(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.rand(y, 1) * 2 - 1 for y in sizes[1:]]
        self.weights = [np.random.rand(y, x) * 2 - 1 for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, A):
        for b, w in zip(self.biases, self.weights):
            A = __sigmoid__(np.dot(w, A) + b)
        return A

    def SGD(self, data, iterations, batch_size, lr):
        for epoch in range(iterations):
            random.shuffle(data)
            batches = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
            for batch in batches:
                self.process_batch(batch, batch_size, lr)
            print(f"Epoch {epoch} complete")

    def process_batch(self, batch, batch_size, lr):
        gradient_biases = [np.zeros(b.shape) for b in self.biases]
        gradient_weights = [np.zeros(w.shape) for w in self.weights]
        for values, label in batch:
            delta_grad_b, delta_grad_w = self.backprop(values, label)
            gradient_weights = [gw + dgw for gw, dgw in zip(gradient_weights, delta_grad_w)]
            gradient_biases = [gb + dgb for gb, dgb in zip(gradient_biases, delta_grad_b)]
        self.weights = [weight - (lr / batch_size) * grad_w for weight, grad_w in zip(self.weights, gradient_weights)]
        self.biases = [bias - (lr / batch_size) * grad_b for bias, grad_b in zip(self.biases, gradient_biases)]

    def backprop(self, values, label):
        gradient_biases = [np.zeros(b.shape) for b in self.biases]
        gradient_weights = [np.zeros(w.shape) for w in self.weights]
        activation = values
        activations = [values]
        Z_vectors = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            Z_vectors.append(z)
            activation = __sigmoid__(z)
            activations.append(activation)
        delta = cost_derivative(activations[-1], label) * __derive_sigmoid__(Z_vectors[-1])
        gradient_biases[-1] = delta
        gradient_weights[-1] = np.dot(delta, activations[-2].T)
        for i in range(2, self.num_layers):
            z = Z_vectors[-i]
            sp = __derive_sigmoid__(z)
            delta = np.dot(self.weights[-i + 1].T, delta) * sp
            gradient_biases[-i] = delta
            gradient_weights[-i] = np.dot(delta, activations[-i - 1].T)
        return gradient_biases, gradient_weights

    def evaluate_accuracy(self, test_data):
        return sum(int(x == y) for (x, y) in
                   [(np.argmax(self.feedforward(x)), np.argmax(y)) for (x, y) in test_data]) / len(test_data)


def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


f = gzip.open('mnist.pkl.gz', 'rb')
tr_d, test_d, _ = pickle.load(f, encoding='latin1')

f.close()
training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
training_results = [vectorized_result(y) for y in tr_d[1]]
training_data = list(zip(training_inputs, training_results))[:4000]
test_inputs = [np.reshape(x, (784, 1)) for x in test_d[0]]
test_results = [vectorized_result(y) for y in test_d[1]]
test = list(zip(test_inputs, test_results))[:4000]
network = NonConvolutionalNeuralNetwork([784, 20, 10])
network.SGD(training_data, 50, 20, 1)

print(network.evaluate_accuracy(test))

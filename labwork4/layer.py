from neuron import Neuron


class Layer:
    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad, lr):
        raise NotImplementedError


class InputLayer(Layer):
    def forward(self, x):
        return x

    def backward(self, grad, lr):
        return grad


class HiddenLayer(Layer):
    def __init__(self, input_dim, n_neurons, activation):
        self.neurons = [Neuron(input_dim) for _ in range(n_neurons)]
        self.activation = activation

    def forward(self, inputs):
        self.inputs = inputs
        self.z = []
        self.outputs = []

        for neuron in self.neurons:
            z_i = neuron.ws(inputs)
            self.z.append(z_i)
            self.outputs.append(self.activation.forward(z_i))

        return self.outputs

    def backward(self, grad_output, lr):
        assert len(grad_output) == len(self.neurons)

        x = [1.0] + self.inputs
        grad_input = [0.0] * len(self.inputs)

        for i, neuron in enumerate(self.neurons):
            a = self.outputs[i]
            dz = grad_output[i] * a * (1 - a)

            old_weights = neuron.weights.copy()

            for j in range(len(neuron.weights)):
                neuron.weights[j] -= lr * dz * x[j]

            for j in range(1, len(old_weights)):
                grad_input[j - 1] += dz * old_weights[j]

        return grad_input


class OutputLayer(Layer):
    def __init__(self, input_dim, n_neurons, activation):
        self.neurons = [Neuron(input_dim) for _ in range(n_neurons)]
        self.activation = activation

    def forward(self, inputs):
        self.inputs = inputs
        self.z = []
        self.outputs = []

        for neuron in self.neurons:
            z_i = neuron.ws(inputs)
            self.z.append(z_i)
            self.outputs.append(self.activation.forward(z_i))

        return self.outputs

    def backward(self, grad_output, lr):
        assert len(grad_output) == len(self.neurons)

        x = [1.0] + self.inputs
        grad_input = [0.0] * len(self.inputs)

        for i, neuron in enumerate(self.neurons):
            dz = grad_output[i]
            old_weights = neuron.weights.copy()

            for j in range(len(neuron.weights)):
                neuron.weights[j] -= lr * dz * x[j]

            for j in range(1, len(old_weights)):
                grad_input[j - 1] += dz * old_weights[j]

        return grad_input

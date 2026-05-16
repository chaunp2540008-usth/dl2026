from random import random


class Neuron :
    def __init__(self, input_dim):
        self.weights = [random() - 0.5 for _ in range(input_dim + 1)]

    # weighted sum of inputs + bias
    def ws(self, inputs) -> float:
        x_with_bias = [1.0] + list(inputs)
        return sum(w * x for w, x in zip(self.weights, x_with_bias))

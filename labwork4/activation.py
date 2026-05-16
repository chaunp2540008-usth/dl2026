import math


class ActivationFn:
    def forward(self, x):
        raise NotImplementedError

    def derivative(self, x):
        raise NotImplementedError


class Sigmoid(ActivationFn):
    def forward(self, x):
        if x >= 0:
            return 1 / (1 + math.exp(-x))
        else:
            exp_x = math.exp(x)
            return exp_x / (1 + exp_x)

    def derivative(self, x):
        s = self.forward(x)
        return s * (1 - s)


class ReLU(ActivationFn):
    def forward(self, x):
        return max(0, x)

    def derivative(self, x):
        return 1.0 if x > 0 else 0.0

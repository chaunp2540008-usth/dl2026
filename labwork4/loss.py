import math


class Loss:
    def forward(self, y_pred, y_true):
        raise NotImplementedError

    def derivative(self, z_logit, y_true):
        raise NotImplementedError


class CrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        epsilon = 1e-15
        p = max(min(y_pred[0], 1 - epsilon), epsilon)
        return -(y_true[0] * math.log(p) + (1 - y_true[0]) * math.log(1 - p))

    def derivative(self, z_logit, y_true):
        z = z_logit[0]
        s = 1 / (1 + math.exp(-z))
        return [s - y_true[0]]

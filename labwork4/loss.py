import math

class Loss:
    def forward(self, y_pred, y_true):
        raise NotImplementedError

    def derivative(self, y_pred, y_true):
        raise NotImplementedError
    

class CrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        epsilon = 1e-15
        y_pred = max(min(y_pred[0], 1 - epsilon), epsilon)
        return - (y_true[0] * math.log(y_pred) + (1 - y_true[0]) * math.log(1 - y_pred))

    def derivative(self, y_pred, y_true):
        epsilon = 1e-15
        y_pred = max(min(y_pred[0], 1 - epsilon), epsilon)
        return [(y_pred - y_true[0]) / (y_pred * (1 - y_pred))]
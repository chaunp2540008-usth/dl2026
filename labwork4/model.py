import math

from activation import Sigmoid

class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad, lr):
        for layer in reversed(self.layers):
            
            grad = layer.backward(grad, lr)

    def fit(self, X, y, epochs, lr, loss):
        for epoch in range(epochs):
            total_loss = 0.0

            for xi, yi in zip(X, y):
                outputs = self.forward(xi)

                total_loss += loss.forward(outputs, [yi])

                grad = loss.derivative(outputs, [yi])
                print("grad_loss:", grad)
                self.backward(grad, lr)

            if epoch % 1000 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

    def predict(self, X):
        preds = []

        for x in X:
            z = self.forward(x)[0]

            sigmoid = Sigmoid()

            y = sigmoid.forward(z)

            preds.append(1 if y > 0.5 else 0)

        return preds
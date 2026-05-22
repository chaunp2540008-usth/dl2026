from typing import Any


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
        output_layer = self.layers[-1]

        for epoch in range(epochs):
            total_loss = 0.0

            for xi, yi in zip[tuple](X, y):
                outputs = self.forward(xi)
                total_loss += loss.forward(outputs, [yi])

                grad = loss.derivative([output_layer.z[0]], [yi])
                self.backward(grad, lr)

            if epoch % 1000 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

    def predict(self, X):
        preds = []
        for x in X:
            p = self.forward(x)[0]
            preds.append(1 if p > 0.5 else 0)
        return preds

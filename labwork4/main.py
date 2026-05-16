from activation import Sigmoid
from layer import InputLayer, HiddenLayer, OutputLayer
from model import NeuralNetwork
from loss import CrossEntropy

X = [[0, 0], [0, 1], [1, 0], [1, 1]]

y = [0, 1, 1, 0]


model = NeuralNetwork()
model.add(InputLayer())
model.add(HiddenLayer(2, 2, Sigmoid()))
model.add(OutputLayer(2, 1))

loss = CrossEntropy()

model.fit(X, y, epochs=1000, lr=1e-5, loss=loss)

print("Predictions:", model.predict(X))

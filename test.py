import numpy as np
from intelligence.NeuralNetwork import NeuralNetwork

# Layers
from intelligence.Layers.DenseLayer import DenseLayer
from intelligence.Layers.OutputLayer import OutputLayer
from intelligence.Layers.DropoutLayer import DropoutLayer

# Activation Functions
from intelligence.Activations.Functions import *

# np.random.seed(0)


X = \
np.array([
	[5, 80, 0.75, 4, 0.1, 10],
	
])



y = np.array([
	[1116]
])


nn = NeuralNetwork(X, y, [
	DenseLayer(100, sigmoid),
	# DropoutLayer(0.3),
	OutputLayer(y, relu)
])


nn.train(6000)

print("\n\n")



print(nn.predict(X))

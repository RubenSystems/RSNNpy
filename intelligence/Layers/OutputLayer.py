import numpy as np
from intelligence.Layers.DenseLayer import DenseLayer
from intelligence.Activations.Derivative import *


class OutputLayer(DenseLayer):

	def __init__(self, y, activation): 
		super(OutputLayer, self).__init__(y.shape[1], activation)
		self.y = y

	# Back for out layr
	def bProp(self, previous, **kwargs):
		# Weights
		savedWeights = self.weights.copy()
		error = self.y - self.output

		self.errorVal = str(round((1 - np.mean(np.abs(error))) * 100, 0))
		delta = error * derivitave(self.activation, self.output)

		alter = np.dot(previous.T, delta)
		self.weights += alter

		# Biases
		for i in delta:
			self.bias += i

		return (delta, savedWeights)
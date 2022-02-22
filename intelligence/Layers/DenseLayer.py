import numpy as np
from intelligence.Activations.Derivative import *

"""

-RSNN Dense Layer implamentaion 

-Inputs
--n_neurons (the number of neurons on the layer)
--activation (the activation function for the layer)

-Notes 
--Only initalise, the Neural Network class will handle back/forward propigation 
--If you try to b/fprop it will throw an error as it must be "created" as well as initalised 
--Similar syntax to output layer except output layer has different 
"""


class DenseLayer: 

	def __init__(self, n_neurons, activation): 
		self.n_neurons = n_neurons 
		self.activation = activation

	def create(self, inputs):
		self.weights = 2*np.random.random((inputs.shape[1],self.n_neurons)) - 1
		self.bias = np.zeros((1, self.n_neurons))
		self.input = inputs

	def update(self, inputs):
		self.input = inputs

	def fProp(self):
		self.output = self.activation(np.dot(self.input, self.weights) + self.bias)

	def out(self, inputs):
		output = self.activation(np.dot(inputs, self.weights) + self.bias)
		return output

	def bProp(self, bDelta, bWeights, previous, **kwargs):
		if kwargs["drop"] is not None:
			val = kwargs["drop"]
		else:
			val = 1
		# Weights
		savedWeights = self.weights.copy()
		error = np.dot(bDelta, bWeights.T)
		delta = error * derivitave(self.activation, self.output)
		alter = np.dot(previous.T, delta)
		self.weights += alter

		# Biases
		for i in delta:
			self.bias += i


		return (delta, savedWeights)


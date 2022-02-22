import numpy as np
import math
from intelligence.Layers.DropoutLayer import DropoutLayer


"""
- RSNN Implamentation
-- Inputs requred: X (training data), y (labels), layers (The layers requried to make a Neural Network)

-Notes
--NN layers must end with an output layer. If it does not it will not output anything
--forward and backward Propigation are not to be called, they are called by the neural network during training 
--The NN will suggest the number of layers requred 


-Usage 
--Initalise
--Call train
--Wait for train
--Run predictions




"""


class NeuralNetwork:
	def __init__(self, X, y, layers):
		self.created = False
		self.X = X
		self.y = y
		self.layers = layers

		reccomendedLayers = math.ceil(len(X) / (2 * (layers[0].n_neurons + layers[::-1][0].n_neurons)))
		print(f"Layers reccomended: {reccomendedLayers}, n_layers: {len(layers) - 1}")


	def fProp(self):
		inputs = self.X
		for i in self.layers:

			if self.created == False: 
				i.create(inputs)
			else:
				i.update(inputs)

			i.fProp()
			inputs = i.output
		self.output = self.layers[::-1][0].output

		if self.created == False:
			self.created = True

	def predict(self, inputs):
		inDat = inputs

		for i in self.layers: 
			
			inDat = i.out(inDat)



		return inDat

	def bProp(self):
		outputs = [x.output for x in self.layers[::-1]]

		outputs.pop(0)
		outputs.append(self.X)


		delta, weights = self.layers[::-1][0].bProp(outputs[0])
		self.errorVal = self.layers[::-1][0].errorVal
		dropVal = None
		for i in range(1, len(self.layers[::-1])):
			layer = self.layers[::-1][i]
			if isinstance(layer, DropoutLayer):
				drop = layer.bProp()
			else:
				delta, weights = layer.bProp(delta, weights, outputs[i], drop = dropVal)

	def train(self, epochs):
		epoch = 0
		self.fProp()
		for i in range(epochs):
			self.bProp()
			self.fProp()
			
			if (i % 1000) == 0 :
				print(f"Accuriacy on epoch {epoch}: {self.errorVal}%")
				epoch += 1



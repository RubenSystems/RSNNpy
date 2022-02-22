import numpy as np

from intelligence.Layers.DenseLayer import DenseLayer


"""

-RSNN Dropout Layer implamentaion 

-Inputs
--n_neurons (the number of neurons on the layer)
--activation (the activation function for the layer)

-Notes 
--Only initalise, the Neural Network class will handle back/forward propigation 
--If you try to b/fprop it will throw an error as it must be "created" as well as initalised 
--Similar syntax to output layer except output layer has different 
"""

class DropoutLayer (DenseLayer): 

	def __init__(self, p):
		self.p = p

	def create(self, inputs):
		self.drop = np.random.binomial(1, self.p, size=inputs.shape[1])
		self.input = inputs

	def update(self, inputs):
		self.create(inputs)

	def fProp(self):
		self.output = self.input * self.drop

	def out(self, inputs):
		output = inputs * self.p
		return output

	def bProp(self):
		return self.drop
import numpy as np

# example data
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

# X = features  | y = targets/output variables
X, y = spiral_data(100, 3)

# inputs
X = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]

# Dense layer class
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons) # * 0.10 to get values between -1 and 1
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        # Call this each pass forward, using the raw inputs or inputs from previous layer
        self.output = np.dot(inputs, self.weights) + self.biases


# ReLU activation
class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs )

# inputs = num of features
# neurons = whatever you want
layer1 = Layer_Dense(4,5)

# input to next layer has to be num of neurons/outputs from last layer, neurons can be anything again
layer2 = Layer_Dense(5,2) 

# Feed data into nn
layer1.forward(X)
layer2.forward(layer1.output)
print(layer2.output)
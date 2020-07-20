import numpy as np

# example data
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

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

# Softmax activation
class Activation_Softmax:
    def forward(self, inputs):
        # Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        # Normalize for each sample/obs
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities

# X = features
# y = targets/output variables
# Create dataset
X, y = spiral_data(100, 3)

# Create dense layer with 2 input features and 3 output values
dense1 = Layer_Dense(2,3) # 2 inputs (each sample has 2 features), 3 outputs

# Create ReLU activation
activation1 = Activation_ReLU()

# Create a second layer
dense2 = Layer_Dense(3,3) # 3 inputs, 3 outputs

# Create softmax activation
activation2 = Activation_Softmax()

# Make a forward pass of our training data through this layer
dense1.forward(X)

# Forward pass through activation function
# Takes in output from previous layer
activation1.forward(dense1.output)

# Make a forward pass through the second layer
dense2.forward(activation1.output)

# Make a forward pass through the activation function
activation2.forward(dense2.output)

print(activation2.output[:5])


import numpy as np

# Network will consist of single hidden layer containing 3 neurons

# 3 samples (feature sets), 4 features
inputs = np.array([
    [1, 2, 3, 2.5],
    [2., 5., -1., 2],
    [-1.5, 2.7, 3.3, -0.8]
])

# Weights transposed so we can take dot product
weights = np.array([
    [0.2, 0.8, -0.5, 1],
    [0.5, -0.91, 0.26, -0.5],
    [-0.26, -0.27, 0.17, 0.87]
]).T 

# Biases
biases = np.array([[2], [3], [0.5]]).T

# Forward pass
layer_outputs = np.dot(inputs, weights) + biases
relu_outputs = np.maximum(0, layer_outputs)

# Backpropagation
# ReLU Activation
# Simulates derivate with respect to input values from next layer passed to current layer during backpropation
relu_dvalues = np.ones(relu_outputs.shape) 
relu_dvalues[layer_outputs <= 0] = 0
drelu = relu_dvalues

# Dense layer
dinputs = np.dot(drelu, weights.T) # dinputs - multiply by weights
dweights = np.dot(inputs.T, drelu) # dweights - multiply by inputs
dbiases = np.sum(drelu, axis=0, keepdims=True) # dbiases - sum values over samples

# Update parameters
weights += -0.001 * dweights
biases += -0.001 * dbiases

print(weights)
print(biases)
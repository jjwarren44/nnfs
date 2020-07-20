import numpy as np

class Layer_Dense:
    # Layer initialization
    def __init__(self, inputs, neurons):
        self.weight = 0.01 * np.random.randn(inputs, neurons)
        self.biases = np.zeros((1, neurons))

    # Forward pass
    def forward(self, inputs):
        self.inputs = inputs # Save inputs for later (backpropagation)
        self.output = np.dot(inputs, self.weights) + self.biases

    # Backward pass / Backpropogation
    def backward(self, dvalues):
        # Gradients on paramaters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        # Gradient on values
        self.dvalues = np.dot(dvalues, self.weights.T)

# ReLU activation
class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        # Since we need to modify original variable, save a copy of dvalues
        self.dvalues = dvalues.copy()

        # Zero gradient where input values were negative
        self.dvalues[self.inputs <= 0] = 0

# Softmax activation
class Activation_Softmax:
    def forward(self, inputs):
        # Remember input values for backpropagation
        self.inputs = inputs

        # Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        # Softmax outputs will be in range (0,1) and all add to 1
        # Can be intepreted as probabilities
        self.output = probabilities
    
    def backward(self, dvalues):
        self.dvalues = dvalues.copy()

# Cross-entropy loss
class Loss_CategoricalCrossentropy:
    def forward(self, y_pred, y_true):
        # Number of samples in a batch
        samples = y_pred.shape[0]

        # Probabilities for target values - only if categorical labels
        if len(y_true.shape) == 1:
            y_pred = y_pred[range(samples), y_true]

        # Losses
        negative_log_likelihoods = -np.log(y_pred)

        # Mask values - only for one-hot encoded labels
        if len(y_true.shape) == 2:
            negative_log_likelihoods *= y_true

        # Overall loss
        data_loss = np.sum(negative_log_likelihoods) / samples
        return data_loss
    
    def backward(self, dvalues, y_true):
        samples = dvalues.shape[0]

        # Copy so we can safely modify
        self.dvalues = dvalues.copy()

        self.dvalues[range(samples), y_true] -= 1
        self.dvalues = self.dvalues / samples


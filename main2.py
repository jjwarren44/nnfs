import numpy as np
import random

random.seed(0)
np.random.seed(0)

# THIS FORM OF BACKPROPAGATION IS CALLED STOCHASTIC GRADIENT DESCENT (SGD) #
# Subtracting a fraction of the gradient for each weight and bias paramater #
# Most optimizers are variants of SGD

# TAKEN FROM THE BOOK #
# Our sample dataset
def create_data(n, k):
    X = np.zeros((n*k, 2))  # data matrix (each row = single example)
    y = np.zeros(n*k, dtype='uint8')  # class labels
    for j in range(k):
        ix = range(n*j, n*(j+1))
        r = np.linspace(0.0, 1, n)  # radius
        t = np.linspace(j*4, (j+1)*4, n) + np.random.randn(n)*0.2  # theta
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = j
    return X, y

# Dense layer
class Layer_Dense:

    # Layer initialization
    def __init__(self, inputs, neurons):
        # Initialize weights and biases
        self.weights = 0.01 * np.random.randn(inputs, neurons)
        self.biases = np.zeros((1, neurons))

    # Forward pass
    def forward(self, inputs):
        # Remember input values
        self.inputs = inputs
        # Calculate output values from input ones, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases

    # Backward pass
    def backward(self, dvalues):
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # Gradient on values
        self.dvalues = np.dot(dvalues, self.weights.T)


# ReLU activation
class Activation_ReLU:

    # Forward pass
    def forward(self, inputs):
        # Remember input values
        self.inputs = inputs
        # Calculate output values from input ones
        self.output = np.maximum(0, inputs)

    # Backward pass
    def backward(self, dvalues):
        # Since we need to modify original variable, 
        # let's make a copy of values first
        self.dvalues = dvalues.copy()

        # Zero gradient where input values were negative 
        self.dvalues[self.inputs <= 0] = 0 


# Softmax activation
class Activation_Softmax:

    # Forward pass
    def forward(self, inputs):
        # Remember input values
        self.inputs = inputs

        # get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities

    # Backward pass
    def backward(self, dvalues):
        self.dvalues = dvalues.copy()


# Cross-entropy loss
class Loss_CategoricalCrossentropy:

    # Forward pass
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

    # Backward pass
    def backward(self, dvalues, y_true):

        samples = dvalues.shape[0]

        self.dvalues = dvalues.copy()  # Copy so we can safely modify
        self.dvalues[range(samples), y_true] -= 1
        self.dvalues = self.dvalues / samples

class Optimizer_SGD:
    # Initialize optimizer - set settings
    # Learning rate of 1 is the default for this optimizer
    def __init__(self, learning_rate=1.0, decay=0.1):
        self.learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
    
    # Call once before any parameter updates, apply learning rate decay
    def pre_update_params(self):
        if self.decay:
            self.learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    # Update parameters
    def update_params(self, layer):
        layer.weights += -self.learning_rate * layer.dweights
        layer.biases += -self.learning_rate * layer.dbiases

    # Call once after any parameter updates, update iterations
    def post_update_params(self):
        self.iterations += 1



# Create dataset
X, y = create_data(100, 3)

# Create dense layer with 2 input features and 64 output values
dense1 = Layer_Dense(2, 64)

# Create ReLU activation (to be used with dense layer)
activation1 = Activation_ReLU()

# Create second dense layer with 64 input features (as we take output of previous layer here)
# And 3 output values
dense2 = Layer_Dense(64, 3)

# Create softmax activation (to be used with dense layer)
activation2 = Activation_Softmax()

# Create loss function
loss_function = Loss_CategoricalCrossentropy()

# Create optimizer
optimizer = Optimizer_SGD(decay=5e-8)

# Train in loop
for epoch in range(10001):

    ### MAKE A FORWARD PASS ###

    # Make a forward pass of our training data thru this layer
    dense1.forward(X)

    # Make a forward pass thru activation function
    # it takes the output of first dense layer
    activation1.forward(dense1.output)

    # Make a forward pass thru second Dense layer
    # it takes outputs of activation function of first layer of inputs
    dense2.forward(activation1.output)

    # Make a forward pass thru activation function
    # it takes the ouput of second dense layer
    activation2.forward(dense2.output)


    ### CALCULATE LOSS AND ACCURACY ###

    # Calculate loss from output of activation2 (softmax activation)
    loss = loss_function.forward(activation2.output, y)


    # Calculate accuracy from output of activation2 and targets
    predictions = np.argmax(activation2.output, axis=1)
    accuracy = np.mean(predictions==y) # predictions==y returns a list of 0s and 1s (True & False), then take mean

    if not epoch % 100:
        print(f'epoch: {epoch}, acc: {accuracy:.3f}, loss: {loss:.3f}, lr: {optimizer.learning_rate}')


    ### BACKWARD PASS ###

    loss_function.backward(activation2.output, y)
    activation2.backward(loss_function.dvalues)
    dense2.backward(activation2.dvalues)
    activation1.backward(dense2.dvalues)
    dense1.backward(activation1.dvalues)


    ### USE OPTIMIZER ###
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()





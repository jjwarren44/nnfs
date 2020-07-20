import numpy as np

np.random.seed(0)

def create_data(points, classes):
    X = np.zeros((points*classes, 2))
    y = np.zeros(points*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0, 1, points)  # radius
        t = np.linspace(class_number*4, (class_number+1)*4, points) + np.random.randn(points)*0.05
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
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
        # Calculate output values from input ones, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases


# ReLU activation
class Activation_ReLU:

    # Forward pass
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


# Softmax activation
class Activation_Softmax:

    # Forward pass
    def forward(self, inputs):

        # get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities


# Cross-entropy loss
class Loss_CategoricalCrossentropy:

    def forward(self, y_pred, y_true):

        # Number of samples in a batch
        samples = len(y_pred)

        # Probabilities for target values
        y_pred = y_pred[range(samples), y_true]

        # Losses
        negative_log_likelihoods = -np.log(y_pred)

        # Average loss
        data_loss = np.sum(negative_log_likelihoods) / samples
        return data_loss

# Create dataset
X, y = create_data(100, 3)

# Create Dense layer with 2 input features and 3 output values
dense1 = Layer_Dense(2, 3)  # first dense layer, 2 inputs (each sample has 2 features), 3 outputs

# Create ReLU activation (to be used with Dense layer):
activation1 = Activation_ReLU()

# Create second Dense layer with 3 input features (as we take output of previous layer here) and 3 output values (output values)
dense2 = Layer_Dense(3, 3)  # second dense layer, 3 inputs, 3 outputs

# Create Softmax activation (to be used with Dense layer):
activation2 = Activation_Softmax()

# Create loss function
loss_function = Loss_CategoricalCrossentropy()

# Make a forward pass of our training data through this layer
dense1.forward(X)

# Make a forward pass through activation function - we take output of previous layer here
activation1.forward(dense1.output)

# Make a forward pass through second Dense layer - it takes outputs of activation function of first layer as inputs
dense2.forward(activation1.output)

# Make a forward pass through activation function - we take output of previous layer here
activation2.forward(dense2.output)

# Let's see output of few first samples:
print(activation2.output[:5])

# Calculate loss from output of activation2 (softmax activation)
loss = loss_function.forward(activation2.output, y)

# Let's print loss value
print('loss:', loss)

# Calculate accuracy from output of activation2 and targets
# take max output for each prediction and store in list
predictions = np.argmax(activation2.output, axis=1)

# take the mean of predictions vs actuals
accuracy = np.mean(predictions==y)

print('acc:', accuracy)


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
    def __init__(self, inputs, neurons, weight_regularizer_l1=0, weight_regularizer_l2=0, 
                    bias_regularizer_l1=0, bias_regularizer_l2=0):
        # Initialize weights and biases
        self.weights = 0.01 * np.random.randn(inputs, neurons)
        self.biases = np.zeros((1, neurons))

        # Set regularization strength
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

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

        # Gradients on regularization
        # L1 on weights
        if self.weight_regularizer_l1 > 0:
            dL1 = self.weights.copy()
            dL1[dL1 >= 0] = 1
            dL1[dL1 < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1

        # L2 on weights
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights
        
        # L1 on biases
        if self.bias_regularizer_l1 > 0:
            dL1 = self.biases.copy()
            dL1[dL1 >= 0] = 1
            dL1[dL1 < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1

        # L2 on biases
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases

        # Gradient on values
        self.dvalues = np.dot(dvalues, self.weights.T)


# Dropout layer
class Layer_Dropout:
    def __init__(self, rate):
        self.rate = 1 - rate

    def forward(self, values):
        # Save input values
        self.input = values
        
        self.binary_mask = np.random.binomial(1, self.rate, size=values.shape) / self.rate
        self.output = values * self.binary_mask

    def backward(self, dvalues):
        # Gradient on values
        self.dvalues = dvalues * self.binary_mask


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

# Sigmoid activation
class Activation_Sigmoid:
    def forward(self, inputs):
        # Save input and calculate/save output of the sigmoid function
        self.input = inputs
        self.output = 1 / (1 + np.exp(-inputs))

    def backward(self, dvalues):
        # Derivate - calculates from output of the sigmoid function
        self.dvalues = dvalues * (1 - self.output) * self.output

# General/common loss class
class Loss:
    
    # Regularization loss calculation
    def regularization_loss(self, layer):
        # 0 by default
        regularization_loss = 0

        # L1 regularization - weights
        if layer.weight_regularizer_l1 > 0: # only calc when factor greater than 0
            regularization_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))

        # L2 regularization - weights
        if layer.weight_regularizer_l2 > 0:
            regularization_loss += layer.weight_regularizer_l2 * np.sum(layer.weights**2)

        # L1 regularization - bias
        if layer.bias_regularizer_l1 > 0:
            regularization_loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))

        # L2 regularization - bias
        if layer.bias_regularizer_l2 > 0:
            regularization_loss += layer.bias_regularizer_l2 * np.sum(layer.biases**2)

        return regularization_loss

# Cross-entropy loss
class Loss_CategoricalCrossentropy(Loss):

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

class Loss_BinaryCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Calculate sample-wise loss
        sample_losses = -(y_true, np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))

        return sample_losses

    def backward(self, dvalues, y_true):
        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        clipped_dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)

        # Gradient on dvalues
        self.dvalues = -(y_true / clipped_dvalues - (1 - y_true) / (1 - clipped_dvalues))


class Optimizer_SGD:
    # Initialize optimizer - set settings
    # Learning rate of 1 is the default for this optimizer
    def __init__(self, learning_rate=1.0, decay=0.1, momentum=0):
        self.learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum
    
    # Call once before any parameter updates, apply learning rate decay
    def pre_update_params(self):
        if self.decay:
            self.learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    # Update parameters
    def update_params(self, layer):
        # If layer does not contain momentum arrays, create them filled with zeros
        if not hasattr(layer, 'weight_momentums'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)

        # If we use momentum
        if self.momentum:

            # Build weight updates with momentum
            # Take previous updates multipled by retain factor
            # and update with current gradients
            weight_updates = (
                (self.momentum * layer.weight_momentums) -
                (self.learning_rate * layer.dweights)
            )

            layer.weight_momentums = weight_updates

            # Bias updates
            bias_updates = (
                (self.momentum * layer.bias_momentums) -
                (self.learning_rate * layer.dbiases)
            )

            layer.bias_momentums = bias_updates

        else: # Vanilla SGD updates
            weight_updates = (-self.current_learning_rate *                      
                              layer.dweights)
            bias_updates = (-self.current_learning_rate * 
                            layer.dbiases)

        layer.weights += weight_updates
        layer.biases += bias_updates

    # Call once after any parameter updates, update iterations
    def post_update_params(self):
        self.iterations += 1


class Optimizer_Adagrad:
    # Initialize optimizer - set settings
    # Learning rate of 1 is the default for this optimizer
    def __init__(self, learning_rate=1.0, decay=0., momentum=0, epsilon=1e-7):
        self.learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum
        self.epsilon = epsilon
    
    # Call once before any parameter updates, apply learning rate decay
    def pre_update_params(self):
        if self.decay:
            self.learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    # Update parameters
    def update_params(self, layer):
        # If layer does not contain cache arrays, create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update cache with squared current gradients
        layer.weight_cache += layer.dweights**2
        layer.bias_cache += layer.dbiases**2    

        # Vanilla SGD parameter update + 
        # normalization with square rooted cache
        layer.weights += -self.learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)

    # Call once after any parameter updates, update iterations
    def post_update_params(self):
        self.iterations += 1

class Optimizer_RMSprop:

    # Initialize optimizer - set settings
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, rho=0.9):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho

    # Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    # Update parameters
    def update_params(self, layer):

        # If layer does not contain cache arrays,
        # create ones filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update cache with squared current gradients
        layer.weight_cache = self.rho * layer.weight_cache + (1 - self.rho) * layer.dweights**2
        layer.bias_cache = self.rho * layer.bias_cache + (1 - self.rho) * layer.dbiases**2

        # Vanilla SGD parameter update + normalization
        # with square rooted cache
        layer.weights += -self.learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)

    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1

# Adam optimizer - Adaptive moment
class Optimizer_Adam:

    # Initialize optimizer - set settings
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    # Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.current_learning_rate * (1. / (1. + self.decay * self.iterations))

    # Update parameters
    def update_params(self, layer):

        # If layer does not contain cache arrays,
        # create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update momentum  with current gradients
        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases
        
        # Get corrected momentum
        # self.iteration is 0 at first pass
        # and we need to start with 1 here
        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        
        # Update cache with squared current gradients
        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights**2
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases**2
        
        # Get corrected cachebias
        weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))

        # Vanilla SGD parameter update + normalization with square rooted cache
        layer.weights += -self.current_learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)

    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1




# Create dataset
X, y = create_data(100, 2)

# Reshape lables to be a list of lists
# Inner list contains one output (either 0 or 1)
# per each output neuron, 1 in this case
# Go from [0,0,0,0,0]
# to [[0],
#     [0],
#     [0],
#     [0],
#     [0]]
y = y.reshape(-1, 1)

# Create dense layer with 2 input features and 64 output values
dense1 = Layer_Dense(2, 64, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4)

# Create ReLU activation (to be used with dense layer)
activation1 = Activation_ReLU()

# Create dropout layer
#dropout1 = Layer_Dropout(0.1)

# Create second dense layer with 64 input features (as we take output of previous layer here)
# And 3 output values
dense2 = Layer_Dense(64, 1)

# Create softmax activation (to be used with dense layer)
activation2 = Activation_Sigmoid()

# Create loss function
loss_function = Loss_BinaryCrossentropy()

# Create optimizer
optimizer = Optimizer_Adam(decay=1e-8)

# Train in loop
for epoch in range(10001):

    ### MAKE A FORWARD PASS ###

    # Make a forward pass of our training data thru this layer
    dense1.forward(X)

    # Make a forward pass thru activation function
    # it takes the output of first dense layer
    activation1.forward(dense1.output)

    # Make a forward pass thru Dropout layer
    dropout1.forward(activation1.output)

    # Make a forward pass thru second Dense layer
    # it takes outputs of activation function of first layer of inputs
    dense2.forward(dropout1.output)

    # Make a forward pass thru activation function
    # it takes the ouput of second dense layer
    activation2.forward(dense2.output)


    ### CALCULATE LOSS AND ACCURACY ###

    # Calculate loss from output of activation2 (softmax activation)
    data_loss = loss_function.forward(activation2.output, y)

    # Calculate regularization penalty
    regularization_loss = loss_function.regularization_loss(dense1) + loss_function.regularization_loss(dense2)

    # Calculate overall loss
    loss = data_loss + regularization_loss

    # Calculate accuracy from output of activation2 and targets
    predictions = np.argmax(activation2.output, axis=1)
    accuracy = np.mean(predictions==y) # predictions==y returns a list of 0s and 1s (True & False), then take mean

    if not epoch % 100:
        print('epoch:', epoch, 'acc:', f'{np.mean(predictions==y):.3f}', 'loss:', f'{loss:.3f}', '(data_loss:', f'{data_loss:.3f}', 'reg_loss:', f'{regularization_loss:.3f})', 'lr:', optimizer.learning_rate)


    ### BACKWARD PASS ###

    loss_function.backward(activation2.output, y)
    activation2.backward(loss_function.dvalues)
    dense2.backward(activation2.dvalues)
    dropout1.backward(dense2.dvalues)
    activation1.backward(dropout1.dvalues)
    dense1.backward(activation1.dvalues)


    ### USE OPTIMIZER ###
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()



# Validate model #

# Create test dataset
X_test, y_test = create_data(100, 3)

# Make a forward pass of our test data thru this layer
dense1.forward(X_test)

# Make a forward pass thru activation function - we take output of previous layer here
activation1.forward(dense1.output)

# Make a forward pass thru second Dense layer - it takes outputs of activation function of first layer as inputs
dense2.forward(activation1.output)

# Make a forward pass thru activation function - we take output of previous layer here
activation2.forward(dense2.output)

# Calculate loss from output of activation2 so softmax activation
loss = loss_function.forward(activation2.output, y_test)

# Calculate accuracy from output of activation2 and targets
predictions = np.argmax(activation2.output, axis=1)  # calculate values along first axis
accuracy = np.mean(predictions==y_test)

print(f'validation, acc: {accuracy:.3f}, loss: {loss:.3f}')






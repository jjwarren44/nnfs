import math
import numpy as np

layer_outputs = [4.8, 1.21, 2.385]

class Activation_Softmax:
    def forward(self, inputs):
        # Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        # Normalize for each sample/obs
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities



'''
BOOK EXAMPLES OF IMPLEMENTATION
'''

'''
WITHOUT NUMPY
'''

# First step is to exponentiate outputs with Euler's/natural number `e`
# Exponentiation is mainly used for calculating a more meaningful loss
exp_values = []
for output in layer_outputs:
    exp_values.append(math.e ** output)

print('Exponentiated values:')
print(exp_values)

# Now normalize values by summing all values and divinding individual values by the sum
norm_base = sum(exp_values)
norm_values = []
for value in exp_values:
    norm_values.append(value / norm_base)

print('Normalized exponentiated values:')
print(norm_values) 

print('sum of normalized values:') # == 1
print(sum(norm_values))


'''
WITH NUMPY
'''
# Exponentiate values
exp_values = np.exp(layer_outputs)
print('Exponentiated values:')
print(exp_values)

# Normalize values
norm_values = exp_values / np.sum(exp_values)
print('Normalized exponentiated values:')
print(norm_values)

print('sum of normalized values:') # == 1
print(np.sum(norm_values))


'''
MAKING SURE NEURONS DON't DIE OR EXPLODE
'''
exp_values = np.exp(layer_outputs)
# axis 0 = columns, axis 1 = rows
# keepdims=True outputs a n by 1 vector (row for each sum) instead of a list
probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True) 



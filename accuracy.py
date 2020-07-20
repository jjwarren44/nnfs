import numpy as np

softmax_outputs = [[0.7, 0.2, 0.1],  # probabilities for 3 samples
                   [0.5, 0.1, 0.4],  # values swapped here
                   [0.02, 0.9, 0.08]]
targets = [0, 1, 1]  # target (ground truth) labels for 3 samples

predictions = np.argmax(softmax_outputs, axis=1)  # calculate values along second axis (axis of index 1)
accuracy = np.mean(predictions==targets) # True evaluates to 1; False to 0

print('acc:', accuracy)
import math
import numpy as np

class Loss_CategoricalCrossentropy:
    def forward(self, y_pred, y_true):
        # num of samples in a batch
        n_samples = len(y_pred)

        # Probabilities for target values
        y_pred = y_pred[range(n_samples), y_true]
        print('y_pred:', y_pred)

        # Losses
        negative_log_likelihoods = -np.log(y_pred)

        # Overall loss (agg)
        data_loss = np.mean(negative_log_likelihoods)
        return data_loss

# EXAMPLE #
# probabilities for 3 samples
softmax_outputs = np.array([[0.7, 0.1, 0.2],
                           [0.1, 0.5, 0.4],
                           [0.02, 0.9, 0.08]])

# Assume we have 3 targets -> [human, dog, cat]
# our target ouputs for these example model outputs is dog, cat, cat
class_targets = [0, 1, 1]

loss_function = Loss_CategoricalCrossentropy()
loss = loss_function.forward(softmax_outputs, class_targets)
print(loss)



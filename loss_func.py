

# Since this is a classification problem, we take argmax() of outputs
# and use that index as the answer outputted by the model
# but think about these two outputs:
    # 1. [0.22, 0.6, 0.18]
    # 2. [0.32, 0.36, 0.32]
# both return index 1 (2nd element), but the 2nd output isnt as confident in that answer
# we want to reward the model more if it is more confident about the correct answer

# Cross-entropy is used as a the loss function
# it compares 'ground-truth' probability (p or 'targets) and some predicted distribution (q or 'predictions')
# this formula is: -log(correct_class_confidence)
import math
import numpy as np

softmax_output = [0.7, 0.1, 0.2] # example output from a layer in neural network

# target output is considered `one-hot` since one value is `hot (on)` and the rest are not (off)
target_output = [1, 0, 0] # ground truth

loss = -(math.log(softmax_output[0])*target_output[0] +
         math.log(softmax_output[1])*target_output[1] +
         math.log(softmax_output[2])*target_output[2])

# Since only one index isn't 0, we can rewrite loss as:
loss = -(math.log(softmax_output[0])*target_output[0])

print(loss)


'''
MAKING DYNAMIC WITH MORE SAMPELS
'''
# probabilities for 3 samples
softmax_outputs = np.array([[0.7, 0.1, 0.2],
                           [0.1, 0.5, 0.4],
                           [0.02, 0.9, 0.08]])

# Assume we have 3 targets -> [human, dog, cat]
# our target ouputs for these example model outputs is dog, cat, cat
class_targets = [0, 1, 1]

# Print loss
print('all loss vals:', -np.log(softmax_outputs[range(len(softmax_outputs)), class_targets]))

# We want the average loss for the batch (current outputs)
neg_log = -np.log(softmax_outputs[range(len(softmax_outputs)), class_targets])
avg_loss = np.mean(neg_log)
print('avg loss:', avg_loss)

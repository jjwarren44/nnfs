import cv2
import matplotlib.pyplot as plt
import numpy as np

# Read in image as grayscale
image_data = cv2.imread('fashion_mnist_images/new/tshirt.png', cv2.IMREAD_GRAYSCALE)

# Resize image to 28x28
image_data = cv2.resize(image_data, (28,28))

# Plot image
plt.imshow(image_data, cmap='gray')
plt.show()

# Reshape the data to a 1x784 array
# and squeeze values between -1 and 1
image_data = (image_data.reshape(1, -1).astype(np.float32) - 127.5) / 127.5
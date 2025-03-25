from padding import convolution
import numpy as np
import cv2
import matplotlib.pyplot as plt

image = cv2.imread('gatitos.jpg', cv2.IMREAD_GRAYSCALE)

kernel = np.array([[-1, -1, -1], 
                   [-1,  8, -1], 
                   [-1, -1, -1]])

image = convolution(image, kernel)

plt.imshow(image, cmap='gray')
plt.axis('off')
plt.show()
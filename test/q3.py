import cv2
import numpy as np
image = cv2.imread('test2.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
print(image.shape)
mask = np.all(image == [0,0,0,255], axis=2)
image[mask] = [0,0,0,0]
cv2.imwrite("test3.png", image)
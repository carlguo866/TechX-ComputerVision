import numpy as np
from PIL import Image
img_src = Image.open('images/lena.jpg').convert("L")
img = np.array(img_src).astype(np.float32)
h,w = img.shape[:2]
x_edge= abs(img[:,1:]-img[:,:w-1])
y_edge= abs(img[1:,:]-img[:h-1,:])
x_edge= x_edge[:h-1,:]
y_edge= y_edge[:,:w-1]
# print(x_edge.shape)
# print(y_edge.shape)
gradient= np.sqrt((x_edge**2)+(y_edge**2))
print(gradient.max(),gradient.min(),gradient.mean())
t = 20
filtered = gradient[::]
filtered[filtered<t] =0;

import matplotlib.pyplot as plt
plt.subplot(2,2,1)
plt.imshow(np.uint8(x_edge),cmap="gray")
plt.subplot(2,2,2)
plt.imshow(np.uint8(y_edge),cmap="gray")
# gradient[gradient>=t] = 255;
plt.subplot(2,2,3)
plt.imshow(np.uint8(filtered),cmap="gray")
plt.show()
#
# import cv2
# img_edge = cv2.Canny(np.uint8(img),100,200)
# plt.imshow(img_edge,cmap="gray")
# plt.show()

import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
# def gaussian_filter():
#
#
# def low_pass_filter(image, filter):
#     shifted = np.fft.fftshift(np.fft.fft2(image))
#     blurred = np.fft.fftshift(np.fft.fft2(filter))
#     result = shifted * blurred
#     return np.fft.ifft2(np.fft.ifftshift(result))
def normal_filter(mar, ein):
    result = cv2.GaussianBlur(mar, (5, 5), cv2.BORDER_DEFAULT)
    result2 = img2 - cv2.GaussianBlur(ein, (5, 5), cv2.BORDER_DEFAULT)
    return result+result2

def gaussian_kernel(size, size_y=None):
    size = int(size)
    if not size_y:
        size_y = size
    else:
        size_y = int(size_y)
    x, y = np.mgrid[-size:size+1, -size_y:size_y+1]
    g = np.exp(-(x**2/float(size)+y**2/float(size_y)))
    return g / g.sum()

img = (Image.open("images/einstein.png")).convert("L")/255.0
img2 = (cv2.imread("images/marilyn.png")).convert("L")/255.0
print(img.shape)
gaussian = gaussian_kernel(img.shape[0]/2,img.shape[1]/2)
gaussian_f = np.fft.fftshift(np.fft.fft2(gaussian))
print(gaussian_f.shape)
f = np.fft.fftshift(np.fft.fft2(img))
print(f.shape)
f *= gaussian_f
result = np.fft.ifft(np.fft.ifftshift(f)).real
plt.imshow(result, cmap="gray")
plt.show()



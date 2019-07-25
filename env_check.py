print("Testing")
try:
    import numpy
except ImportError:
    print("Numpy Error")
try:
    import skimage
except ImportError:
    print("Skimage Error")
try:
    import matplotlib
except ImportError:
    print("matplotlib Error")
try:
    import jupyter
except ImportError:
    print("jupyter error")
try:
    import PIL
except ImportError:
    print("Pillow Error")
try:
    import cv2
except ImportError:
    print("OpenCV Error")

print("Test completed")

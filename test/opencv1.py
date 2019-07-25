import cv2
import numpy as np
import os, sys
from PIL import Image
# img = cv2.imread("messi5.jpg")
# # print(img)
# px = img[100,100]
# blue = img[100,100,0]
# print(px)
# print(blue)
# img[100,100] = [0,0,0]
#
# print(img.item(10,10,2))
# img.itemset((10,10,1),255)
# print(img.item(10,10,2))
#
# print(img.shape)
#
# ball = img[280:340, 330:390]
# img[273:333, 100:160] = ball
# img[:,:,2] = 0
# cv2.imshow('image',img)
# cv2.waitKey(0)
# cv2.imwrite("image.jpg",img)
# cv2.destroyAllWindows()
# # print("hello world")
# flags = [i for i in dir(cv2) if i.startswith('COLOR_')]
# print(flags)
# from PIL import Image
im = Image.open("messi5.jpg")
print(im.format, im.size, im.mode)
# box = (100, 100, 400, 400)
# region = im.crop(box)
# region = region.transpose(Image.ROTATE_180)

def roll(image, delta):
    """Roll an image sideways."""
    xsize, ysize = image.size

    delta = delta % xsize
    if delta == 0: return image

    part1 = image.crop((0, 0, delta, ysize))
    part2 = image.crop((delta, 0, xsize, ysize))
    image.paste(part1, (xsize-delta, 0, xsize, ysize))
    image.paste(part2, (0, 0, xsize-delta, ysize))

    return image

# im.paste(region, box)
im = roll(im,360);
im.show()





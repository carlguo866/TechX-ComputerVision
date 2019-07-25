from PIL import Image
import numpy
img_src = Image.open('images/mars.png')
image_mat = numpy.array(img_src)
result = Image.fromarray(image_mat)
result.save("images/mars1.png")

diagonalPic= Image.new('L', (128, 128), color = "white")
image_mat = numpy.array(diagonalPic)
for i in range(128):
    image_mat[i,i] = 0
result = Image.fromarray(image_mat)
result.save("images/p1.pgm")

img_src = Image.open('images/mars.png')
image_mat = numpy.array(img_src)
for i in range(img_src.size[0]):
    for j in range(img_src.size[1]):
        image_mat[i,j] = 255* (image_mat[i,j]/255)** (1/2.2)
result = Image.fromarray(image_mat)
result.save("images/mars2.png")



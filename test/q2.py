import cv2
import numpy as np
img = cv2.imread("Q2.png",1)
b, g, r = cv2.split(img)
img = cv2.merge((b, g, r))
cv2.imshow("image",img)
cv2.imwrite("test2.png",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
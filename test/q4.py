import cv2
import numpy as np
img = cv2.imread("test3.png",-1)
img = cv2.resize(img,(img.shape[1]*2, img.shape[0]*2))
stack = np.hstack((img,img))
cv2.imwrite("test4.png",stack)
cv2.waitKey(0)
cv2.destroyAllWindows()

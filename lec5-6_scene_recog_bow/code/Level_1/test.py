import descriptors
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
img = cv2.imread("../../data/test/Bedroom/image_0001.jpg")
# plt.imshow(img)
# plt.show()
img_disc, y= descriptors.orb(img,[],1,1)
descriptors.cluster_features(img_disc,MiniBatchKMeans())

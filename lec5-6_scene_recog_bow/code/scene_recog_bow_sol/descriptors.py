import cv2
import numpy as np
import matplotlib.pyplot as plt

def orb(img,img_descs,y,class_number):
    """
    Calculate the ORB descriptors for an image.
    Args:
        img (greyscale image): The image that will be used.
        img_descs: (list of arrays) accumulator list for ORB descriptors to be appended to.
        y: (int list) accumulator list for labels to be appended to.
        class_number: (int) label numbering.
    Returns:
        img_descs: (list of floats arrays) The descriptors found in the image.
        y: (original) image labels
    """
    orb = cv2.ORB_create(5000)
    kp, des = orb.detectAndCompute(img, None)
    if des is not None:
        img_descs.append(des)
        y.append(class_number)
        # uncomment to draw only keypoints location, not size and orientation
        # img2 = cv2.drawKeypoints(img,kp,color=(0,255,0), outImage=np.array([]), flags=0)
        # plt.imshow(img2),plt.show()
    else:
        print('orb descriptor not found (for this image)... but it is ok!')
    
    return img_descs,y

def cluster_features(img_descs, cluster_model):
    """
    Cluster the training features using the cluster_model
    and convert each set of descriptors in img_descs
    to a Visual Bag of Words histogram.
    Args:
        img_descs : list of lists of ORB descriptors
        cluster_model : clustering model (eg KMeans from scikit-learn)
            The model used to cluster the ORB features 
            (cluster_model can be None if istrain=True)
    Returns:
        X: (nested numpy array) a bag of words (histogram) representation 
            of the group of iamges.
        cluster_model: kmeans model that has been fit to the training set
    """
    n_clusters = cluster_model.n_clusters

    # train k-means or other cluster model on those descriptors selected above
    cluster_model.fit(np.concatenate(img_descs,axis=0))
    print('done clustering.')

    img_bow_hist = img_to_vect(img_descs, cluster_model)

    return img_bow_hist, cluster_model

def img_to_vect(img_descs, cluster_model):
    """
    Given the descriptors of a group of images and a trained clustering model (eg KMeans),
    generates a feature vector representing that group of images.
    Useful for processing new images for a classifier prediction.
    Args:
        img_descs: (nested numpy array) The descriptors found in the group of images.
        cluster_model: trained clustering model.
    Returns:
        img_bow_hist: (nested numpy array) a bag of words (histogram) representation of the 
            group of iamges.
    """

    print('using clustering model to generate BoW histograms for each image.')
    # compute set of cluster-reduced words for each image
    clustered_desc = [cluster_model.predict(raw_words) for raw_words in img_descs]
    # make a histogram of clustered word counts for each image. These are the final features.
    img_bow_hist = np.array([np.histogram(clustered_words, bins=cluster_model.n_clusters)[0] 
        for clustered_words in clustered_desc])
    print('done generating BoW histograms.')

    return img_bow_hist
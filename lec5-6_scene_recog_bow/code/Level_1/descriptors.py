import cv2
import numpy as np
import matplotlib.pyplot as plt


MAX_FEATURES=300

def orb(img_path,img_descs,y,class_number):
    """
    Calculate the ORB descriptors for an image.
    Args:
        img (BGR matrix): The image that will be used.
    Returns:
        img_descs: (list of floats array) The descriptors found in the image.
        y: (original) image labels
    """
    
    ##################################################################################
    #                             BEGINNING OF YOUR CODE                             #
    ##################################################################################
    print(img_path)

    img = cv2.imread(img_path)
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints, img_desc = orb.detectAndCompute(img, None)
   # img_descs = np.vstack([this_img_descs,img_descs])
    ##################################################################################
    #                                END OF YOUR CODE                                #
    ##################################################################################
    
    # HINT:

    # you can use the below code to visualize ORB descriptors
    # (draw only keypoints location (not size and orientation))
    
    # img2 = cv2.drawKeypoints(img,keypoints,color=(0,255,0), outImage=np.array([]), flags=0)
    # plt.imshow(img2),plt.show()

    return img_desc,y

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
    # this is the number of clusters you have (i.e. num of visual words you have )
    n_clusters = cluster_model.n_clusters
    ##################################################################################
    #                             BEGINNING OF YOUR CODE                             #
    ##################################################################################
    cluster_model.fit(img_descs)
    print(img_descs)
    words = cluster_model.predict(img_descs)
    # print(img_descs.shape)
    # print(len(words))
    # print(words)
    X, bin_edges = np.histogram(words, bins=cluster_model.n_clusters)
    ##################################################################################
    #                                END OF YOUR CODE                                #
    ##################################################################################

    # HINT: use the below procedure

    # 1. train k-means or other cluster model on those descriptors selected above
    # 2. compute set of cluster-reduced words for each image
    # 3. finally make a histogram of clustered word counts for each image. T
    #   hese are the final features.

    return X, cluster_model

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

    ##################################################################################
    #                             BEGINNING OF YOUR CODE                             #
    ##################################################################################

    words = cluster_model.predict(img_descs)
    img_bow_hist, bin_edges = np.histogram(words, bins=cluster_model.n_clusters)
    
    
    ##################################################################################
    #                                END OF YOUR CODE                                #
    ##################################################################################

    # HINT: use the below procedure

    # 1. compute set of cluster-reduced words for each image 
    # 2. make a histogram of clustered word counts for each image. These are the final features.

    return img_bow_hist
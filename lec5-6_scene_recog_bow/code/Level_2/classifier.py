import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import scipy
import pickle
from sklearn.svm import SVC
# from sklearn.model_selection import RandomizedSearchCV
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import RandomForestClassifier

import descriptors
import utils


class Classifier(object):
    def __init__(self, dataset):
        self.dataset = dataset
        self.istrain = True
        self.LOAD_CACHE = False
        self.plot_feat_hist = False

    def train(self, k=200, verbose=False):
        """
        Train the classifier (SVM or RandomForestClassifier) on the dataset
        Args:
            k: number of k-means clusters
            verbose: print more log info to stdout
        Returns:
            svm: trained classifier
            cluster_model: trained k-means clustering model
        """
        self.istrain = True

        ##################################################################################
        #                             BEGINNING OF YOUR CODE                             #
        ##################################################################################

        ## extract ORB descriptors and train k-means clustering to form Bag of Words
        

        ## SVM 
        # SVM instantiation:
        
        
        # SVM fitting:

        
        ##################################################################################
        #                                END OF YOUR CODE                                #
        ##################################################################################
        
        # NOTE (not HINT): just for those over-achievers: the below code does some 
        #   hyperparam search that you can play around with

        # param_dist = {'C': scipy.stats.expon(scale=100),
        #             'gamma': scipy.stats.expon(scale=.1),
        #             'kernel': ['rbf']}

        # gs = RandomizedSearchCV(estimator=svm,
        #                 param_distributions=param_dist,
        #                 scoring='accuracy',
        #                 verbose=1
        #                 )
        # print('searching for optimal svm...')
        # gs = gs.fit(x,y)
        
        # print(f'Best Training Score = {gs.best_score_:.3f} with parameters {gs.best_params_}')
        # svm = gs.best_estimator_

        return svm, cluster_model

    def test(self, svm, cluster_model, k=200, verbose=False):
        """
        Test the classifier (SVM or RandomForestClassifier) on the test dataset
        Args:
            svm: trained classifier
            cluster_model: trained k-means clustering model
            k: number of k-means clusters
        Returns:
            result: the classifier's predicted scenes of the test images
            y: ground truth/targets of test images
        """
        self.istrain = False

        ##################################################################################
        #                             BEGINNING OF YOUR CODE                             #
        ##################################################################################

        ## extract ORB descriptors and get Bag of Words from k-means clustering
        
        
        # plot some feat histograms
        if self.plot_feat_hist:
            plot_num = 5
            for i,hist in enumerate(x[:plot_num]):
                hist = np.reshape(np.array(hist), -1)
                plt.figure(i)
                plt.bar(np.arange(cluster_model.n_clusters), hist)

        ## SVM 
        # SVM prediction:
        
        
        # SVM evaluation:

        
        ##################################################################################
        #                                END OF YOUR CODE                                #
        ##################################################################################

        # HINT: think about how to use pickle to cache models and BoG representations to 
        #   save you a ton of waiting while debugging

        # picle can be used in the below way:

        # with open('vocab.pkl', 'rb') as handle:
        #     cluster_model = pickle.load(handle)
        # with open('vocab.pkl', 'wb') as handle:
        #     pickle.dump(cluster_model, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return result,y

    def compute_descriptors(self, img_set, cluster_model, k=200):
        """
        Given a set of images and a trained clustering model, compute their 
            Bag of Words representations (remember it's just a histogram of words)
        Args:
            img_set: (dict) with keys being scene classes and values paths to images
            cluster_model: k-means clustering model (if self.istrain, this can be None)
            k: number of k-means clusters
        Returns:
            X: (nested numpy array) Bag of Words representations
            y: (numpy array) ground truth/targets of test images
            cluster_model: trained k-means clustering model
        """

        ##################################################################################
        #                             BEGINNING OF YOUR CODE                             #
        ##################################################################################
        
        
        
        
        ##################################################################################
        #                                END OF YOUR CODE                                #
        ##################################################################################

        # HINT: use the below procedure

        # 1. ORB descriptors
        # 2. k-means cluster the descriptors to form the Bag of Words representation

        # More Hint: (2.) can be achieved using descriptors.cluster_features (you write it!)
        #   during training phase (if self.istrain=True) and descriptors.img_to_vect 
        #   (you write it too!) during testing phase (if self.istrain=False)

        return X, y, cluster_model
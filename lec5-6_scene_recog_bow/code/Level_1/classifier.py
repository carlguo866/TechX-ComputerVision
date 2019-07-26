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

##################################################################################
#              IMPORTANT NOTE (will save you tons of time waiting)               #
##################################################################################
# If you are debugging, please first run the program with self.LOAD_CACHE=False  #
# This will compute ORB descriptors and k-means clustering Bag of Words and      #
# store them in the *.pkl files in '.' After the first run, please set           #
# self.LOAD_CACHE=True. This will save you tons of time debugging (since k-means)#
# takes a long time and with cache the program will directly load pre-computed   #
# descriptors and BoG representations from the .pkl files. However, if you ever  #
# changed the k or vocab_size in main.py, please re-run to update cache.         #
#################################################################################
# if you set self.plot_feat_hist true, you will see some visualization of BoG    #
# histogram in the plt window popping up.                                        #
#################################################################################

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

        # extract ORB descriptors and train k-means clustering to form Bag of Words
        if self.LOAD_CACHE:
            print('loading train vocab, training feats and labels from cache...')
            with open('vocab.pkl', 'rb') as handle:
                cluster_model = pickle.load(handle)
            with open('train_feats_and_labels.pkl', 'rb') as handle:
                (x,y) = pickle.load(handle)  
            print('successfully loaded train vocab and training feats and labels.')
        else:
            print('calculating global descriptors for the training set...')
            start = time.time()
            x,y,cluster_model = self.compute_descriptors(
                img_set=self.dataset.train, cluster_model=None, k=k
            )
            end = time.time()
            if verbose: print('elapsed {} for clustering'.format(utils.humanize_time(end-start)))

            # save to pickle for cache
            with open('vocab.pkl', 'wb') as handle:
                pickle.dump(cluster_model, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open('train_feats_and_labels.pkl', 'wb') as handle:
                pickle.dump((x,y), handle, protocol=pickle.HIGHEST_PROTOCOL)

        # SVM classification
        print('start svm')

        ##################################################################################
        #                             BEGINNING OF YOUR CODE                             #
        ##################################################################################
        
        # SVM instantiation:

        svm = SVC(kernel="poly")
        svm.fit(x,y)
        print(svm.score(x,y))
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

        # extract ORB descriptors and get Bag of Words from k-means clustering
        if self.LOAD_CACHE:
            print('loading test vocab, training feats and labels from cache...')
            with open('vocab.pkl', 'rb') as handle:
                cluster_model = pickle.load(handle)
            with open('test_feats_and_labels.pkl', 'rb') as handle:
                (x,y) = pickle.load(handle)  
            print('successfully loaded test vocab and training feats and labels.')

        else:
            print('calculating global descriptors for the test set...')
            start = time.time()
            x,y,cluster_model = self.compute_descriptors(
                img_set=self.dataset.test, cluster_model=cluster_model, k=k
            )
            end = time.time()
            if verbose: print('elapsed {} for clustering'.format(utils.humanize_time(end-start)))
            
            # save to pickle for cache
            with open('test_feats_and_labels.pkl', 'wb') as handle:
                pickle.dump((x,y), handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        # plot some feat histograms
        if self.plot_feat_hist:
            plot_num = 5
            for i,hist in enumerate(x[:plot_num]):
                hist = np.reshape(np.array(hist), -1)
                plt.figure(i)
                plt.bar(np.arange(cluster_model.n_clusters), hist)

        # SVM classification
        ##################################################################################
        #                             BEGINNING OF YOUR CODE                             #
        ##################################################################################
        
        # SVM prediction:
        predicts = svm.predict(x)
        print(svm.score(x,y))
        # SVM evaluation:
        #result = svm.decision_function(predicts)
        
        ##################################################################################
        #                                END OF YOUR CODE                                #
        ##################################################################################

        return predicts,y

    def compute_descriptors(self, img_set, cluster_model, k=200):
        """
        Given a set of images and a trained clustering model, compute their 
            Bag of Words representations (remember it's just a histogram of words)
        Args:
            img_set: (dict) with keys being scene classes and values paths to images
            cluster_model: k-means clutering model (if self.istrain, this can be None)
            k: number of k-means clusters
        Returns:
            X: (nested numpy array) Bag of Words representations
            y: (numpy array) ground truth/targets of test images
            cluster_model: trained k-means clustering model
        """

        ##################################################################################
        #                             BEGINNING OF YOUR CODE                             #
        ##################################################################################
        #print(img_set)
        print("K" , k)
        ylist = np.float32([]).reshape(0, 1)
        if cluster_model is None:
            cluster_model = MiniBatchKMeans(k)
        predicted_all = np.float32([]).reshape(0, k)
        for i,(class_name,img_paths) in enumerate(img_set.items()):
            print(class_name)
            #print(img_paths)
            imgs = descriptors.path_to_img(img_paths)
            descriptors_2d = []
            descriptors_3d = []
            for img in imgs:
                img_desc,_ = descriptors.orb(img,[],[],class_name)
                if img_desc is not None:
                    for descriptor in img_desc:
                        descriptors_2d.append(descriptor)
                    descriptors_3d.append(img_desc)
                    ylist = np.vstack((ylist,i))
            if self.istrain:
                cluster_model.fit(descriptors_2d)
                centers = cluster_model.cluster_centers_
                print("the center has shape of ", centers.shape)
            for img_desc in descriptors_3d:
                if img_desc is not None:
                    predict_img = descriptors.img_to_vect(img_desc,cluster_model)
                    predicted_all = np.vstack((predicted_all,predict_img))



            # print(class_name)
            # kmeans_fit = []
            # kmeans_fit2 = []
            # for img_path in img_paths:
            #     img_desc,y = descriptors.orb(img_path,[],[],class_name)
            #     #print(img_desc_array)
            #     #print(type(img_desc))
            #     for descriptor in img_desc:
            #         kmeans_fit.append(descriptor)
            #     kmeans_fit2.append(img_desc)
            # if self.istrain:
            #     descriptors.kmeans_fit_function(kmeans_fit,cluster_model)
            # for descriptor in kmeans_fit2:
            #     print(descriptor)
            #     bow,cluster_model = descriptors.img_to_vect(descriptor,cluster_model)
            #     ylist.append(i)
            #     X.append(bow)

        ##################################################################################
        #                                END OF YOUR CODE                                #
        ##################################################################################

        # HINT: use the below procedure

        # 1. ORB descriptors
        # 2. k-means cluster the descriptors to form the Bag of Words representation

        # More Hint: (2.) can be achieved using descriptors.cluster_features (you write it!)
        # during training phase (if self.istrain=True) and descriptors.img_to_vect 
        # (you write it too!) during testing phase (if self.istrain=False)
        print("X.shape",predicted_all.shape)
        print("Y.shape",ylist.shape)
        print(predicted_all)
        print(ylist)
        return predicted_all, ylist, cluster_model


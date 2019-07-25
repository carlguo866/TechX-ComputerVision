from pathlib import Path
import numpy as np

from dataset import Dataset
from classifier import Classifier
import utils

def main():
    # dataset loading
    # NOTE: default dataset path '../../data' - change if needed
    data_path = Path.cwd().parent.parent / 'data'
    data = Dataset(data_path)
    data.load_dataset()

    # classifier training and testing
    classifier = Classifier(data)
    vocab_size = 400 # up to you to decide the tradeoff between acc and speed
    svm, cluster_model = classifier.train(k=vocab_size, verbose=True)
    result, labels = classifier.test(svm, cluster_model, k=vocab_size, verbose=True)

    # compute the confusion matrix

    ##################################################################################
    #                             BEGINNING OF YOUR CODE                             #
    ##################################################################################
    






    ##################################################################################
    #                                END OF YOUR CODE                                #
    ##################################################################################

    print("Confusion Matrix =\n{0}".format(confusion_matrix))
    utils.show_conf_mat(confusion_matrix, id2name=data.class_id2name)

if __name__ == '__main__':
    main()
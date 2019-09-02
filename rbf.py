import random as rnd
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import math


from TinyMNIST_loader import * #from dataloader import select_features
import numpy as np
import os
from scipy.misc import imread
'''
val_num = 1000
train_num = 49000
test_num = 10000

train_data, train_labels, test_data, test_labels,\
    class_names, n_train, n_test, n_class, n_features = select_features()
'''
X_train = train_data
y_train = train_labels
X_test = test_data
y_test = test_labels
# Subsample the data
#df_train = X_train
#df_test = X_test
trueLabels = np.concatenate((y_train,y_test),axis=0)
print ('Train data shape: ', X_train.shape)
print ('Train labels shape: ', y_train.shape)
print ('Test data shape: ', X_test.shape)
print ('Test labels shape: ', y_test.shape)


def main():
    t = RBFNet()
    nlabels = 10  # total number of labels
    nclusters = 50  # number of clusters for k-means
    ksplits = 10  # ksplits-fold cross validation
    np.random.seed(0)

    # X_train = train_data[:1000,:]
    # y_train = train_labels[:1000]

    #############################################################################
    ###   Calculate optimal beta for this network and put it in to optimalBeta
    ###                              with crossvalidatin
    #############################################################################
    optimalBeta = None

    betas = [0.01, 0.49, 1, 2, 3, 4, 9, 25, 49, 81]
    N = X_train.shape[0]
    Fold_size = math.floor(N / ksplits)
    mean_cv_accuracy = np.zeros(len(betas))
    b_index = -1

    for beta in betas:
        b_index += 1
        # accuracy[i]=t.kFoldValidation(validationData,ksplits,nlabels,nclusters,beta)
        accuracy = np.zeros(ksplits)
        # def kFoldValidation(self,data,k,nlabels,nclusters,beta):
        # k_fold_cv
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=ksplits)
        cv_index = -1
        for train_index, test_index in kf.split(X_train):
            # print('TRAIN:', train_index, 'TEST:', test_index)
            X_training_fold, X_testing_fold = X_train[train_index], X_train[test_index]
            y_training_fold, y_testing_fold = y_train[train_index], y_train[test_index]
            # print(y_train.shape)
            predicted_y_training_fold, centers, centroidLabel = \
                t.trainRBF(X_training_fold, y_training_fold, nclusters, beta, nlabels, y_training_fold)
            predicted_y_testing_fold = \
                t.RBF(X_testing_fold, y_testing_fold, beta, centers, centroidLabel, nlabels)

            accuracy = np.sum(y_testing_fold == predicted_y_testing_fold) / len(y_testing_fold)
            mean_cv_accuracy[b_index] += accuracy * 100

        mean_cv_accuracy[b_index] = mean_cv_accuracy[b_index] / ksplits
        print('beta %g mean_cv_accuracy %f' % (beta, mean_cv_accuracy[b_index]))

    optimalBeta = betas[np.argmax(mean_cv_accuracy)]
    print('optimal beta: ', optimalBeta)
    optimalAccuracy = np.max(mean_cv_accuracy)
    print('Optimal Accuracy', optimalAccuracy)
    print('---------')
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    # Train
    print('training...')
    (predictedLabels_train, centers_train, centroidLabel_train) = \
        t.trainRBF(X_train, y_train, nclusters, optimalBeta, nlabels, y_train)

    # Test
    print('testing...')
    predictedTestLabels = \
        t.RBF(X_test, y_test, optimalBeta, centers_train, centroidLabel_train, nlabels)

    # Test accuracy
    testLabels = y_test
    accuracy = 0
    for y in range(len(predictedTestLabels)):
        if predictedTestLabels[y] == testLabels[y]:
            accuracy += 1
    accuracy = (accuracy / len(predictedTestLabels)) * 100
    print('Percent accuracy on test data:', accuracy)


class RBFNet():
    def __init__(self):
        pass

    def calcKmean(self, X, n):  # data -> X
        kmeanz = KMeans(n_clusters=n).fit(X)
        centers = np.array(kmeanz.cluster_centers_)
        # print('pairwise_distances_argmin_min... ')
        closest, _ = pairwise_distances_argmin_min(kmeanz.cluster_centers_, X)
        closest = np.array(closest)
        return (centers, closest)

    def RBF(self, X, y, beta, centers, centroidLabels, nlabels):  # data -> X, y
        #############################################################################
        ### Train RBF to produce predicted labels
        ### you should return predictedlabels as shape of (N,)
        #############################################################################

        centroid_scores = np.zeros((len(centers), X.shape[0]))
        for c in range(len(centers)):
            for sample_i in range(X.shape[0]):
                # print(np.squeeze(beta))
                centroid_scores[c, sample_i] = np.exp(
                    (-1. / beta) * np.square(np.linalg.norm(np.subtract(X[sample_i], centers[c]))))

        Label_scores = np.zeros((nlabels, X.shape[0]))
        for c in range(len(centroidLabels)):
            for sample_i in range(X.shape[0]):
                # print(centroidLabels[c])
                # print(centroid_scores[c, sample_i])
                Label_scores[int(centroidLabels[c]), sample_i] += centroid_scores[c, sample_i]
        predictedLabels = np.argmax(Label_scores, axis=0)
        # print(predictedLabels)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return predictedLabels

    def trainRBF(self, X, y, k, beta, nlabels, trueLabels):  # data -> X,y
        # k-means: Getting centroids and the row indices (in df_train) of the data points closest to the centroids
        t = RBFNet()
        # print('training started!')
        (centers, indices) = t.calcKmean(X, k)
        # The label of each centroid according training data
        # print('finished Kmeans!')
        centroidLabel = np.zeros(len(centers))
        for x in range(len(centers)):
            # print(y.shape)
            centroidLabel[x] = trueLabels[indices[x]]  # trueLabels[indices[x]]
        # print('trained!')
        predictedLabels = t.RBF(X, y, beta, centers, centroidLabel, nlabels)
        return (predictedLabels, centers, centroidLabel)


if __name__ == "__main__":
    main()

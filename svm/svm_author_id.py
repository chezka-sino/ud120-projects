#!/usr/bin/python3

""" 
    this is the code to accompany the Lesson 2 (SVM) mini-project

    use an SVM to identify emails from the Enron corpus by their authors
    
    Sara has label 0
    Chris has label 1

"""
    
import sys
from time import time
from sklearn import svm
from sklearn.metrics import accuracy_score
import numpy as np

sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

### make sure you use // when dividing for integer division


#########################################################
### your code goes here ###

def svm_train(X_train, Y_train, X_test, Y_test, k='linear', c = 1.0):
    clf = svm.SVC(kernel=k, C=c)

    t0 = time()
    clf.fit(X_train, Y_train)
    print("training time:", round(time() - t0, 3), "s")

    t0 = time()
    pred = clf.predict(X_test)
    print("predicting time:", round(time() - t0, 3), "s")

    print("Accuracy:", round(accuracy_score(Y_test, pred), 4), '\n')

    print(np.sum(pred))

# Training on all the training set with linear kernel
# svm_train(features_train, labels_train, features_test, labels_test)


# Training on a smaller training set
features_train_sub = features_train[:len(features_train)//100]
labels_train_sub = labels_train[:len(labels_train)//100]

# svm_train(features_train_sub, labels_train_sub, features_test, labels_test)

# Training with rbf kernel
# svm_train(features_train_sub, labels_train_sub, features_test, labels_test, 'rbf')

# C = [10.0, 100.0, 1000.0, 10000.0]
#
# for c in C:
#     print('C =', c)
#     svm_train(features_train_sub, labels_train_sub, features_test, labels_test, 'rbf', c)

svm_train(features_train, labels_train, features_test, labels_test, 'rbf', 10000)


#########################################################



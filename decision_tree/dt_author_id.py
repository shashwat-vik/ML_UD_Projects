#!/usr/bin/python

"""
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess

features_train, features_test, labels_train, labels_test = preprocess()

print len(features_train[0])
#########################################################
from sklearn import tree

clf = tree.DecisionTreeClassifier(min_samples_split=40)
t0 = time()
clf.fit(features_train, labels_train)
print "\nLEARNING TIME :", round(time()-t0, 3)
t1 = time()
pred = clf.predict(features_test)
print "PREDICTION TIME :", round(time()-t1, 3)
accuracy = sum(x == y for x, y in zip(pred, labels_test))/(len(labels_test)*0.01)
print "ACCURACY :", accuracy
#########################################################

'''
>> WHEN THE FEATURES IN DATSET IS SET TO 10 percentile
INSIGHTS:
    LEARNING TIME : 81.654
    PREDICTION TIME : 0.026
    ACCURACY : 97.7815699659

>> WHEN THE FEATURES IN DATASET IS SET TO 1 percentile
INSIGHTS:
    LEARNING TIME : 5.671
    PREDICTION TIME : 0.0
    ACCURACY : 96.6439135381
'''

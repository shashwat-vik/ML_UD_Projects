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

#########################################################
from sklearn.svm import SVC

cls = SVC(kernel='linear')
t0 = time()
cls.fit(features_train, labels_train)
print "\nLEARNING TIME :", round(time()-t0, 3)
t1 = time()
pred = cls.predict(features_test)
print "PREDICTION TIME :", round(time()-t1, 3)
accuracy = sum(x == y for x, y in zip(pred, labels_test))/(len(labels_test)*0.01)
print "ACCURACY :", accuracy
#########################################################


'''
INSIGHTS:
    LEARNING TIME : 195.006
    PREDICTION TIME : 21.291
    ACCURACY : 98.4072810011
'''

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

features_train = features_train[:len(features_train)/100]
labels_train = labels_train[:len(labels_train)/100]

#########################################################
### your code goes here ###
from sklearn.svm import SVC

cls = SVC(kernel='rbf', C=10000)
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
>> LEARNING IS DONE ON ONLY 1% of the DATASET

INSIGHTS:
    # SVC(kernel='linear')
    LEARNING TIME : 0.116
    PREDICTION TIME : 1.305
    ACCURACY : 88.4527872582

    # SVC(kernel='rbf')
    LEARNING TIME : 0.132
    PREDICTION TIME : 1.328
    ACCURACY : 61.6040955631

    # SVC(kernel='rbf', C=10000)
    LEARNING TIME : 0.105
    PREDICTION TIME : 1.071
    ACCURACY : 89.2491467577

>> NOW WE OPTIMIZED THE VALUE OF C BY TESTING ON 1% of DATASET
   HENCE EXPANDING THIS INSIGHT ON 100% of DATASET NOW

   # SVC(kernel='rbf', C=10000)
    LEARNING TIME : 127.244
    PREDICTION TIME : 12.366
    ACCURACY : 99.0898748578
'''

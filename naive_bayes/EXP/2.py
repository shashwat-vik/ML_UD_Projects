from time import time
from sklearn.svm import SVC
from class_vis import prettyPicture
from prep_terrain_data import makeTerrainData

features_train, labels_train, features_test, labels_test = makeTerrainData()

clf = SVC(kernel='rbf', C=50000.0)

#########################
t0 = time()
clf.fit(features_train, labels_train)
print "\nLEARNING TIME :", round(time()-t0, 3)
t1 = time()
pred = clf.predict(features_test)
print "PREDICTION TIME :", round(time()-t1, 3)
accuracy = sum(x == y for x, y in zip(pred, labels_test))/(len(labels_test)*0.01)
print "ACCURACY :", accuracy
#########################

prettyPicture(clf, features_test, labels_test, "2.png")

import os
os.system('2.png')

'''
INSIGHTS:
    # SVC(kernel='rbf', gamma=500)
    LEARNING TIME : 0.047
    PREDICTION TIME : 0.0
    ACCURACY : 94.4

    # SVC(kernel='rbf', C=50000.0)
    LEARNING TIME : 0.065
    PREDICTION TIME : 0.001
    ACCURACY : 95.2
'''

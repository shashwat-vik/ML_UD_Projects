from time import time
from sklearn import tree
from class_vis import prettyPicture
from prep_terrain_data import makeTerrainData

features_train, labels_train, features_test, labels_test = makeTerrainData()

clf = tree.DecisionTreeClassifier()

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

prettyPicture(clf, features_test, labels_test, "base.png")

import os
os.system("base.png")

'''
STATS :
    LEARNING TIME : 0.0
    PREDICTION TIME : 0.0
    ACCURACY : 90.8
'''

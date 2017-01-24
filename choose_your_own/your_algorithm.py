#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]

'''
#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
################################################################################
'''

### your code here!  name your classifier object clf if you want the
### visualization code (prettyPicture) to show you the decision boundary
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from time import time

clf = AdaBoostClassifier(n_estimators=200, learning_rate=0.1, max_depth=10)
#clf = RandomForestClassifier()
t0 = time()
clf.fit(features_train, labels_train)
print "\nLEARNING TIME :", round(time()-t0, 3)
t1 = time()
pred = clf.predict(features_test)
print "PREDICTION TIME :", round(time()-t1, 3)
accuracy = sum(x == y for x, y in zip(pred, labels_test))/(len(labels_test)*0.01)
print "ACCURACY :", accuracy

try:
    prettyPicture(clf, features_test, labels_test)
    import os
    os.system("test.png")
except NameError:
    pass


'''
    # RandomForestClassifier()
    LEARNING TIME : 0.032
    PREDICTION TIME : 0.015
    ACCURACY : 92.0

    # RandomForestClassifier(n_estimators=10, min_samples_split=60)
    LEARNING TIME : 0.032
    PREDICTION TIME : 0.015
    ACCURACY : 93.6

    # AdaBoostClassifier()
    LEARNING TIME : 0.185
    PREDICTION TIME : 0.0
    ACCURACY : 92.4
'''

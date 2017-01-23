from time import time
from sklearn.naive_bayes import GaussianNB
from class_vis import prettyPicture
from prep_terrain_data import makeTerrainData

features_train, labels_train, features_test, labels_test = makeTerrainData()

clf = GaussianNB()

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

prettyPicture(clf, features_test, labels_test, "1.png")

'''
STATS :
    LEARNING TIME : 0.0
    PREDICTION TIME : 0.0
    ACCURACY : 88.4
'''

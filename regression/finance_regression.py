#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
dictionary = pickle.load( open("../final_project/final_project_dataset_modified.pkl", "r") )

features_list = ["bonus", "salary"]
data = featureFormat( dictionary, features_list, remove_any_zeroes=True)
target, features = targetFeatureSplit( data )

from sklearn.cross_validation import train_test_split
feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size=0.5, random_state=42)
train_color = "b"
test_color = "r"

import matplotlib.pyplot as plt
for feature, target in zip(feature_test, target_test):
    plt.scatter( feature, target, color=test_color)
for feature, target in zip(feature_train, target_train):
    plt.scatter( feature, target, color=train_color)

# labels for the legends
plt.scatter(feature_test[0], target_test[0], color=test_color, label="test")
plt.scatter(feature_train[0], target_train[0], color=train_color, label="train")

plt.xlabel(features_list[1])
plt.ylabel(features_list[0])

#########################################
from sklearn import linear_model

reg = linear_model.LinearRegression()

reg.fit(feature_train, target_train)
plt.plot(feature_test, reg.predict(feature_test), color='black')
print reg.score(feature_test, target_test)

# REVERSING TRAIN AND TEST DATA
reg.fit(feature_test, target_test)
plt.plot(feature_train, reg.predict(feature_train), color='blue')
print reg.score(feature_train, target_train)

plt.legend()
plt.show()

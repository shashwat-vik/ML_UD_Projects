#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot as plt
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
data_dict.pop('TOTAL', 0)
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)

salary, bonus = zip(*data)
plt.scatter(salary, bonus)
plt.xlabel('salary')
plt.ylabel('bonus')
plt.show()

for key in data_dict.keys():
    if data_dict[key]['salary'] != 'NaN' and data_dict[key]['bonus'] != 'NaN':
        if data_dict[key]['salary'] > 10**6 and data_dict[key]['bonus'] >= 5*10**6:
            print key

plt.xlabel('salary')
plt.ylabel('bonus')
plt.show()

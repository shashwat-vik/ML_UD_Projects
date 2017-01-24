#!/usr/bin/python

import numpy
import matplotlib.pyplot as plt
import pickle

### load up some practice data with outliers in it
ages = pickle.load( open("practice_outliers_ages.pkl", "r") )
net_worths = pickle.load( open("practice_outliers_net_worths.pkl", "r") )

### by convention, n_rows is the number of data points
### and n_columns is the number of features
ages = numpy.reshape( numpy.array(ages), (len(ages), 1))
net_worths = numpy.reshape( numpy.array(net_worths), (len(net_worths), 1))
plt.scatter(ages, net_worths)

from sklearn.cross_validation import train_test_split
ages_train, ages_test, net_worths_train, net_worths_test = train_test_split(ages, net_worths, test_size=0.1, random_state=42)

from sklearn import linear_model
reg = linear_model.LinearRegression()
reg.fit(ages_train, net_worths_train)
print '\nINITIAL SCORE :', reg.score(ages_test, net_worths_test)
print 'INITAL SLOPE  :', reg.coef_
plt.plot(ages, reg.predict(ages), color="blue")
plt.xlabel("ages")
plt.ylabel("net worths")
plt.show()


from outlier_cleaner import outlierCleaner
cleaned_data = []
predictions = reg.predict(ages_train)
cleaned_data = outlierCleaner(predictions, ages_train, net_worths_train )

ages, net_worths, errors = zip(*cleaned_data)
ages = numpy.reshape( numpy.array(ages), (len(ages), 1))
net_worths = numpy.reshape( numpy.array(net_worths), (len(net_worths), 1))
plt.scatter(ages, net_worths)

### refitting cleaned data!
reg.fit(ages, net_worths)
print '\nFINAL SCORE :', reg.score(ages_test, net_worths_test)
print 'FINAL SLOPE :', reg.coef_
plt.plot(ages, reg.predict(ages), color="blue")
plt.xlabel("ages")
plt.ylabel("net worths")
plt.show()

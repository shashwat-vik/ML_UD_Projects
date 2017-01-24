import matplotlib.pyplot as plt
from ages_net_worths import ageNetWorthData

ages_train, ages_test, net_worths_train, net_worths_test = ageNetWorthData()

# ORIGINAL DATA
plt.scatter(ages_train, net_worths_train, color='b', label='train data')
plt.scatter(ages_test, net_worths_test, color='r', label='test data')

####################################
from sklearn import linear_model

reg = linear_model.LinearRegression()
reg.fit(ages_train, net_worths_train)
plt.plot(ages_test, reg.predict(ages_test), color='black')

plt.legend(loc=2)   #UPPER-LEFT
plt.xlabel("ages")
plt.ylabel("net worths")

plt.savefig('base.png')
####################################

import os
os.system('base.png')

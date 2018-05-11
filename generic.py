# Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing, cross_validation, svm

# Importing the dataset

data = pd.read_csv('xAPI-Edu-Data.csv')

# Data pre-processing

arr = []
for i in range(17):
    if i != 1 and i !=4 and i!=7:
        arr.append(i)
f = [0,1,2,3,4,5,10,11,12,13]

X = data.iloc[:, arr].values

dependent = data.iloc[:, 4].values
y = []
dependent = np.array(dependent)
for i in dependent:
    y.append(int(str(i)[-2:]))
y = np.array(y)


# Encoding the dataset

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
for i in f:
    X[:, i] = labelencoder.fit_transform(X[:, i])
onehotencoder = OneHotEncoder(categorical_features = f)
X = onehotencoder.fit_transform(X).toarray()

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

y_test = y_test.astype(np.float64)
y_train = y_train.astype(np.float64)
y = y.astype(np.float64)


#fitting multiple-linear regression to training set

from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
#regressor= svm.SVR()
regressor.fit(X_train , y_train)

accuracy = regressor.score(X_test, y_test)
accuracy = "{:.3f}".format(accuracy * 100)
print("Accuracy is " + str(accuracy) + " %")

#predicting test set results
y_pred= regressor.predict(X_test)

t = [i for i in range(96)]


#ploting results on training set

plt.plot(t,y_test,t,y_pred)
plt.title('Plot')
plt.xlabel('X axis')
plt.ylabel('y axis')
plt.show()
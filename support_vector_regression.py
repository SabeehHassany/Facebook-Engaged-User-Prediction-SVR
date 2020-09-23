"""
Support Vector Regression is a more advanced version of linear regression. It uses support vectors which are data points not on the line as a way to stabilize itself on the line as well be able to expect vectors as predicted values. This dataset is retrieved from the UCI Machine Learning repository and contains 2014 Facebook post metrics from over 500 posts from an international cosmetics company. It has multiple independent and dependent variables that we can use. In this case, I used the metrics to predict Lifetime Engaged Users as a baseline (although any other predicted variable can be used).

### Importing the libraries
These are the three go to libraries for most ML.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""### Importing the dataset
I imported the dataset through Pandas dataframe and used iloc to assign the variables. Remember that the name of the dataset has to be updated for diff usecases AND it must be in the same folder as your .py file or uploaded on Jupyter Notebooks or Google Collab.
"""

dataset = pd.read_csv('dataset_Facebook.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
y = y.reshape(-1, 1)
#print(dataset.info())
# ^ this is just to see the missing entries in our data. As an alternative you can also use conditional formatting on the csv.

"""### Missing Data
For this dataset there were a total of 3 columns with missing data. For index 8 and 9 (last index is not included hence '8:10') I used the 'median' to impute to compensate for outliers. For index 6 I used 'most_frequent' because it was a binary datapoint.
"""

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='median')
imputer.fit(X[:, 8:10])
X[:, 8:10] = imputer.transform(X[:, 8:10])

imputer2 = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imputer2.fit(X[:, 6:7])
X[:, 6:7] = imputer2.transform(X[:, 6:7])

"""### Encoding categorical data
Index 1 had categorical data that had to be converted using OneHotEncoding.This must be done AFTER imputing missing values since OneHotEncoding automatically makes the encoded column the first index which displaces all the rest.
"""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = ct.fit_transform(X)

"""### Feature Scaling
For SVR we always have to feature scale our data. Both X and Y. This is because it can help our model find vectors much easier which can help increase efficicy, especially as variables increase.
"""

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X[:, 4:] = sc_X.fit_transform(X[:, 4:])
y = sc_y.fit_transform(y)

"""### Splitting the dataset into the Training set and Test set
Because of the large dataset and many variables for a simple algorothim I used a 90/10 split. The random state is tuned to 5 for consistency sakes.
"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10, random_state = 5)

"""### Training the SVR model on the whole dataset
There are a lot of different kernels that we can use with SVR but using a radial basis function (rbf) is best as its optimized for support vector machines and uses euclidean distance measurements.
"""

from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X_train, y_train)

"""### Predicting the Test set results
By using the concatenate function I display the predicted values and  actual values in a side by side 2D array through '(len(y_wtv), 1))' for easy viewing.
"""

y_pred = sc_y.inverse_transform(regressor.predict(X_test))
y_test = sc_y.inverse_transform(y_test)
np.set_printoptions(precision=0, suppress=True)
#print(np.concatenate((y_pred.astype(int).reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

#y_pred = regressor.predict(X_test)
#np.set_printoptions(precision=0, suppress=True)
#print(np.concatenate((y_pred.astype(int).reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

"""### Evaluating Model Performance
We use two metrics to evaluate our model performance, r^2 being the more superior. These are both simple to understand and are covered in one of my Medium articles! In this model we achieved a .80 r2 which means 80% of our data can be predicted by our linear regression! Even when we use 'random_state = None' which creates a different test_train_split each time, we get upwards of a .70 r2!
"""

from sklearn.metrics import r2_score, mean_squared_error as mse
print("r^2: " + str(r2_score(y_test, y_pred)))
print("MSE: " + str(mse(y_test, y_pred)))
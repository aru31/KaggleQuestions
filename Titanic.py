# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset1 = pd.read_csv('train.csv')
dataset1['Embarked'] = dataset1['Embarked'].fillna("S")
dataset1['Sex'] = dataset1['Sex'].fillna("female")

X_train = dataset1.iloc[:, [2, 4, 5, 6, 7, 9, 11]].values
y_train = dataset1.iloc[:, 1].values

# Test Set
dataset2 = pd.read_csv('test.csv')
dataset1['Embarked'] = dataset1['Embarked'].fillna("S")
dataset1['Sex'] = dataset1['Sex'].fillna("female")
X_test = dataset2.iloc[:, [1, 3, 4, 5, 6, 8, 10]].values


dataset3 = pd.read_csv('gender_submission.csv')
y_test = dataset3.iloc[:, 1].values

# 0, 2, 3 are PClass, Gender and Age respetively of X_train
#Taking care of missing data....
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X_train[:, [0, 2, 3, 4, 5]])
X_train[:, [0, 2, 3, 4, 5]] = imputer.transform(X_train[:, [0, 2, 3, 4, 5]])
X_test[:, [0, 2, 3, 4, 5]] = imputer.transform(X_test[:, [0, 2, 3, 4, 5]])


# Encoding Categorical Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_train = LabelEncoder()
X_train[:, 1] = labelencoder_X_train.fit_transform(X_train[:, 1])
onehotencoder = OneHotEncoder(categorical_features = [1])
X_train = onehotencoder.fit_transform(X_train).toarray()

labelencoder_X_test = LabelEncoder()
X_test[:, 1] = labelencoder_X_test.fit_transform(X_test[:, 1])
onehotencoder = OneHotEncoder(categorical_features = [1])
X_test = onehotencoder.fit_transform(X_test).toarray()

labelencoder_X_train = LabelEncoder()
X_train[:, 6] = labelencoder_X_train.fit_transform(X_train[:, 6])
onehotencoder = OneHotEncoder(categorical_features = [6])
X_train = onehotencoder.fit_transform(X_train).toarray()

labelencoder_X_test = LabelEncoder()
X_test[:, 6] = labelencoder_X_test.fit_transform(X_test[:, 6])
onehotencoder = OneHotEncoder(categorical_features = [6])
X_test = onehotencoder.fit_transform(X_test).toarray()


X_train = X_train[:, 1:]
X_test = X_test[:, 1:]

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# Fitting classifier to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', gamma=0.5, C=0.8)
classifier.fit(X_train, y_train)


# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

classifier.score(X_train, y_train)
classifier.score(X_test, y_test)





# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()





# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV as GSCV
# Will be a list of Dictionary
# Can make this dictionary by optimizing the values that we need
# to put in class SVC...(in this question)
parameters = [{'C': [1, 0.9, 0.8, 0.7]}
              ]
# Grid Search investigates all different Combinations and brings
# out the best one
# cv i.e Applying k-fold
grid_search = GSCV(estimator = classifier, param_grid = parameters, scoring = 'accuracy', cv = 10)
grid_search = grid_search.fit(X_train, y_train)
# accuracy that we get through 10 fold validation
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_


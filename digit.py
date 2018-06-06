
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.utils import np_utils

# Importing the dataset
dataset1 = pd.read_csv('train.csv')
X = dataset1.iloc[:, 1:].values
y = dataset1.iloc[:, 0].values

dataset2 = pd.read_csv('test.csv')
AX = dataset2.iloc[:, 0:].values


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Encoding categorical data
n_classes = 10
print("Shape before one-hot encoding: ", y_train.shape)
y_train = np_utils.to_categorical(y_train, n_classes)
y_test = np_utils.to_categorical(y_test, n_classes)
print("Shape after one-hot encoding: ", y_train.shape)


# Feature Scaling
# Feature Scaling is the most important in Deep Learning Models
X_train = (X_train)/255
X_test = (X_test)/255
AX = (AX)/255

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 512, init = 'uniform', activation = 'relu', input_dim = 784))

# Adding the output layer
classifier.add(Dense(output_dim = 10, init = 'uniform', activation = 'softmax'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 64, nb_epoch = 15)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(AX)
preds = classifier.predict_classes(AX, verbose=0)



score = classifier.evaluate(X_test, y_test)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

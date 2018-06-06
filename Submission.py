# Classification template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset1 = pd.read_csv('train_HK6lq50.csv')
dataset2 = pd.read_csv('test_2nAIblo.csv')

## To count frequency of string variables
dataset1.isnull().sum() # To get all the nan values
pd.value_counts(dataset1['trainee_engagement_rating'].values, sort=False)

X_ = dataset1.iloc[:, [1, 3, 5, 6, 8, 9, 10, 11, 12, 13, 14]].values
y = dataset1.iloc[:, 15].values

X_ans_ = dataset2.iloc[:, [1, 3, 5, 6, 8, 9, 10, 11, 12, 13, 14]].values
"""
1)-> program_id -> 0
2-> program_type(categorical) -> XXXXXXXXXXXXXXXXXXXX
3-> program_duration(numerical) -> 1
5-> test_type(offline/online) ->2
6-> Difficulty Level(3 categories) -> 3
8-> gender(M/F) -> 4
9-> education(categorical) -> 5
10-> city_tier -> 6
11-> age -> 7
12-> total_programs_enrolled -> 8 
13-> is_handicapped(y or n) -> 9
14-> trainee_engagement_rating(0 to 5) -> 10
"""

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'median', axis = 0)
imputer = imputer.fit(X_[:, [7]])
X_[:, [7]] = imputer.transform(X_[:, [7]])

imputer3 = Imputer(missing_values = 'NaN', strategy = 'median', axis = 0)
imputer3 = imputer3.fit(X_ans_[:, [7]])
X_ans_[:, [7]] = imputer3.transform(X_ans_[:, [7]])


imputer1 = Imputer(missing_values = 'NaN', strategy = 'most_frequent', axis = 0)
imputer1 = imputer1.fit(X_[:, [10]])
X_[:, [10]] = imputer1.transform(X_[:, [10]])

imputer2 = Imputer(missing_values = 'NaN', strategy = 'most_frequent', axis = 0)
imputer2 = imputer2.fit(X_ans_[:, [10]])
X_ans_[:, [10]] = imputer2.transform(X_ans_[:, [10]])


# 6
df1 = pd.DataFrame(X_)
X = pd.get_dummies(df1, columns =[0, 2, 3, 4, 5, 9], drop_first=True)

df2 = pd.DataFrame(X_ans_)
X_ans = pd.get_dummies(df2, columns =[0, 2, 3, 4, 5, 9], drop_first=True)


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_ans = sc.fit_transform(X_ans)

# Fitting classifier to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X, y)
classifier.fit(X_train, y_train)

from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

classifier.probability = True
predAns = classifier.predict_proba(X_ans)[:,1]

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 2)
accuracies.mean()
accuracies.std()


classifier.score(X_test, y_test)


# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Classifier (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Classifier (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


# Read the submission file
submission=pd.read_csv("sample_submission_vaSxamm.csv")

# Fill the is_pass variable with the predictions
submission['is_pass'] = predAns


# Converting the submission file to csv format
submission.to_csv('logistic_submission.csv', index=False)


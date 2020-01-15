# KNN on breast cancer dataset
# dataset link: https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/

from sklearn import neighbors, preprocessing, model_selection
import numpy as np 
import pandas as pd 
import pickle

# loading dataset
dataset = pd.read_csv('breast-cancer-wisconsin.data')
# replacing missing data
dataset.replace('?', -99999, inplace=True)
# dropping the id column, because it won't come to any use
dataset.drop(['id'], 1, inplace=True)

# getting the X and y
X = np.array(dataset.drop(['class'], 1))
y = np.array(dataset['class'])

# splitting the train and test values
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

# running the classifier
clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)

# dumping the classifier to a file
# clf_file = open('clf.pickle', 'wb')
# pickle.dump(clf, clf_file)

# reading the classifier from a file
# clf_file = open('clf.pickle', 'rb+')
# clf = pickle.load(clf_file)

# calculating the accuracy of the classifier
accuracy = clf.score(X_test, y_test)
print(accuracy)

# predicting a data
X_predict = np.array([[5,8,3,5,9,3,4,3,7]])
y_predict = clf.predict(X_predict)
print(y_predict)

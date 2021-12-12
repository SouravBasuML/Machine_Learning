"""
---------------------------------------------------------------------------------------------------------------------
Saving a Model:
---------------------------------------------------------------------------------------------------------------------
1. Pickling:
    import pickle
    Serialization of any python object. We pickle a classifier so we don't have to train the model everytime

2. Joblib:
    import joblib
    https://scikit-learn.org/stable/modules/model_persistence.html
    Joblib is better than pickle if the object has large number of numpy arrays internally
---------------------------------------------------------------------------------------------------------------------

"""

import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection, neighbors
import pickle, joblib


# 1. Pickling:
# if __name__ == '__main__':
#
#     df = pd.read_csv('breast-cancer-wisconsin.data')
#     df.replace('?', -99999, inplace=True)               # Model will treat -99999 as an outlier and will ignore
#     df.drop(['id'], axis=1, inplace=True)               # If you don't drop id, accuracy drops to below 60%
#
#     X = np.array(df.drop(['class'], axis=1))
#     y = np.array(df['class'])
#     X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
#
#     clf = neighbors.KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
#     clf.fit(X_train, y_train)
#
#     # Save the classifier in a .pickle file once it is trained
#     with open('model.pickle', 'wb') as f:               # wb as we will write binary data
#         pickle.dump(clf, f)
#     # Open the .pickle file and load the classifier
#     # Once saved, we can do away with the last 4 lines of code of training and saving the classifier
#     pickle_in = open('model.pickle', 'rb')
#     clf = pickle.load(pickle_in)
#
#     accuracy = clf.score(X_test, y_test)
#     print('Accuracy:', accuracy)


X_pred = np.array([[4, 2, 2, 3, 1, 1, 3, 1, 1], [8, 7, 8, 3, 1, 1, 3, 1, 1]])
y_pred = clf.predict(X_pred)
print('Prediction:', y_pred)


# Joblib:
if __name__ == '__main__':

    df = pd.read_csv('breast-cancer-wisconsin.data')
    df.replace('?', -99999, inplace=True)               # Model will treat -99999 as an outlier and will ignore
    df.drop(['id'], axis=1, inplace=True)               # If you don't drop id, accuracy drops to below 60%

    X = np.array(df.drop(['class'], axis=1))
    y = np.array(df['class'])
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

    clf = neighbors.KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    clf.fit(X_train, y_train)

    # Save the model
    joblib.dump(clf, 'model_joblib')

    # Load the model
    clf = joblib.load('model_joblib')

    accuracy = clf.score(X_test, y_test)
    print('Accuracy:', accuracy)

    X_pred = np.array([[4, 2, 2, 3, 1, 1, 3, 1, 1], [8, 7, 8, 3, 1, 1, 3, 1, 1]])
    y_pred = clf.predict(X_pred)
    print('Prediction:', y_pred)

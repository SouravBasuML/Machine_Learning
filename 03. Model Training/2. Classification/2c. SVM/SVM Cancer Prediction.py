import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection, svm


# https://archive.ics.uci.edu/ml/datasets.php

#  #  Attribute                       Domain
#  -- -----------------------------------------
#  1. Sample code number              id number
#  2. Clump Thickness                 1 - 10
#  3. Uniformity of Cell Size         1 - 10
#  4. Uniformity of Cell Shape        1 - 10
#  5. Marginal Adhesion               1 - 10
#  6. Single Epithelial Cell Size     1 - 10
#  7. Bare Nuclei                     1 - 10
#  8. Bland Chromatin                 1 - 10
#  9. Normal Nucleoli                 1 - 10
# 10. Mitoses                         1 - 10
# 11. Class:                          2 for benign, 4 for malignant
# ---------------------------------------------------------------------------------------------------------------------
# There are 16 instances in Groups 1 to 6 that contain a single missing (i.e., unavailable) attribute value,
# now denoted by "?".
# ---------------------------------------------------------------------------------------------------------------------
# Class distribution:
#    Benign: 458 (65.5%)
#    Malignant: 241 (34.5%)
# ---------------------------------------------------------------------------------------------------------------------

df = pd.read_csv('breast-cancer-wisconsin.data')
df.replace('?', -99999, inplace=True)                   # Model will treat -99999 as an outlier and will ignore
df.drop(['id'], axis=1, inplace=True)                   # If you don't drop id, accuracy drops to below 60%

X = np.array(df.drop(['class'], axis=1))
y = np.array(df['class'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

clf = svm.SVC()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print('Accuracy:', accuracy)


# X_pred = np.array([[4, 2, 2, 3, 1, 1, 3, 1, 1], [8, 7, 8, 3, 1, 1, 3, 1, 1]])
# X_pred = X_pred.reshape(len(X_pred), -1)
# y_pred = clf.predict(X_pred)
# print('Prediction:', y_pred)

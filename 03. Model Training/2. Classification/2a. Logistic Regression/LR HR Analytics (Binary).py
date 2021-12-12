import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection
from sklearn.linear_model import LogisticRegression


# ---------------------------------------------------------------------------------------------------------------------
# Logistic Regression:
# ---------------------------------------------------------------------------------------------------------------------
# Uses Sigmoid or Logit function, which converts inputs into a range between 0 and 1
# Feed y = mx + b into Sigmoid function's 'z'.
# ---------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------------------------------
# Binary Classification:
# ---------------------------------------------------------------------------------------------------------------------
def convert_cols_to_numeric():
    # Convert non-numeric columns to numeric
    le = preprocessing.LabelEncoder()
    df.department = le.fit_transform(df.department)
    df.salary = le.fit_transform(df.salary)


def exploratory_data_analysis():
    pd.set_option('display.max_columns', 10)

    # Check if any columns have NA
    print(df.columns[df.isna().any()])

    print(df[df.left == 1].shape)               # Left: (3571, 10)
    print(df[df.left == 0].shape)               # Retained: (11428, 10)
    print(df[df.left == 1].describe())

    # Based on the mean, we can decide which columns impact employees' decision to leave,
    # and hence we can decide which columns to keep for training the model
    print(df.groupby('left').mean())


if __name__ == '__main__':

    # https://www.kaggle.com/giripujar/hr-analytics
    df = pd.read_csv('hr_analytics.csv')

    # EDA:
    exploratory_data_analysis()

    # Convert 'department' and 'salary' columns to numeric:
    convert_cols_to_numeric()

    X = np.array(df.drop(['left'], axis='columns'))
    y = np.array(df['left'])
    X = preprocessing.StandardScaler().fit_transform(X)

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

    clf = LogisticRegression()
    clf.fit(X_train, y_train)

    # prediction = clf.predict(X_test)
    # print(clf.predict_proba(X_test))

    print('Accuracy: ', clf.score(X_test, y_test))

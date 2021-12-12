"""
-----------------------------------------------------------------------------------------------------------------------
Handling Missing Data:
-----------------------------------------------------------------------------------------------------------------------
Analyze each column to identify if the missing data is because:
    - it wasn't recorded - fill in the missing data based on other values in the row and column (Impute)
    - it doesn't exist - may keep as NaN or replace with some other identifier

1. Drop columns with missing data - not recommended

2. Imputation: fills in missing values with some value e.g. mean, most_frequent
    strategy:
        a. mean             - this is the default
        b. median           - use if there are outliers
        c. most_frequent    - may lead to imbalance within that feature
        d. constant         - e.g. replace missing categorical features with label "Missing"

3. Extended Imputation:
    Imputation is the standard approach, and it usually works well. However, imputed values may be systematically
    above/below their actual values (which weren't collected in the dataset). Or rows with missing values may be unique
    in some other way. In that case, our model would make better predictions by considering which values were
    originally missing.
    In extended imputation, we impute the missing values, as before. Additionally, for each column with missing entries
    in the original dataset, we add a new column that shows the location of the imputed entries. In some cases, this
    will meaningfully improve results. In other cases, it doesn't help at all.

4. Use a Machine Learning Algorithm to predict the missing value (for categorical variables):
    a. Classifier:
        Drop the categorical feature having missing data, use the remaining features and the label as the new feature
        set, and create a classification model to predict the categorical column having missing data. Once the model
        is created, use it to predict the missing values. Use rows with missing data as the test set.
    b. Clustering:
        Drop the categorical feature having missing data, and use the remaining features to cluster the data.
        The number of clusters will be equal to the number of unique categories in the feature with missing data.
        Once the model is created, use it to identify to which cluster the rows with the missing data belong.
-----------------------------------------------------------------------------------------------------------------------
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor


# 1. Drop columns with missing data:
def drop_missing_data():

    # Initial dataset shape: (2930, 81)
    # After Dropping alphanumeric columns: (2930, 38)
    # After dropping missing data: (2930, 27)
    df1 = df.dropna(axis='columns')                                 # (2930, 27)

    X = df1.drop(['SalePrice'], axis='columns')
    y = df1.SalePrice
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    clf = RandomForestRegressor(n_estimators=200, min_samples_split=20)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print('\nDropping cols with missing data:')
    print('Mean Absolute Error: ', mean_absolute_error(y_pred, y_test))
    print('Accuracy: ', clf.score(X_test, y_test))


# 2. Imputation:
def simple_imputation():
    imputer = SimpleImputer(strategy='mean')
    df2 = pd.DataFrame(imputer.fit_transform(df))               # (2930, 38)

    # missing_cols = list(df.columns[df2.isna().any()])
    # print(len(missing_cols))                                    # 0

    # Imputation removes column names. We need to add them back:
    df2.columns = df.columns

    X = df2.drop(['SalePrice'], axis='columns')
    y = df2.SalePrice
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    clf = RandomForestRegressor(n_estimators=200, min_samples_split=20)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print('\nImputing:')
    print('Mean Absolute Error: ', mean_absolute_error(y_pred, y_test))
    print('Accuracy: ', clf.score(X_test, y_test))


# 3. Extended Imputation:
def extended_imputation():
    # Add new columns indicating what will be imputed:
    missing_cols = [col for col in df.columns if df[col].isnull().any()]
    df3 = df.copy()
    for col in missing_cols:
        df3[col + '_was_missing'] = df3[col].isnull()               # (2930, 49) -> df3 has missing data.

    imputer = SimpleImputer(strategy='mean')
    df4 = pd.DataFrame(imputer.fit_transform(df3))                  # (2930, 49) -> no missing data in df4

    # Imputation removes column names. We need to add them back:
    df4.columns = df3.columns

    X = df4.drop(['SalePrice'], axis='columns')
    y = df4.SalePrice
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    clf = RandomForestRegressor(n_estimators=200, min_samples_split=20)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print('\nExtended Imputing:')
    print('Mean Absolute Error: ', mean_absolute_error(y_pred, y_test))
    print('Accuracy: ', clf.score(X_test, y_test))


if __name__ == '__main__':
    df = pd.read_csv('Iowa_Housing.csv', index_col='Order')
    df = df.select_dtypes(exclude=['object'])                                       # Keep only numerical data types

    # # Identify columns with missing data:
    # missing_cols = list(df.columns[df.isna().any()]) or
    # missing_cols = [col for col in df.columns if df[col].isnull().any()]
    # print(len(missing_cols))                                                      # 11
    # print(df.isna().any())
    # print(df.isna().sum())

    drop_missing_data()
    simple_imputation()
    extended_imputation()

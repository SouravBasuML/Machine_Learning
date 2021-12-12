"""
-----------------------------------------------------------------------------------------------------------------------
Categorical Variable Types:
-----------------------------------------------------------------------------------------------------------------------
Nominal Variables:
A variable that has two or more categories, but there is no intrinsic ordering to the categories.
e.g. Yes/No, R/G/B, Male/Female

Ordinal Variables:
A variable that has clear ordering of the categories
e.g. Low/Med/High, Small/Med/Large, Satisfied/Neutral/Dissatisfied

Interval Variables:
Similar to an ordinal, except that the intervals between the values of the numerical variable are equally spaced.
e.g. Income (5K, 10K, 15K)
-----------------------------------------------------------------------------------------------------------------------


-----------------------------------------------------------------------------------------------------------------------
Encoding: Handling Text Data in Feature set:
-----------------------------------------------------------------------------------------------------------------------

-----------------------------------------------------------------------------------------------------------------------
1. Integer/Label/Ordinal Encoding (Ordinal):
-----------------------------------------------------------------------------------------------------------------------
    Each unique value in the column is assigned different integer values e.g. low=1, med=2, high=3
    This may result in the model assuming relationships between the values of a column. e.g. low < med < high
    Integer encoding can be applied to Ordinal Variables. For tree-based models (like decision trees and random
    forests). you can expect ordinal encoding to work well with ordinal variables.

    Both OrdinalEncoder and LabelEncoder have the same functionality. OrdinalEncoder is for converting features, while
    LabelEncoder is for converting target variable. That's why OrdinalEncoder can fit data that has the shape of
    (n_samples, n_features) while LabelEncoder can only fit data that has the shape of (n_samples,)
    Ordinal Encoding can be implemented in two ways:
        a. Pandas factorize()
        b. sklearn OrdinalEncoder, LabelEncoder

-----------------------------------------------------------------------------------------------------------------------
2. One-Hot Encoding (Nominal):
-----------------------------------------------------------------------------------------------------------------------
    You create a new column for each category in the column that has Nominal Variables and assign binary value 0 or 1,
    indicating the absence (0) or presence (1) of a value. This approach does not assume an ordering of the categories.
    One-hot encoding generally does not perform well if the categorical variable takes on a large number of values.

    Identify Cardinality of all categorical columns of the feature set. Cardinality is the number of unique entries of
    a categorical variable. Apply one-hot encoding to columns having low cardinality. High cardinality columns can
    either be dropped from the dataset, or we can use ordinal encoding.

    Another approach is to limit one-hot encoding to the 10 most frequent labels of a feature variable. That is, for a
    feature, create one binary variable for each of the 10 most frequent labels only, and group all the other labels
    under a new category, which can be dropped or included as 'one' category for one-hot encoding.

    The new columns that are created by one-hot encoding are also called Dummy Variables.
    One-Hot Encoding can be implemented in two ways:
        a. Pandas get_dummies()
        b. sklearn OneHotEncoder

    handle_unknown='ignore': avoids errors when the validation data contains classes that aren't represented in the
                             training data
    sparse=False           : ensures that the encoded columns are returned as a numpy array (instead of a sparse matrix)

-----------------------------------------------------------------------------------------------------------------------
3. Target Encoding:
-----------------------------------------------------------------------------------------------------------------------
3a. Target Guided Ordinal Encoding (Ordinal):
    This method uses the target (or label) to create the encoding. It is a supervised feature engineering technique.
    Read the feature (that has ordinal categories) along with the target. Take mean of the label values for each of the
    categories, and rank the categories based on the mean (e.g. category with the highest mean gets the highest rank).
    Replace the categorical values in the feature with the calculated rank for that category.

3b. Mean Encoding (Nominal):
    Similar to Target Guided Encoding. Read the feature along with the label. Take mean of the label values for
    each of the categories. Replace the categorical values in the feature with the calculated mean of that category. If
    applied to a binary target, it's also called 'bin counting'. Other names include: likelihood encoding, impact
    encoding, and leave-one-out encoding.

    Problems with Target/Mean Encoding:
    -----------------------------------
    i. Unknown categories:
            When you join the encoding to future split, Pandas will fill in missing values for any categories not
            present in the encoding split. These missing values need to be imputed.

    ii. Rare categories:
             When a category only occurs a few times in the dataset, any statistics calculated on its group are
             unlikely to be very accurate. Target encoding rare categories can make over-fitting more likely.

    Solution (Smoothing):
    ---------------------
    A solution to the above problems is to add smoothing, where we blend the in-category average with the overall
    average. Rare categories get less weight on their category average, while missing categories just get the overall
    average.

        encoding = weight * in_category_avg + (1 - weight) * overall_avg
            where weight is a value between 0 and 1 calculated from the category frequency.

            An easy way to determine the value for weight is to compute an m-estimate:
                weight = n / (n + m)
                    where n is the total number of times that category occurs in the data. The parameter m determines
                    the "smoothing factor". Larger values of m put more weight on the overall estimate. When choosing a
                    value for m, consider how noisy you expect the categories to be. For noisy data (data with high
                    variation), choose large value for m, and vice versa.

    Implementation:
    ---------------
        from category_encoders import MEstimateEncoder

            To avoid over-fitting, apply the MEstimateEncoder encoding (apply fit()) to a sample of data taken from the
            training set, say 20%, and then use it (i.e. apply transform()) on the remaining 80% to create X_train.

    Use Cases for Target Encoding:
    ------------------------------
    1. High-cardinality features:
            A feature with a large number of categories can be troublesome to encode: a one-hot encoding would generate
            too many features and alternatives, like a label encoding, might not be appropriate for that feature. A
            target encoding derives numbers for the categories using the feature's most important property: its
            relationship with the target.

    2. Domain-motivated features:
            From prior experience, you might suspect that a categorical feature should be important even if it scored
            poorly with a feature metric. A target encoding can help reveal a feature's true informativeness.

-----------------------------------------------------------------------------------------------------------------------
4. Count/Frequency Encoding (Nominal):
-----------------------------------------------------------------------------------------------------------------------
    Used for nominal categories having high cardinality. Replace each label of the categorical variable by the:
        - count (number of times each label appears in the dataset) or
        - frequency (percentage of observations within that category)
    If some labels have the same count/frequency, they will be replaced by the same value, thus losing information.

-----------------------------------------------------------------------------------------------------------------------
5. Probability Ratio Encoding (Nominal):
-----------------------------------------------------------------------------------------------------------------------
    Find probability of output labels, based on the categorical feature
    Replace categorical column with prob(1) / prob (0) -> for binary classification
-----------------------------------------------------------------------------------------------------------------------
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer


# 1. Drop columns with categorical variables:
def drop_categorical_variables():
    df1 = df.select_dtypes(exclude=['object'])                      # (2930, 38) -> 41 columns dropped

    # Imputation. Fill missing numerical values with mean of the column values
    imputer = SimpleImputer(strategy='mean')
    df2 = pd.DataFrame(imputer.fit_transform(df1))                  # (2930, 38)
    df2.columns = df1.columns                                       # Adding back column names removed by Imputation

    X = df2.drop(['SalePrice'], axis='columns')
    y = df2.SalePrice
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    clf = RandomForestRegressor(n_estimators=200, min_samples_split=20)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print('\nDropping categorical variables:')
    print('Mean Absolute Error: ', mean_absolute_error(y_pred, y_test))
    print('Accuracy: ', clf.score(X_test, y_test))


# 2. Ordinal/Integer/Label Encoding:
def ordinal_encoding():
    ordinal_encoder = OrdinalEncoder()
    df_oe = df
    df_oe[object_cols] = ordinal_encoder.fit_transform(df[object_cols])            # (2930, 81)

    # Imputation. Fill missing numerical values with mean of the column values
    imputer = SimpleImputer(strategy='mean')
    df3 = pd.DataFrame(imputer.fit_transform(df_oe))                # (2930, 81)
    df3.columns = df_oe.columns                                     # Adding back column names removed by Imputation

    X = df3.drop(['SalePrice'], axis='columns')
    y = df3.SalePrice
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    clf = RandomForestRegressor(n_estimators=200, min_samples_split=20)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print('\nOrdinal Encoding:')
    print('Mean Absolute Error: ', mean_absolute_error(y_pred, y_test))
    print('Accuracy: ', clf.score(X_test, y_test))


# 3. One Hot Encoding:
def one_hot_encoding():
    oh_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    df_OHE = pd.DataFrame(oh_encoder.fit_transform(df[object_cols]))        # (2930, 284) -> Only OH encoded columns

    # One-Hot Encoding removes the index; adding it back
    df_OHE.index = df.index

    # Drop categorical cols from original DF (will replace with OH encoding). df_num contains numeric columns with NaN
    df_num = df.drop(object_cols, axis=1)                                   # (2930, 38) -> 41 columns dropped

    # Imputation. Fill missing numerical values with mean of the column values
    imputer = SimpleImputer(strategy='mean')
    df_num_imp = pd.DataFrame(imputer.fit_transform(df_num))        # (2930, 38)
    df_num_imp.columns = df_num.columns                             # Adding back column names removed by Imputation
    df_num_imp.index = df_num.index

    # Add one-hot encoded columns (df_OHE) to imputed numerical features (df_num_imp)
    df_final = pd.concat([df_num_imp, df_OHE], axis=1)

    X = df_final.drop(['SalePrice'], axis='columns')
    y = df_final.SalePrice
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    clf = RandomForestRegressor(n_estimators=200, min_samples_split=20)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print('\nOne-Hot Encoding:')
    print('Mean Absolute Error: ', mean_absolute_error(y_pred, y_test))
    print('Accuracy: ', clf.score(X_test, y_test))


if __name__ == '__main__':
    df = pd.read_csv('Iowa_Housing.csv', index_col='Order')         # (2930, 81)

    # Identify columns with missing data:
    s = (df.dtypes == 'object')
    object_cols = list(s[s].index)
    print('Categorical variables: ', object_cols)                   # 43 columns with text values

    drop_categorical_variables()
    ordinal_encoding()
    one_hot_encoding()


# ---------------------------------------------------------------------------------------------------------------------
# One-Hot Encoding using Pandas get_dummies()
# ---------------------------------------------------------------------------------------------------------------------
# df = pd.read_csv('home_prices.csv')
#
# # get_dummies creates a new column for each unique value in town column and assign binary value 0 or 1.
# dummies = pd.get_dummies(df.town)
# print(dummies)
#
# # To avoid dummy variable trap, drop any one of the new columns. We can do it manually or use drop_first parameter
# # Dummy Variable trap is a scenario in which the independent variables are multicollinear - a scenario in which
# # two or more variables are highly correlated; i.e. one variable can be predicted from the others.
# dummies = pd.get_dummies(df.town, drop_first=True)
# print(dummies)
#
# # Merge the dummies df with the original df:
# merged_df = pd.concat([df, dummies], axis='columns')
#
# # Drop the original column that has text values.
# final_df = merged_df.drop(['town'], axis='columns')
# print(final_df)
#
# X = final_df.drop('price', axis='columns')
# y = final_df.price
#
# clf = LinearRegression()
# clf.fit(X, y)
#
# print('2800 sq. ft. house in robinsville: ', clf.predict([[2800, 1, 0]]))
# print('3400 sq. ft. house in monroe township: ', clf.predict([[3400, 0, 0]]))
# print('Accuracy: ', clf.score(X, y))

# ---------------------------------------------------------------------------------------------------------------------
# One-Hot Encoding using sklearn OneHotEncoder
# ---------------------------------------------------------------------------------------------------------------------
# df = pd.read_csv('home_prices.csv')
#
# le = LabelEncoder()
# df_le = df
#
# # Encode town column - convert text values to numeric values 0, 1, 2, ... etc.
# df_le.town = le.fit_transform(df_le.town)
# print(df_le)
#
# X = df_le[['town', 'area']].values
# y = df_le.price
#
# # Apply One-Hot Encoding:
# ct = ColumnTransformer([('one_hot_encoder', OneHotEncoder(categories='auto'), [0])], remainder='passthrough')
# X = ct.fit_transform(X)
#
# # Drop any one column to avoid dummy variable trap (here dropping first column)
# X = X[:, 1:]
#
# clf = LinearRegression()
# clf.fit(X, y)
#
# print('2800 sq. ft. house in Robinsville: ', clf.predict([[1, 0, 2800]]))
# print('3400 sq. ft. house in Monroe township: ', clf.predict([[0, 0, 3400]]))
# print('Accuracy: ', clf.score(X, y))
# ---------------------------------------------------------------------------------------------------------------------

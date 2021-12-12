"""
-----------------------------------------------------------------------------------------------------------------------
Pipeline:
-----------------------------------------------------------------------------------------------------------------------
The sklearn.pipeline module implements utilities to build a composite estimator, as a chain of transforms
and an estimator.
-----------------------------------------------------------------------------------------------------------------------
sklearn.pipeline.Pipeline
Pipeline of transforms with a final estimator. Sequentially apply a list of transforms and a final estimator.
Intermediate steps of the pipeline must be ‘transforms’, that is, they must implement fit and transform methods.
The final estimator only needs to implement fit. The purpose of the pipeline is to assemble several steps that
can be cross-validated together while setting different parameters.
-----------------------------------------------------------------------------------------------------------------------
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error


def apply_column_transform():
    # Create a transformer to preprocess numerical data. Missing data will be replaced with mean of the column values
    numerical_transformer = SimpleImputer(strategy='mean')

    # Create a transformer to preprocess categorical data:
    # 1. Impute missing values by replacing them with the most frequent value of the column
    # 2. Apply one-hot encoding
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # bundle preprocessing for numerical and categorical data:
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )
    return preprocessor


if __name__ == '__main__':

    df = pd.read_csv('Iowa_Housing.csv', index_col='Order')             # (2930, 81)

    X = df.drop(['SalePrice'], axis='columns')                          # (2930, 80)
    y = df.SalePrice                                                    # (2930,)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Select categorical columns with relatively low cardinality (number of unique values in a column)
    categorical_cols = [col for col in X_train.columns
                        if X_train[col].nunique() < 10
                        and X_train[col].dtype == 'object']             # 39

    # Select numerical columns:
    numerical_cols = [col for col in X_train.columns
                      if X_train[col].dtype in ['int64', 'float64']]    # 38

    # Keep only selected columns:
    selected_cols = numerical_cols + categorical_cols                   # 76
    X_train = X_train[selected_cols]
    X_test = X_test[selected_cols]

    preprocessor = apply_column_transform()
    estimator = RandomForestRegressor(n_estimators=200)
    my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('estimator', estimator)
                                  ])

    my_pipeline.fit(X_train, y_train)
    y_pred = my_pipeline.predict(X_test)

    print('Mean Absolute Error: ', mean_absolute_error(y_pred, y_test))
    print('Accuracy: ', my_pipeline.score(X_test, y_test))


# ---------------------------------------------------------------------------------------------------------------------
# if __name__ == '__main__':
#     df = pd.read_csv('email spam.csv')
#     # Convert 'category' column to numeric. 1: Spam, 0: Ham
#     df['spam'] = df['category'].apply(lambda x: 1 if x == 'spam' else 0)
#
#     X = df['message']
#     y = df['spam']
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#
#     # Create a Pipeline of transforms with a final estimator
#     clf = Pipeline([
#         ('vectorizer', CountVectorizer()),                  # transformer
#         ('nb', MultinomialNB())                             # estimator
#     ])
#
#     # We can now train directly on un-vectorized text data, as the pipeline will internally convert the text data
#     # to vector and then apply the Naive Bayes classifier
#     clf.fit(X_train, y_train)
#     print('Accuracy : ', clf.score(X_test, y_test))
#
#     # Predict:
#     emails = [
#         'Hey mohan, can we get together to watch football game tomorrow?',
#         'Upto 20% discount on parking, exclusive offer just for you. Dont miss this reward!'
#     ]
#     print('Prediction: ', clf.predict(emails))

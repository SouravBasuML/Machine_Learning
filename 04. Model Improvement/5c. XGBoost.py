"""
-----------------------------------------------------------------------------------------------------------------------
XGBoost (Extreme Gradient Boosting):
-----------------------------------------------------------------------------------------------------------------------
XGBoost stands for extreme gradient boosting, which is an implementation of gradient boosting with several additional
features focused on performance and speed. (Scikit-learn has another version of gradient boosting, but XGBoost has some
technical advantages.)

n_estimators:
    Specifies how many times to go through the modeling cycle. It is equal to the number of models that we include in
    the ensemble.
    Too low a value causes under fitting; too high a value causes over fitting
    Typical values range from 100-1000, though this depends a lot on the learning_rate parameter.

early_stopping_rounds:
    Offers a way to automatically find the ideal value for n_estimators. Early stopping causes the model to stop
    iterating when the validation score stops improving, even if we aren't at the hard stop for n_estimators. Generally
    a high value for n_estimators is set and early_stopping_rounds is used to find the optimal time to stop iterating.

    Since random chance sometimes causes a single round where validation scores don't improve, you need to specify a
    number for how many rounds of straight deterioration to allow before stopping. Setting early_stopping_rounds=5 is
    a reasonable choice. In this case, we stop after 5 straight rounds of deteriorating validation scores.

eval_set:
    When using early_stopping_rounds, you also need to set aside some data for calculating the validation scores -
    this is done by setting the eval_set parameter. If you later want to fit a model with all of your data, set
    n_estimators to whatever value you found to be optimal when run with early stopping.

learning_rate:
    Instead of getting predictions by simply adding up the predictions from each component model, we can multiply the
    predictions from each model by a small number (known as the learning rate) before adding them in. This means each
    tree we add to the ensemble helps us less. So, we can set a higher value for n_estimators without over fitting.
    If we use early stopping, the appropriate number of trees will be determined automatically.

    In general, a small learning rate and large number of estimators will yield more accurate XGBoost models, though it
    will also take the model longer to train since it does more iterations through the cycle. As default, XGBoost sets
    learning_rate=0.1.

n_jobs:
    On larger datasets where runtime is a consideration, you can use parallelism to build your models faster. It's
    common to set the parameter n_jobs equal to the number of cores on your machine. On smaller datasets, this won't
    help. The resulting model won't be any better, so micro-optimizing for fitting time is typically nothing but a
    distraction. But, it's useful in large datasets where the fit command would otherwise take a long time.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor


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
    y = df.SalePrice  # (2930,)
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

    # Preprocess numerical and categorical columns
    preprocessor = apply_column_transform()
    estimator = XGBRegressor(n_estimators=1000, learning_rate=0.1)
    my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('estimator', estimator)
                                  ])

    my_pipeline.fit(X_train, y_train)
    y_pred = my_pipeline.predict(X_test)

    print('Mean Absolute Error: ', mean_absolute_error(y_pred, y_test))
    print('Accuracy: ', my_pipeline.score(X_test, y_test))

"""
                FEATURE TRANSFORMER         MODEL TRAINING
                -------------------         --------------
TRAIN DATA:     fit_transform()             fit()
TEST DATA:      transform()                 predict()

-----------------------------------------------------------------------------------------------------------------------
fit():
-----------------------------------------------------------------------------------------------------------------------
For Feature Engineering:
    fit() calculates the parameter values for the transform. e.g.
        - for StandardScaler, calculates mu and sigma
        - for SimpleImputer, calculates mean of the column with missing values
        - for PCA, calculates Eigen Vectors etc.

For Model Training:
    fit() learns parameters and weights of the model. e.g.
        - for LinearRegression, learn m and c
    Applied to TRAINING data
-----------------------------------------------------------------------------------------------------------------------


-----------------------------------------------------------------------------------------------------------------------
transform():
-----------------------------------------------------------------------------------------------------------------------
transform() applies the parameters calculated using fit() to transform input features to another format before
model training. e.g.
    - for StandardScaler, transforms the input features by applying mu and sigma calculated by fit()
    - for SimpleImputer, fills in missing values with the mean calculated by fit()
    - for PCA, reduces dimensions by applying Eigen Vectors calculated by fit()
Only transform() is applied to TEST data (not fit_transform()):
    - So that the parameters calculated by fit() using training data is applied to transform the test data.
    - This avoids over-fitting
-----------------------------------------------------------------------------------------------------------------------


-----------------------------------------------------------------------------------------------------------------------
fit_transform():
-----------------------------------------------------------------------------------------------------------------------
Performs both fit() and transform() one after the other
Applied to TRAINING data
-----------------------------------------------------------------------------------------------------------------------


-----------------------------------------------------------------------------------------------------------------------
predict():
-----------------------------------------------------------------------------------------------------------------------
Applies the parameters and weights learned by fit() in model training to new data to predict the outcome
Applied to TEST or NEW data
-----------------------------------------------------------------------------------------------------------------------

"""

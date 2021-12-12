import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso, Ridge, ElasticNet

# ---------------------------------------------------------------------------------------------------------------------
# Ridge Regression (L2 Norm):
# ---------------------------------------------------------------------------------------------------------------------
# Linear least squares with l2 regularization. Ridge regression imposes a penalty on the size of the coefficients.
# The ridge coefficients minimize a penalized residual sum of squares. The complexity parameter alpha (>= 0) controls
# the amount of shrinkage: the larger the value of alpha, the greater the amount of shrinkage and thus the coefficients
# become more robust to collinearity.
# ---------------------------------------------------------------------------------------------------------------------
# An over-fit model has low bias (close to zero Sum of Squared Error (SSE) on training data), but high variance
# (very high SSE on test data). Ridge Regression finds a new line that doesn't fit the training data as well (i.e.
# some bias is introduced), but gives a significant drop in variance (i.e. future predictions will be better).
#
# Ridge regression minimizes:
# 1. Sum of squared residuals
# 2. Alpha * square(all parameters) - all parameters of the hypothesis are included except the y-intercept (or theta0).
#    This adds a penalty and alpha determines the severity of the penalty.
#
# Ridge regression helps reduce variance by shrinking parameters (slope estimate) and making our predictions less
# sensitive to them. The ridge regression line has a smaller slope compared to the regular regression line. This means,
# y is less sensitive to X. As alpha increases, the slope of the ridge regression line reduces and gets asymptotically
# close to 0 (but not equal to 0). This means, y keeps getting less and less sensitive to X. Use K-fold cross
# validation to determine optimal alpha that results in the lowest variance (i.e. best accuracy/score).
#
# Ridge regression is applied where we have few training samples (m) but large number of features (n). By adding the
# ridge regression penalty we can solve for all the n parameters with only m training samples. Ridge regression reduces
# variance and makes the predictions less sensitive to training data.
#
# Ridge regression can also be applied to Logistic Regression and Neural Networks.
# ---------------------------------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------------------------------
# Lasso Regression (L1 Norm):
# ---------------------------------------------------------------------------------------------------------------------
# Linear Model trained with L1 prior as regularizer. The Lasso is a linear model that estimates sparse coefficients.
# ---------------------------------------------------------------------------------------------------------------------
# Lasso regression is very similar to Ridge regression. It minimizes:
# 1. Sum of squared residuals
# 2. Alpha * |all parameters| - all parameters of the hypothesis are included except the y-intercept (or theta0).
#    This adds a penalty and alpha determines the severity of the penalty.
#
# As alpha increases, the slope of the Lasso regression line reduces and can get to 0. This is the main difference
# between Ridge and Lasso regression. This means, lower order polynomial terms in the regression equation will shrink
# a little, but higher order polynomial terms can be shrunk all the way to 0. Lasso regression is little better at
# reducing variance than Ridge regression in models that contain a lot of useless variables. Ridge regression should
# be used where most the variables are meaningful.
#
# Lasso regression can also be applied to Logistic Regression and Neural Networks.
# ---------------------------------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------------------------------
# Elastic-Net Regression (L1 and L2 Norm):
# ---------------------------------------------------------------------------------------------------------------------
# Linear regression with combined L1 and L2 priors as regularizer.
# ---------------------------------------------------------------------------------------------------------------------
# Elastic-Net Regression combines the strengths of Lasso and Ridge regression. It minimizes:
# 1. Sum of squared residuals
# 2. Alpha1 * |all parameters|
# 3. Alpha2 * square(all parameters)
#
# Cross validation on different values of alpha1 and alpha2 are used to identify the optimal values.
# alpha1 = 0, alpha2 = 0    -> Regular Linear Regression (Least Squared)
# alpha1 = 0, alpha2 > 0    -> Ridge Regression
# alpha1 > 0, alpha2 = 0    -> Lasso Regression
# alpha1 > 0, alpha2 > 0    -> Elastic-Net Regression
#
# Elastic-Net regression is used when we do not know if the features have useless variables or not. It is especially
# good in situations when there are correlations between parameters.
# ---------------------------------------------------------------------------------------------------------------------

def linear_regression():
    clf = LinearRegression()
    clf.fit(X_train, y_train)
    print('\nRegular Linear Regression:')
    print('Accuracy on test set:', clf.score(X_test, y_test))
    print('Accuracy on train set:', clf.score(X_train, y_train))


def lasso_regression():
    # L1 regularization
    clf = Lasso(alpha=50, max_iter=100, tol=0.1)
    clf.fit(X_train, y_train)
    print('\nLasso (L1) Linear Regression:')
    print('Accuracy on test set:', clf.score(X_test, y_test))
    print('Accuracy on train set:', clf.score(X_train, y_train))


def ridge_regression():
    # L2 regularization
    clf = Ridge(alpha=50, max_iter=100, tol=0.1)
    clf.fit(X_train, y_train)
    print('\nRidge (L2) Linear Regression:')
    print('Accuracy on test set:', clf.score(X_test, y_test))
    print('Accuracy on train set:', clf.score(X_train, y_train))


def elastic_net_regression():
    # L1 and L2 regularization
    # alpha = 0         -> Regular Linear Regression (Least Squared)
    # l1_ratio = 0      -> Ridge Regression (L2 penalty)
    # 0 < l1_ratio < 1  -> Elastic-Net Regression (L1 and L2 penalty)
    clf = ElasticNet(alpha=1, l1_ratio=0.5, max_iter=100, tol=0.1)
    clf.fit(X_train, y_train)
    print('\nElastic-Net (L1 and L2) Linear Regression:')
    print('Accuracy on test set:', clf.score(X_test, y_test))
    print('Accuracy on train set:', clf.score(X_train, y_train))


if __name__ == '__main__':
    df = pd.read_csv('melbourne_house_price.csv')
    # print(df.isna().sum())

    # Change NaN of certain columns to 0
    cols_to_fill_zero = ['Propertycount', 'Distance', 'Bedroom2', 'Bathroom', 'Car']
    df[cols_to_fill_zero] = df[cols_to_fill_zero].fillna(0)

    # Change NaN of certain columns to the mean of values in those columns
    df['Landsize'] = df['Landsize'].fillna(df['Landsize'].mean())
    df['BuildingArea'] = df['BuildingArea'].fillna(df['BuildingArea'].mean())

    # Drop the remaining rows that have some NaNs in certain columns
    df.dropna(inplace=True)

    # Apply one-hot encoding to text columns to replace them with binary 0 and 1
    df = pd.get_dummies(df, drop_first=True)

    X = df.drop('Price', axis='columns')
    y = df['Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    linear_regression()
    lasso_regression()
    ridge_regression()
    elastic_net_regression()

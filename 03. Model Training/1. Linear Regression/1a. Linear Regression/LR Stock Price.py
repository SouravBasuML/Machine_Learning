import pandas as pd
import math
import datetime
import numpy as np
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

# ---------------------------------------------------------------------------------------------------------------------
# Linear Regression
# ---------------------------------------------------------------------------------------------------------------------
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
# LinearRegression (aka Least Squares) fits a linear model with coefficients w = (w1, …, wp) to minimize the residual
# sum of squares between the observed targets in the dataset, and the targets predicted by the linear approximation.
# ---------------------------------------------------------------------------------------------------------------------
# Uni-variate LR: y = mx + b
# coef_: m (slope)
# intercept_: b (y-intercept)
# ---------------------------------------------------------------------------------------------------------------------
# Multi-variate LR: y = (m1.x1 + m2.x2 + ... + mn.xn) + b
# coef_: m1, m2, ..., mn (slope)
# intercept_: b (y-intercept)
# ---------------------------------------------------------------------------------------------------------------------

# Read google_stock_price csv
df = pd.read_csv('google_stock_price.csv')

# Add new features
df['HL_PCT'] = (df['High'] - df['Low']) / df['Low'] * 100.0             # % volatility for the day
df['PCT_Change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0      # % change for the day

# Replace NaN with -99999 in case there are NaNs in the dataset
df.fillna(-99999, inplace=True)

# Final feature set
df = df[['Close', 'HL_PCT', 'PCT_Change', 'Volume']]

# Define the Label which is same value as 'Close', but pushed by number of days that is 1% of the dataset
forecast_col = 'Close'
forecast_out = int(math.ceil(0.01*len(df)))                     # 13
df['Label'] = df[forecast_col].shift(-forecast_out)


# Define X as numpy array by dropping Label column (axis = 1)
X = np.array(df.drop(['Label'], axis=1))

# Standardize features by removing the mean and scaling to unit variance (each column has mean 0 and STD 1)
# This can be checked by converting X to a dataframe (df2 = pd.DataFrame(X)) and describing it (print(df2.describe()))
X = preprocessing.StandardScaler().fit_transform(X)


# Keep the last 13 rows (1% of the dataset), whose y/Label is undefined, for prediction.
# So X_pred is all rows from forecast_out till end and X is all rows from beginning till forecast_out
X_pred = X[-forecast_out:]
X = X[:-forecast_out]


# Drop rows that have NaN in 'Label' column. Those would be the last 35 odd rows (1% of the dataset)
df.dropna(inplace=True)

# Define y as numpy array
y = np.array(df['Label'])


# Define train and test set
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)


# Linear Regression:
# n_jobs is the number of jobs to use for the computation. Default is 1. -1 means using all processors.
clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)

accuracy_LN = clf.score(X_test, y_test)
forecast_set = clf.predict(X_pred)
print('Forecasted Stock Prices (using Linear Regression): ', forecast_set)
print('Accuracy (Linear Regression): ',  accuracy_LN)


# # Plotting:
# df['Forecast'] = np.nan
# last_date = df.iloc[-1].name                    # get the last date's name (12/30/2016)
# last_unix = last_date.timestamp()               # changes this date text into unix timestamp format
# one_day = 86400
# next_unix = last_unix + one_day
#
# for i in forecast_set:
#     next_date = datetime.datetime.fromtimestamp(next_unix)
#     next_unix += one_day
#     df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]
#
# df['Close'].plot()
# df['Forecast'].plot()
# plt.legend(loc=4)
# plt.xlabel('Date')
# plt.ylabel('Price')
# plt.show()


# # SVM:
# clf = svm.SVR(kernel='linear')
# clf.fit(X_train, y_train)
# accuracy_SVM = clf.score(X_test, y_test)
# forecast_set = clf.predict(X_pred)
# print('Forecasted Stock Prices (using SVM): ', forecast_set)
# print('Accuracy (SVM): ',  accuracy_SVM)

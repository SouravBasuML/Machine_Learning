import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_digits
from sklearn import model_selection

# ---------------------------------------------------------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------------------------------------------------------
# Random Forest is a collection of Decision Trees. Divide your feature set into batch of random datasets, and
# apply Decision Tree algorithm to each. To get prediction, take a majority vote from every decision tree outcome.
# Used for both Regression and Classification
# ---------------------------------------------------------------------------------------------------------------------
# Decision Tres suffer from inaccuracy. They work great with the data used to create them, but are not flexible with
# classifying new samples. Random Forests combine the simplicity of DT with flexibility, improving accuracy vastly.
# Steps to create a Random Forest:
# 1. Create a bootstrapped dataset from your original dataset (This is done by selecting random samples from the
#    original dataset, and creating a new dataset of same size. Same samples can be picked in the new dataset.)
# 2. Create a decision tree using the boot strapped dataset, but only use a random subset of variables (or columns)
#    at each step (level of the DT).
# 3. Repeat Steps 1 and 2 multiple times; i.e create a new bootstrapped dataset, and build a DT considering a subset
#    of variables at each step. This wide variety of decision trees creates a Random Forest and this variety makes the
#    Random Forest more effective than individual DTs.
# 4. To make a prediction, we run the new sample on every decision tree in the Random Forest and take the majority
#    vote of the decisions from the individual DTs. Bootstrapping the dataset plus using the aggregate to make a
#    decision is called Bagging.
# 5. Accuracy is measured by taking the samples that are not included in the bootstrapped dataset (typically about
#    one-third of the dataset) and using them as test data. These samples are called Out-Of-Bag dataset. The Out-Of-Bag
#    samples are run on all DTs that were created using data that did not include the Out-Of-Bag samples, and a
#    majority vote is used for prediction. The proportion of Out-Of-Bag samples that were incorrectly classified is
#    called the Out-Of-Bag Error.
# 6. All the above steps are repeated by creating DTs at Step 2, using a different random number of columns and
#    accuracy is measured. This is again repeated to get the best accuracy. We typically start with the square root of
#    the number of columns and then try a few settings above and below the sqrt value.
# ---------------------------------------------------------------------------------------------------------------------
# Missing Data:
# Missing data is filled initially with a random guess (e.g. most common value, mean of numeric values etc. for similar
# samples) and then refining it iteratively using Random Forest until the missing values converge (i.e. they no longer
# change each time we recalculate).
# ---------------------------------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------------------------------
# Hand-Written Digit Classification:
# ---------------------------------------------------------------------------------------------------------------------
# sklearn.datasets.load_digits - Each datapoint is a 8x8 image of a digit
# https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html#sklearn.datasets.load_digits
# Classes           : 10
# Samples per class : ~180
# Samples total     : 1797
# Dimensionality    : 64
# Features          : integers 0-16 (gray scale value from 0 to 16)
# ---------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    digits = load_digits()
    print(dir(digits))          # ['DESCR', 'data', 'feature_names', 'frame', 'images', 'target', 'target_names']

    df = pd.DataFrame(digits.data)
    df['label'] = digits.target

    X = df.drop(['label'], axis='columns')
    y = df['label']
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

    clf = RandomForestClassifier(n_estimators=25)       # n_estimators= number of trees, 10 is default
    clf.fit(X_train, y_train)

    print('Accuracy: ', clf.score(X_test, y_test))

"""
-----------------------------------------------------------------------------------------------------------------------
Bagging (Bootstrap Aggregation):
-----------------------------------------------------------------------------------------------------------------------
Bootstrap aggregating, also called bagging, is a machine learning ensemble meta-algorithm designed to improve the
stability and accuracy of machine learning algorithms used in statistical classification and regression
-----------------------------------------------------------------------------------------------------------------------
1. The feature set is bootstrapped; i.e., multiple new datasets are created by taking a sample of records from the
   original dataset (duplicates are allowed). This technique is also called 'Resampling with Replacement'
2. These bootstrapped datasets are run through multiple machine learning models (estimators), e.g. multiple DTs.
   That means, every estimator (DT in this case) is trained on a different sample of training data. These individual
   estimators are called 'Weak Learners', which means they will not over fit.
3. To make a prediction, we run the new sample on every estimator and aggregate the results:
   a. For classification: take the majority vote of the decisions from the individual estimators.
   b. For regression: take the mean of the outputs from each estimator.
4. Accuracy is measured by taking the samples that are not included in any of the bootstrapped datasets (typically
   about one-third of the dataset) and using them as test data. These samples are called Out-Of-Bag samples. The
   Out-Of-Bag samples are run on all the estimators, and a majority vote is used for prediction. The proportion of
   Out-Of-Bag samples that were incorrectly classified is called the Out-Of-Bag Error.

Bootstrapping the dataset plus using the aggregate to make a decision is called Bagging. Bagging reduces over and
under fitting (high variance and bias).

Random Forest uses Bagging which runs a bootstrapped dataset through an ensemble of Decision Trees, and then
aggregates the decisions and takes a majority vote. Along with 'Row Sampling with Replacement', Random Forest also
uses 'Column/Feature Sampling with Replacement'.
-----------------------------------------------------------------------------------------------------------------------
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier


if __name__ == '__main__':

    df = pd.read_csv('pima_indian_diabetes.csv')                        # (768, 9)
    X = df.drop('Outcome', axis='columns')
    y = df.Outcome
    # stratify=True: To get train and test samples in equal proportion of original dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 5-fold cross validation
    scores = cross_val_score(DecisionTreeClassifier(), X, y, cv=5)
    print(scores)
    print('Accuracy without Bagging (using cross-validation): ', scores.mean())

    # Bagging
    bagged_clf = BaggingClassifier(
        base_estimator=DecisionTreeClassifier(),      # Default is DecisionTreeClassifier
        n_estimators=100,                             # The number of base estimators in the ensemble
        max_samples=0.8,                              # Number of samples to draw from X to train each base estimator
        max_features=1.0,                             # Number of features to draw from X to train each base estimator
        oob_score=True                                # Use out-of-bag samples to estimate the generalization error
    )

    bagged_clf.fit(X_train, y_train)
    # oob_score_ is calculated on the out-of-bag samples of the training data
    print('Accuracy with Bagging (on OOB Samples): ', bagged_clf.oob_score_)
    print('Accuracy with Bagging (on test samples): ', bagged_clf.score(X_test, y_test))

    # Bagging + Cross-Validation: (we don't need X_train, y_train etc.)
    scores = cross_val_score(bagged_clf, X, y, cv=5)
    print('Accuracy with Bagging (using cross-validation): ', scores.mean())

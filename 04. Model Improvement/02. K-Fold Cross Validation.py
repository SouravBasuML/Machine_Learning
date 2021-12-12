"""
-----------------------------------------------------------------------------------------------------------------------
K-Fold Cross Validation:
-----------------------------------------------------------------------------------------------------------------------
Provides train/test indices to split data in train/test sets. Split dataset into k consecutive folds
(without shuffling by default). Default folds: k=5. Each fold is then used once as a validation (test set)
while the k - 1 remaining folds form the training set. The scores from the k test sets are averaged to get the
final score (accuracy). This allows the model to be trained and tested on all samples.

In extreme case, we may consider each sample in the training data a 'fold'. Thus, this will be n=fold cross
validation. This is also called "Leave One Out Cross Validation"

When to use cross-validation?
For small datasets, where extra computational burden isn't a big deal, you should run cross-validation.
For larger datasets, a single validation set is sufficient. Your code will run faster, and you may have enough data
that there's little need to re-use some of it for holdout.
-----------------------------------------------------------------------------------------------------------------------

-----------------------------------------------------------------------------------------------------------------------
Stratified K-Fold Cross Validation:
-----------------------------------------------------------------------------------------------------------------------
Provides train/test indices to split data in train/test sets. This cross-validation object is a variation of KFold
that returns stratified folds. The folds are made by preserving the percentage of samples for each class.
-----------------------------------------------------------------------------------------------------------------------

-----------------------------------------------------------------------------------------------------------------------
sklearn.model_selection.cross_val_score:
-----------------------------------------------------------------------------------------------------------------------
Evaluate a score by cross-validation. This method creates k folds of the data and calculates scores for the
classifier passed. By default 5 folds are used (cv=5)
Here we directly feed X and y to cross_val_score without using train_test_split. This also takes care of
'Train-test Contamination' type of Data Leakage.
-----------------------------------------------------------------------------------------------------------------------
"""

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn import model_selection
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score


def get_score(clf, X_train, X_test, y_train, y_test):
    clf.fit(X_train, y_train)
    return clf.score(X_test, y_test)


def accuracy_legacy():
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

    print('Accuracy (LR) : ', get_score(LogisticRegression(), X_train, y_train, X_test, y_test))
    print('Accuracy (SVM): ', get_score(SVC(), X_train, y_train, X_test, y_test))
    print('Accuracy (KNN): ', get_score(KNeighborsClassifier(), X_train, y_train, X_test, y_test))
    print('Accuracy (DT) : ', get_score(DecisionTreeClassifier(), X_train, y_train, X_test, y_test))
    print('Accuracy (RF) : ', get_score(RandomForestClassifier(), X_train, y_train, X_test, y_test))


def accuracy_kfold_manual():

    # kf = KFold(n_splits=3)
    # for train_index, test_index in kf.split([1, 2, 3, 4, 5, 6, 7, 8, 9]):
    #     print("TRAIN:", train_index, "TEST:", test_index)

    skfolds = StratifiedKFold(n_splits=3)
    Accuracy_LR = []
    Accuracy_SVM = []
    Accuracy_KNN = []
    Accuracy_DT = []
    Accuracy_RF = []

    # This for loop will repeat n_splits times
    for train_index, test_index in skfolds.split(X, y):
        X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]

        Accuracy_LR.append(get_score(LogisticRegression(), X_train, X_test, y_train, y_test))
        Accuracy_SVM.append(get_score(SVC(), X_train, X_test, y_train, y_test))
        Accuracy_KNN.append(get_score(KNeighborsClassifier(), X_train, X_test, y_train, y_test))
        Accuracy_DT.append(get_score(DecisionTreeClassifier(), X_train, X_test, y_train, y_test))
        Accuracy_RF.append(get_score(RandomForestClassifier(), X_train, X_test, y_train, y_test))

    print('Accuracy (LR) : ', Accuracy_LR)
    print('Accuracy (SVM): ', Accuracy_SVM)
    print('Accuracy (KNN): ', Accuracy_KNN)
    print('Accuracy (DT) : ', Accuracy_DT)
    print('Accuracy (RF) : ', Accuracy_RF)


def accuracy_sklearn_cross_val_score():

    # Instead of giving the estimator directly, we can give a pipeline with multiple transforms and the estimator
    # The following code will give a list of 'cv' (e.g. 5) accuracy scores. We can take their mean.
    # Here we are directly feeding X and y to cross_val_score; no need to use train_test_split. This also takes care of
    # Train-test Contamination type of Data Leakage
    print('Accuracy (LR) : ', cross_val_score(LogisticRegression(), X, y, cv=3))
    print('Accuracy (SVM): ', cross_val_score(SVC(), X, y, cv=4))
    print('Accuracy (KNN): ', cross_val_score(KNeighborsClassifier(), X, y, cv=5))
    print('Accuracy (DT) : ', cross_val_score(DecisionTreeClassifier(), X, y, cv=5))
    print('Accuracy (RF) : ', cross_val_score(RandomForestClassifier(), X, y, cv=5))


if __name__ == '__main__':
    digits = load_digits()
    X = digits.data
    y = digits.target

    # accuracy_legacy()
    # accuracy_kfold_manual()
    accuracy_sklearn_cross_val_score()

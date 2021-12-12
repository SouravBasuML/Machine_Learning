"""
------------------------------------------------------------------------------------------------------------------------
Hyper-Parameter Tuning:
------------------------------------------------------------------------------------------------------------------------
Tuning the parameters of the ML algorithms to choose the best parameters for an algorithm (i.e. to get the best
accuracy/score for an ML algorithm).
Hyper-parameters are parameters that are not directly learnt within estimators. In scikit-learn they are passed as
arguments to the constructor of the estimator classes. Typical examples include C, kernel and gamma for Support
Vector Classifier, alpha for Lasso, etc. Two generic approaches to parameter search are provided in scikit-learn:
https://scikit-learn.org/stable/modules/grid_search.html
-----------------------------------------------------------------------------------------------------------------------

-----------------------------------------------------------------------------------------------------------------------
GridSearchCV:
-----------------------------------------------------------------------------------------------------------------------
For given values, GridSearchCV exhaustively considers all parameter combinations. It uses K-fold cross validation.
GridSearchCV is computationally expensive as it trains the model on all combinations of parameter lists.
-----------------------------------------------------------------------------------------------------------------------

-----------------------------------------------------------------------------------------------------------------------
RandomizedSearchCV:
-----------------------------------------------------------------------------------------------------------------------
RandomizedSearchCV can sample a given number of candidates from a parameter space with a specified distribution.
It uses K-fold cross validation.
-----------------------------------------------------------------------------------------------------------------------

Note:
    Here we directly feed X and y to GridSearchCV or RandomizedSearchCV without using train_test_split. This also
    takes care of 'Train-test Contamination' type of Data Leakage.

"""

import pandas as pd
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


def grid_search_cv():
    print('\nGridSearchCV:')
    clf = GridSearchCV(SVC(gamma='auto'), {
        'C': [1, 10, 20],
        'kernel': ['rbf', 'linear']
    },
                       cv=5,                                # 5-fold cross-validation
                       return_train_score=False)

    clf.fit(X, y)

    # cv_results_ gives detailed results of GridSearchCV, but not easy to read
    # print(clf.cv_results_)

    # Download the results into a Dataframe:
    df = pd.DataFrame(clf.cv_results_)
    print(df[['param_C', 'param_kernel', 'mean_test_score']])

    # print(dir(clf))
    print('Best Score: ', clf.best_score_)
    print('Best Estimator: ', clf.best_estimator_)
    print('Best Parameters: ', clf.best_params_)


def randomized_search_cv():
    print('\nRandomizedSearchCV:')
    clf = RandomizedSearchCV(SVC(gamma='auto'), {
        'C': [1, 10, 20],
        'kernel': ['rbf', 'linear']
    },
                             cv=5,                          # 5-fold cross-validation
                             return_train_score=False,
                             n_iter=3)                      # Number of parameter settings that are sampled

    clf.fit(X, y)

    # cv_results_ gives detailed results of GridSearchCV, but not easy to read
    # print(clf.cv_results_)

    # Download the results into a Dataframe:
    df = pd.DataFrame(clf.cv_results_)
    print(df[['param_C', 'param_kernel', 'mean_test_score']])

    # print(dir(clf))
    print('Best Score: ', clf.best_score_)
    print('Best Estimator: ', clf.best_estimator_)
    print('Best Parameters: ', clf.best_params_)


def choose_best_model():
    print('\nChoosing Best Model:')
    # Define models and parameters as a Python dictionary
    model_params = {
        'logistic_regression': {
            'model': LogisticRegression(solver='liblinear', multi_class='auto'),
            'params': {
                'C': [1, 5, 10]
            }
        },
        'svm': {
            'model': SVC(gamma='auto'),
            'params': {
                'C': [1, 10, 20],
                'kernel': ['rbf', 'linear']
            }
        },
        'random_forest': {
            'model': RandomForestClassifier(),
            'params': {
                'n_estimators': [1, 5, 10]
            }
        }
    }

    scores = []
    for model_name, model_parameters in model_params.items():
        clf = GridSearchCV(model_parameters['model'],
                           model_parameters['params'],
                           cv=5,
                           return_train_score=False)
        clf.fit(X, y)
        scores.append({
            'model': model_name,
            'best_score': clf.best_score_,
            'best_parameters': clf.best_params_
        })

    df = pd.DataFrame(scores, columns=['model', 'best_score', 'best_parameters'])
    print(df)


if __name__ == '__main__':
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    grid_search_cv()
    randomized_search_cv()
    choose_best_model()

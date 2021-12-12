"""
-----------------------------------------------------------------------------------------------------------------------
Confusion Matrix:
-----------------------------------------------------------------------------------------------------------------------
A confusion matrix is a n x n matrix, such that each element is equal to the number of observations known to be in
group i and predicted to be in group j. The size of the confusion matrix is decided by the number of classes (n)
we need to predict.

True Positive: You predicted positive and it’s true.
               e.g. patients have cancer and algorithm predicted that they have cancer
True Negative: You predicted negative and it’s true.
               e.g. patients don't have cancer and algorithm predicted that they don't have cancer
False Positive: (Type 1 Error): You predicted positive and it’s false.
               e.g. patients don't have cancer, but algorithm predicted that they have cancer
False Negative: (Type 2 Error): You predicted negative and it’s false.
               e.g. patients have cancer, but algorithm predicted that they don't have cancer
-----------------------------------------------------------------------------------------------------------------------

-----------------------------------------------------------------------------------------------------------------------
Sensitivity/Recall (True Positive Rate):
-----------------------------------------------------------------------------------------------------------------------
When it's actually yes, how often it predicts yes? -> TP/(TP+FN)
e.g. what % of people with cancer were correctly identified by the algorithm
Choose an algorithm with higher Sensitivity if correctly identifying positives is more important
-----------------------------------------------------------------------------------------------------------------------

-----------------------------------------------------------------------------------------------------------------------
Specificity (True Negative Rate):
-----------------------------------------------------------------------------------------------------------------------
When it's actually no, how often does it predict no? -> TN/(FP+TN)
e.g. what % of people without cancer were correctly identified by the algorithm
Choose an algorithm with higher Specificity if correctly identifying negatives is more important
For larger matrices, we need to calculate Sensitivity and Specificity for each output class
-----------------------------------------------------------------------------------------------------------------------

-----------------------------------------------------------------------------------------------------------------------
Precision:
-----------------------------------------------------------------------------------------------------------------------
When it predicts yes, how often is it actually yes? -> TP/(TP+FP)
Proportion of positive results that were correctly classified.
-----------------------------------------------------------------------------------------------------------------------

-----------------------------------------------------------------------------------------------------------------------
Accuracy: Overall, how often is the classifier correct? -> (TP+TN)/total
Mis-classification Rate: Overall, how often is it wrong? -> (FP+FN)/total
F-Score: (2 * Precision * Recall) / (Precision + Recall)
-----------------------------------------------------------------------------------------------------------------------

-----------------------------------------------------------------------------------------------------------------------
ROC (Receiver Operator Characteristic):
-----------------------------------------------------------------------------------------------------------------------
ROC graphs and AUC are useful for consolidating information from various confusion matrices into a single,
easy to interpret graph.
ROC graph has True Positive Rate (Sensitivity) on the y-axis and False Positive Rate (1 - Specificity) on the x-axis.
Some ROC graphs may have Precision on the x-axis. ROC graph is plotted for different thresholds selected for
Logistic Regression (e.g.). The graph then helps us choose the optimal threshold based on True Positive and
False Positive rate plots.
-----------------------------------------------------------------------------------------------------------------------

-----------------------------------------------------------------------------------------------------------------------
AUC (Area Under the Curve):
-----------------------------------------------------------------------------------------------------------------------
AUC helps us compare different ROC curves plotted for different algorithms. The algorithm that gives the largest area
under the ROC curve is better.
-----------------------------------------------------------------------------------------------------------------------

"""

import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sn


if __name__ == '__main__':
    digits = load_digits()

    X_train, X_test, y_train, y_test = model_selection.train_test_split(digits.data, digits.target, test_size=0.2)
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    print('Accuracy: ', clf.score(X_test, y_test))

    # Identify for which samples, the model is predicting incorrectly using a Confusion Matrix
    y_predicted = clf.predict(X_test)
    label = ['digit 0', 'digit 1', 'digit 2', 'digit 3', 'digit 4',
             'digit 5', 'digit 6', 'digit 7', 'digit 9', 'digit 9']

    conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_predicted)
    print('\nConfusion Matrix:\n', conf_matrix)
    print('\nClassification Report:\n', classification_report(y_true=y_test, y_pred=y_predicted, target_names=label))

    plt.figure(figsize=(10, 7))
    sn.heatmap(conf_matrix, annot=True)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

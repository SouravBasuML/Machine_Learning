import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix
import seaborn as sn


# ---------------------------------------------------------------------------------------------------------------------
# Logistic Regression:
# ---------------------------------------------------------------------------------------------------------------------
# Uses Sigmoid or Logit function, which converts inputs into a range between 0 and 1
# Feed y = mx + b into Sigmoid function's 'z'.
# ---------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------------------------------
# Multiclass Classification:
# ---------------------------------------------------------------------------------------------------------------------
# sklearn.datasets.load_digits - Each datapoint is a 8x8 image of a digit
# ---------------------------------------------------------------------------------------------------------------------
# https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html#sklearn.datasets.load_digits
# Classes           : 10
# Samples per class : ~180
# Samples total     : 1797
# Dimensionality    : 64
# Features          : integers 0-16 (gray scale value from 0 to 16)
# ---------------------------------------------------------------------------------------------------------------------

def digits_analysis():
    print(digits.data.shape)          # (1797, 64)
    print(dir(digits))                # ['DESCR', 'data', 'feature_names', 'frame', 'images', 'target', 'target_names']
    print(digits.data[1])             # List of 64 elements, for an 8x8 image
    print(digits.target[0: 5])        # [0 1 2 3 4]
    print(digits.target_names[0: 5])  # [0 1 2 3 4]


def digits_plot():
    plt.gray()
    for i in range(5):
        plt.matshow(digits.images[i])
    plt.show()


def plot_confusion_matrix():
    y_predicted = clf.predict(X_test)
    conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_predicted)
    print(conf_matrix)

    plt.figure(figsize=(10, 7))
    sn.heatmap(conf_matrix, annot=True)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


if __name__ == '__main__':
    digits = load_digits()
    digits_analysis()
    # digits_plot()

    X_train, X_test, y_train, y_test = model_selection.train_test_split(digits.data, digits.target, test_size=0.2)
    clf = LogisticRegression()
    clf.fit(X_train, y_train)

    print(digits.target[67])                            # 6
    print(clf.predict([digits.data[67]]))               # [6]
    print(clf.predict(digits.data[0:5]))                # [0, 1, 2, 3, 4]
    print('Accuracy: ', clf.score(X_test, y_test))

    # Identify for which samples, the model is predicting incorrectly
    plot_confusion_matrix()

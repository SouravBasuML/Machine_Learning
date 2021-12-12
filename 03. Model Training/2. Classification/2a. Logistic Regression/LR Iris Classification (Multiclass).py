import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')


# ---------------------------------------------------------------------------------------------------------------------
# Iris Dataset:
# ---------------------------------------------------------------------------------------------------------------------
# from sklearn.datasets import load_iris
# ---------------------------------------------------------------------------------------------------------------------
# Classes: 3
# Samples per class: 50
# Samples total: 150
# Dimensionality: 4
# Features: real, positive
# ---------------------------------------------------------------------------------------------------------------------
# Classes:
# 1. Setosa
# 2. Versicolour
# 3. Virginica
# ---------------------------------------------------------------------------------------------------------------------
# Features:
# 1. Sepal Length
# 2. Sepal Width
# 3. Petal Length
# 4. Petal Width
# ---------------------------------------------------------------------------------------------------------------------

def iris_data_analysis():
    print(dir(iris))
    # ['DESCR', 'data', 'data_module', 'feature_names', 'filename', 'frame', 'target', 'target_names']

    print(iris.feature_names)   # ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    print(iris.data.shape)      # (150, 4)
    print(iris.target_names)    # ['setosa' 'versicolor' 'virginica']
    print(iris.data[75])        # [5.1 3.5 1.4 0.2]
    print(iris.target[75])      # 0


def plot_data():
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    # df['iris_type'] = df['label'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    df['iris_type'] = df['label'].apply(lambda x: iris.target_names[x])

    df_setosa = df[df['label'] == 0]
    df_versicolor = df[df['label'] == 1]
    df_virginica = df[df['label'] == 2]

    plt.scatter(df_setosa['sepal length (cm)'], df_setosa['sepal width (cm)'], color='green', marker='+')
    plt.scatter(df_versicolor['sepal length (cm)'], df_versicolor['sepal width (cm)'], color='red', marker='+')
    plt.scatter(df_virginica['sepal length (cm)'], df_virginica['sepal width (cm)'], color='blue', marker='.')
    plt.xlabel('sepal length (cm)')
    plt.ylabel('sepal width (cm)')
    plt.show()

    plt.scatter(df_setosa['petal length (cm)'], df_setosa['petal width (cm)'], color='green', marker='+')
    plt.scatter(df_versicolor['petal length (cm)'], df_versicolor['petal width (cm)'], color='red', marker='+')
    plt.scatter(df_virginica['petal length (cm)'], df_virginica['petal width (cm)'], color='blue', marker='.')
    plt.xlabel('petal length (cm)')
    plt.ylabel('petal width (cm)')
    plt.show()


if __name__ == '__main__':
    iris = load_iris()
    # iris_data_analysis()
    plot_data()

    # X = iris.data
    # y = iris.target
    # X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
    #
    # clf = LogisticRegression()
    # clf.fit(X_train, y_train)
    #
    # print(clf.predict([[5.1, 3.5, 1.4, 0.2]]))                  # [0]
    # print(clf.predict([[6.6, 3.0, 4.4, 1.4]]))                  # [1]
    # print('Accuracy: ', clf.score(X=X_test, y=y_test))

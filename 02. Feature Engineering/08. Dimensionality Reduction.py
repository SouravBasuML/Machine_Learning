"""
-----------------------------------------------------------------------------------------------------------------------
The Curse of Dimensionality:
-----------------------------------------------------------------------------------------------------------------------
The curse of dimensionality refers to various phenomena that arise when analyzing and organizing data in
high-dimensional spaces that do not occur in low-dimensional settings such as the three-dimensional physical space of
everyday experience.

Some difficulties that come with high dimensional data manifest during analyzing or visualizing the data to
identify patterns, and some manifest while training machine learning models. The difficulties related to training
machine learning models due to high dimensional data is referred to as the ‘Curse of Dimensionality’.
-----------------------------------------------------------------------------------------------------------------------

-----------------------------------------------------------------------------------------------------------------------
Principal Component Analysis:
-----------------------------------------------------------------------------------------------------------------------
Linear dimensionality reduction using Singular Value Decomposition of the data to project it to a lower dimensional
space. The input data is centered but not scaled for each feature before applying the SVD.

Used for Dimensionality Reduction, while retaining a high percentage of variation (information) from the original
features. Other methods for Dimensionality Reduction are Heatmaps, t-SNE Plots, and Multi-Dimensional Scaling (MDS)

The features should be on similar scale, else PCA will be biased towards the features on the larger scale. The
features should also be centered (i.e. the mean of each feature should be 0). Make sure the library you are using is
centering the data before applying PCA. The above two steps can be performed using SciKit Learn's:
preprocessing.StandardScaler().fit_transform(X)

Technically there is a Principal Component for each feature in the data set. However, if there are fewer samples in
the dataset, than there are features, then the number of Principal Components will be equal to the number of samples.
The remaining PCs will have Eigenvalues = 0 and hence can be ignored.

A PCA Plot converts the correlations (or lack there of) among all the features into a 2D graph. Features that are
highly correlated cluster together. The axes PC1 and PC2 are ranked in order of importance. Differences along PC1
are more important than differences along PC2.

A Scree Plot is a bar graph that shows which Principal Components contain what percentage of information from the
features. Each vertical bar is a PC and the total of all bars = 100%
-----------------------------------------------------------------------------------------------------------------------
"""

import pandas as pd
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA


if __name__ == '__main__':
    digits = load_digits()
    print(digits.keys())
    X = pd.DataFrame(digits.data, columns=digits.feature_names)
    X = StandardScaler().fit_transform(X)
    y = digits.target

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    print('Accuracy before PCA: ', clf.score(X_test, y_test))

    # In n_components you can supply the number of principal components as an integer or you can specify the
    # percentage of variation that you want retained.
    pca = PCA(n_components=0.90)
    print(X.shape)
    X_pca = pca.fit_transform(X)
    print(X_pca.shape)

    print(pca.n_components_)                # number of principal components
    print(pca.explained_variance_)          # amount of variation that each Principal Component provides.
    print(pca.explained_variance_ratio_)    # percentage of variation that each Principal Component accounts for.

    X_train_pca, X_test_pca, y_train, y_test = model_selection.train_test_split(X_pca, y, test_size=0.2)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_pca, y_train)
    print('Accuracy after PCA: ', clf.score(X_test_pca, y_test))

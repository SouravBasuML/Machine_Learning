"""
-----------------------------------------------------------------------------------------------------------------------
SCALING AND NORMALIZATION:
-----------------------------------------------------------------------------------------------------------------------

Scaling vs. Normalization:
    Scaling:
        The 'range' of the data is changed by transforming the data so that it fits within a specific scale, e.g., 0-100
        or 0-1. You want to scale data when you're using methods based on measures of how far apart data points are,
        like SVM or KNN. With these algorithms, a change of "1" in any numeric feature is given the same importance.
        e.g. 1 unit change in height is given the same importance as 1 unit change in weight. By scaling your variables,
        you can help compare different variables on equal footing.

    Normalization:
        The 'shape' of the distribution of the data is changed such that the data follows a normal distribution. In
        general, you'll normalize your data if you're going to be using a machine learning or statistics technique that
        assumes your data is normally distributed e.g., linear discriminant analysis (LDA) and Gaussian naive Bayes.

Algorithms that require Feature Scaling?
    a. Gradient Descent Based Algorithms:
        Algorithms like linear regression, logistic regression, neural network, etc. that use gradient descent as an
        optimization technique require data to be scaled. The presence of feature value X in the gradient descent
        formula affects the step size of the gradient descent. The difference in ranges of features will cause
        different step sizes for each feature. To ensure that gradient descent moves smoothly towards the minima and
        that the steps for gradient descent are updated at the same rate for all the features, we scale the data before
        feeding it to the model. Having features on a similar scale can help the gradient descent converge more quickly
        towards the minima.

    b. Distance-Based Algorithms
        Distance algorithms like KNN, K-means, and SVM are most affected by the range of features. This is because
        behind the scenes they are using distances between data points (e.g. euclidean) to determine their similarity.


Algorithms that do not require Feature Scaling?
        Algorithms using Ensemble techniques e.g. Decision Tree, Random Forest, XGBoost. e.g. decision tree is only
        splitting a node based on a single feature. It splits a node on a feature that increases the homogeneity of the
        node. This split on a feature is not influenced by other features.

Types:
    1. Standardization:
        It is a scaling technique where the values are centered around the mean with a unit standard deviation. This
        means that the mean of the attribute becomes zero and the resultant distribution has a unit standard deviation.
        Standardization can be helpful in cases where the data follows a Gaussian distribution. However, this does not
        have to be necessarily true. Also, unlike normalization, standardization does not have a bounding range. So,
        even if you have outliers in your data, they will not be affected by standardization.

                Z = (x - μ)/σ

        a. Standard Scaler
        b. Robust Scaler

    2. Normalization:
        It is a scaling technique in which values are shifted and rescaled so that they end up ranging between 0 and 1.
        Normalization is good to use when you know that the distribution of your data does not follow a Gaussian
        distribution. This can be useful in algorithms that do not assume any distribution of the data like K-Nearest
        Neighbors and Neural Networks.

                X_scaled - (X - X.min) / (X.max - X.min)

        a. Min-Max Scaler
        b. Normalizer
-----------------------------------------------------------------------------------------------------------------------


-----------------------------------------------------------------------------------------------------------------------
1a. Standard Scaler: (Standardization)
-----------------------------------------------------------------------------------------------------------------------
from sklearn.preprocessing import StandardScaler
-----------------------------------------------------------------------------------------------------------------------
Standardize features by removing the mean and scaling to unit variance.
The standard score of a sample x is calculated as: Z = (x - μ)/σ
Centering and scaling happen independently on each feature. Standardization of a dataset is a common requirement for
many machine learning estimators: they might behave badly if the individual features do not more or less look like
standard normally distributed data (e.g. Gaussian with 0 mean and unit variance). If a feature has a variance that
is orders of magnitude larger that others, it might dominate the objective function and make the estimator unable
to learn from other features correctly as expected.
-----------------------------------------------------------------------------------------------------------------------

-----------------------------------------------------------------------------------------------------------------------
1b. Robust Scaler: (Standardization)
-----------------------------------------------------------------------------------------------------------------------
from sklearn.preprocessing import RobustScaler
-----------------------------------------------------------------------------------------------------------------------
Scale features using statistics that are robust to outliers. This Scaler removes the median and scales the data
according to the quantile range (defaults to IQR: Inter-quartile Range). The IQR is the range between the 1st quartile
(25th quantile) and the 3rd quartile (75th quantile).
    X_scaled = (X - X.median)/IQR

Centering and scaling happen independently on each feature by computing the relevant statistics on the samples in the
training set. Median and inter-quartile range are then stored to be used on later data using the transform method.

Standardization of a dataset is a common requirement for many machine learning estimators. Typically this is done by
removing the mean and scaling to unit variance. However, outliers can often influence the sample mean / variance in a
negative way. In such cases, the median and the inter-quartile range often give better results.
-----------------------------------------------------------------------------------------------------------------------


-----------------------------------------------------------------------------------------------------------------------
2a. Min-Max Scaler: (Normalization)
-----------------------------------------------------------------------------------------------------------------------
from sklearn.preprocessing import MinMaxScaler
-----------------------------------------------------------------------------------------------------------------------
Transform features by scaling each feature to a given range. This estimator scales and translates each feature
individually such that it is in the given range on the training set, e.g. between zero and one.
    X_scaled - (X - X.min) / (X.max - X.min)
Used mostly in deep learning techniques e.g. CNN where every pixel value is scaled to a value between 0 and 1.
-----------------------------------------------------------------------------------------------------------------------
"""

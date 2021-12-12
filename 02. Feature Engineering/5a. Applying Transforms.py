"""
-----------------------------------------------------------------------------------------------------------------------
Applying Transforms:
-----------------------------------------------------------------------------------------------------------------------
For a feature to be useful, it must have a relationship to the target that our model is able to learn. Linear models,
for instance, are only able to learn linear relationships. So, when using a linear model, our goal is to transform the
features to make their relationship to the target linear. The key idea here is that a transformation we apply to a
feature becomes in essence a part of the model itself. Whatever relationships our model can't learn, we can provide
ourselves through transformations.

Some transforms we can apply to transform an existing feature or create additional features. These transforms can be
applied using pandas (see 'EDA using Pandas.py' for examples).

a. Mathematical Transforms:
        Relationships among numerical features are often expressed through mathematical formulas, which you'll
        frequently come across as part of your domain research.
        e.g. Ratios, mathematical formulae etc.

b. Gaussian Transforms (Normalization):
        Transforming a feature with skewed distribution to normal distribution.
        e.g. Log/Exp/BoxCox transforms etc.

c. Aggregating Transforms:
        e.g. pandas count(), sum(), min(), max(), avg()

d. Group Transforms:
        Using an aggregation function, a group transform combines two features: a categorical feature that provides the
        grouping and another feature whose values you wish to aggregate
        e.g. applying aggregate functions like count, mean, max, min, median, var, std using pandas groupby()

e. String Transforms:
        Complex strings like phone number, address, urls, IDs, Date, Time etc. can sometimes usefully be broken into
        simpler pieces. On the contrary, text features may be combined to create a new meaningful feature.
        e.g. pandas split()


-----------------------------------------------------------------------------------------------------------------------
Gaussian Transformation (Normalization):
-----------------------------------------------------------------------------------------------------------------------
ML algorithms like linear and logistic regression etc. expect the features to be normally distributed. We may need to
transform features to Gaussian distribution for better prediction. Some transforms we can apply are:

    Logarithmic Transformation
    Reciprocal Transformation
    Square root Transformation
    Exponential Transformation
    Box-Cox Transformation
        The Box-Cox transformation is defined as: T(Y)=(Y exp(λ)−1)/λ
        where Y is the response variable and λ is the transformation parameter. λ varies from -5 to 5.
        In the transformation, all values of λ are considered and the optimal value for a given variable is selected.

Q-Q Plot:
    Used to identify if a feature is normally distributed.
    If the plots fall in a straight line, the feature is normally distributed.
-----------------------------------------------------------------------------------------------------------------------
"""

import numpy as np
import pandas as pd
import scipy.stats as stat
from matplotlib import pyplot as plt
import pylab


def plot_data(df, feature):
    plt.figure(figsize=(9, 5))

    plt.subplot(1, 2, 1)
    df[feature].hist(rwidth=0.9)

    plt.subplot(1, 2, 2)
    stat.probplot(df[feature], dist='norm', plot=pylab)

    plt.show()


if __name__ == '__main__':
    df = pd.read_csv('titanic_age.csv', index_col='PassengerId')
    df.Age = df.Age.fillna(df.Age.median())

    plot_data(df, 'Age')                                        # Current Distribution

    ''' If the feature has 0.0 values, use np.log1p (i.e., log(1+x)) instead of np.log (i.e. log(x) '''
    df['Age_log'] = np.log(df.Age)                              # Logarithmic Transformation
    plot_data(df, 'Age_log')

    # df['Age_reciprocal'] = 1 / df.Age                           # Reciprocal Transformation
    # plot_data(df, 'Age_reciprocal')

    # df['Age_sqrt'] = np.sqrt(df.Age)                            # Square root Transformation
    # plot_data(df, 'Age_sqrt')

    # df['Age_exp'] = df.Age**(1/1.2)                             # Exponential Transformation
    # plot_data(df, 'Age_exp')

    # df['Age_boxcox'], lambda_param = stat.boxcox(df.Age)        # Box - Cox Transformation
    # plot_data(df, 'Age_boxcox')

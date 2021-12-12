"""
-----------------------------------------------------------------------------------------------------------------------
Mutual Information:
-----------------------------------------------------------------------------------------------------------------------
Mutual Information (MI) is a general-purpose metric measuring associations between a feature and the target. Once you
rank the features using MI, you can then choose a smaller set of the most useful features to develop your model. MI is
a lot like correlation in that it measures a relationship between two quantities. The advantage of mutual information
is that it can detect any kind of relationship, while correlation only detects linear relationships.

MI is especially useful at the start of feature development when you might not know what model you'd use. It is:
    - easy to use and interpret,
    - computationally efficient,
    - theoretically well-founded,
    - resistant to over-fitting, and,
    - able to detect any kind of relationship

Mutual Information describes relationships in terms of uncertainty. The MI between two quantities is a measure of
the extent to which knowledge of one quantity reduces uncertainty about the other. If you knew the value of a feature,
how much more confident would you be about the target?

[Note: Uncertainty is measured using a quantity from information theory known as "entropy". The entropy of a random
variable is the average level of “information“, “surprise”, or “uncertainty” inherent in the variable’s possible
outcomes. It is the measure of uncertainty. As we increase uncertainty, Entropy reduces. Entropy roughly means: "how
many yes-or-no questions you would need to describe an occurrence of that variable, on average." The more questions you
have to ask, the more uncertain you must be about the variable. Mutual information is how many questions you expect the
feature to answer about the target.]

The least possible mutual information between quantities is 0.0. When MI is zero, the quantities are independent:
neither can tell you anything about the other. Conversely, in theory there's no upper bound to what MI can be.
In practice though values above 2.0 or so are uncommon. (MI is a logarithmic quantity, so it increases very slowly.)

Points to remember while applying MI:
1. MI can help you to understand the relative potential of a feature as a predictor of the target, considered by itself.
2. It's possible for a feature to be very informative when interacting with other features, but not so informative all
    alone. MI can't detect interactions between features. It is a uni-variate metric.
3. The actual usefulness of a feature depends on the model you use it with. A feature is only useful to the extent that
    its relationship with the target is one your model can learn. Just because a feature has a high MI score doesn't
    mean your model will be able to do anything with that information. You may need to transform the feature first to
    expose the association.
-----------------------------------------------------------------------------------------------------------------------
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression


def get_mi_score(X, y, discrete_features):
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name='MI Scores', index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores


def plot_mi_scores(mi_scores):
    plt.figure(figsize=(12, 5))
    plt.tight_layout()
    plt.barh(mi_scores.index, mi_scores.values, color='#adad3b', label='Mutual Information Scores')
    plt.show()


if __name__ == '__main__':
    df = pd.read_csv('automobile_data.csv')                         # (193, 25)

    X = df.copy()
    y = X.pop('price')

    """ Apply label encoding for Categorical features """
    for col in X.select_dtypes('object'):
        X[col], _ = X[col].factorize()

    """ Make sure all discreet features have integer dtypes """
    discreet_features = X.dtypes == int                             # Series

    mi_scores = get_mi_score(X, y, discrete_features=discreet_features)
    print(mi_scores)
    plot_mi_scores(mi_scores)

    """ curb-weight has the highest mutual information score. See its relationship with the target """
    plt.scatter(X['curb-weight'], y)
    plt.show()

    """
    fuel_type has a fairly low MI score, but as we can see from the figure, it clearly separates two price populations 
    with different trends within the horsepower feature. This indicates that fuel_type contributes an interaction 
    effect and might not be unimportant after all. Before deciding a feature is unimportant from its MI score, it's 
    good to investigate any possible interaction effects -- domain knowledge can offer a lot of guidance here.
    """
    sns.lmplot(x='horsepower', y='price', hue='fuel-type', data=df)
    plt.show()

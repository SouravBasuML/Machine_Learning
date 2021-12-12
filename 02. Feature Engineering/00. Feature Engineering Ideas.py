"""
-----------------------------------------------------------------------------------------------------------------------
Feature Engineering:
-----------------------------------------------------------------------------------------------------------------------
The goal of feature engineering is to make our data better suited to the problem at hand. It helps to:
    - improve a model's predictive performance
    - reduce computational or data needs
    - improve interpretability of the results

For a feature to be useful, it must have a relationship to the target that our model is able to learn. Linear models,
for instance, are only able to learn linear relationships. So, when using a linear model, our goal is to transform the
features to make their relationship to the target linear. The key idea here is that a transformation we apply to a
feature becomes in essence a part of the model itself.

Whatever relationships our model can't learn, we can provide ourselves through transformations. As we develop our
feature set, think about what information our model could use to achieve its best performance.


-----------------------------------------------------------------------------------------------------------------------
Tips on Discovering New Features:
-----------------------------------------------------------------------------------------------------------------------
    - Understand the features. Refer to your dataset's data documentation, if available.
    - Research the problem domain to acquire domain knowledge. Wikipedia can be a good starting point, but books and
        journal articles will often have the best information.
    - Study previous work.
    - Use data visualization. Visualization can reveal pathologies in the distribution of a feature or complicated
        relationships that could be simplified. Be sure to visualize your dataset as you work through the feature
        engineering process.


-----------------------------------------------------------------------------------------------------------------------
Tips on Creating Features:
-----------------------------------------------------------------------------------------------------------------------
    It's good to keep in mind your model's own strengths and weaknesses when creating features. Here's some guidelines:
    - Linear models learn sums and differences naturally, but can't learn anything more complex.
    - Ratios seem to be difficult for most models to learn. Ratio combinations often lead to some easy performance gains
    - Linear models and neural nets generally do better with normalized features. Neural nets especially need features
        scaled to values not too far from 0. Tree-based models (like random forests and XGBoost) can sometimes benefit
        from normalization, but usually much less so.
    - Tree models can learn to approximate almost any combination of features, but when a combination is especially
        important they can still benefit from having it explicitly created, especially when data is limited.
    - Counts are especially helpful for tree models, since these models don't have a natural way of aggregating
        information across many features at once.
"""

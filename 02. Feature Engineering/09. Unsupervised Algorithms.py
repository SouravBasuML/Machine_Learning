"""
-----------------------------------------------------------------------------------------------------------------------
Feature Discovery using Unsupervised Learning Algorithms
-----------------------------------------------------------------------------------------------------------------------

-----------------------------------------------------------------------------------------------------------------------
Clustering:
-----------------------------------------------------------------------------------------------------------------------
Clustering the feature set and adding a feature of cluster labels can help machine learning models untangle complicated
relationships of space or proximity. e.g.
    - discovering groups/clusters of customers representing a market segment,
    - labeling geographic areas (using latitude and longitude) based on median income to create economic segments
    - labeling geographic areas that share similar weather patterns

Applied to a single real-valued feature, clustering acts like a traditional "binning" or "discretization" transform.
On multiple features, it's like "multidimensional binning" (sometimes called vector quantization).

The motivating idea for adding cluster labels is that the clusters will break up complicated relationships across
features into simpler chunks. Our model can then just learn the simpler chunks one-by-one instead having to learn the
complicated whole all at once. It's a "divide and conquer" strategy.

Label encoding or One-hot encoding may have to be applied to the output of the clustering algorithm.

The k-means algorithm offers an alternative way of creating features. Instead of labelling each feature with the
nearest cluster centroid (using the fit_predict() method), it can measure the distance from a point to all the
centroids (using the fit_transform() method) and return those distances as features.

    fit_predict()   -> Compute cluster centers and predict cluster index for each sample.
    fit_transform() -> Compute clustering and transform X to cluster-distance space


-----------------------------------------------------------------------------------------------------------------------
Principal Component Analysis:
-----------------------------------------------------------------------------------------------------------------------
PCA partitions the dataset based on variation in the data. PCA is a great tool to discover important relationships in
the data and can also be used to create more informative features. The idea of PCA is instead of describing the data
with the original features, describe it with its axes of variation. e.g. Instead of describing the data with 'height'
and diameter, describe it with 'shape' and 'size' (axes of variation). The axes of variation become the new features.

The new features PCA constructs are actually just linear combinations (weighted sums) of the original features. These
new features are called the "principal components" of the data. The weights themselves are called "loadings". A
component's loadings tell us what variation it expresses through signs and magnitudes. There will be as many principal
components as there are features in the original dataset.

PCA also tells us the amount of variation in each component. PCA makes this precise through each component's percent
of explained variance. It's important to remember, however, that the amount of variance in a component doesn't
necessarily correspond to how good it is as a predictor: it depends on what you're trying to predict.

Technical note: PCA is typically applied to standardized data. With standardized data "variation" means "correlation".
With un-standardized data "variation" means "covariance".


PCA for Feature Engineering:
----------------------------
There are two ways you could use PCA for feature engineering:

1. The first way is to use it as a descriptive technique. Since the components tell you about the variation, you
    could compute the MI scores for the components and see what kind of variation is most predictive of your target.
    That could give you ideas for kinds of features to create. You could even try clustering on one or more of the
    high-scoring components.

2. The second way is to use the components themselves as features. Because the components expose the variational
    structure of the data directly, they can often be more informative than the original features. Some use-cases:

    a. Dimensionality reduction: When your features are highly redundant (multi-collinear, specifically), PCA will
        partition out the redundancy into one or more near-zero variance components, which you can then drop since they
        will contain little or no information.

    b. Anomaly detection: Unusual variation, not apparent from the original features, will often show up in the
        low-variance components. These components could be highly informative in an anomaly or outlier detection task.

    c. Noise reduction: A collection of sensor readings will often share some common background noise. PCA can
        sometimes collect the (informative) signal into a smaller number of features while leaving the noise alone,
        thus boosting the signal-to-noise ratio.

    d. De-correlation: Some ML algorithms struggle with highly-correlated features. PCA transforms correlated features
        into uncorrelated components, which could be easier for your algorithm to work with.

PCA basically gives you direct access to the correlational structure of your data.


PCA Best Practices:
-------------------
Few things to keep in mind when applying PCA:
1. PCA only works with numeric features, like continuous quantities or counts.
2. PCA is sensitive to scale. It's good practice standardizing your data before applying PCA, unless you know you have
    good reason not to.
3. Consider removing or constraining outliers, since they can have an undue influence on the results.

"""
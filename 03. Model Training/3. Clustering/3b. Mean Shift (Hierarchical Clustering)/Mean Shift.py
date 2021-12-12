import numpy as np
from sklearn.cluster import MeanShift
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import style
style.use("ggplot")


# ---------------------------------------------------------------------------------------------------------------------
# Clustering:
# ---------------------------------------------------------------------------------------------------------------------
# Unsupervised ML Algorithm:
# The algorithm is given raw feature dataset without labels and it attempts to figure out clusters.
# Clustering Types:
# 1. Flat clustering: We give the algorithm the number of clusters to find
# 2. Hierarchical Clustering: The algorithm figures out the number of clusters to segregate the feature set
# Clustering Algorithm Types:
# 1. K-Means - Flat clustering, K: number of clusters
# 2. Mean Shift - Hierarchical Clustering
# ---------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------------------------------
# Hierarchical Clustering:
# ---------------------------------------------------------------------------------------------------------------------
# Hierarchical Clustering figures out the number of clusters to segregate the feature set based on similarity

# Hierarchical Clustering is often associated with heat maps, which shows the similarities between features (columns)
# and samples (rows). Hierarchical Clustering orders the rows and/or the columns based on similarity, which makes it
# easy to see correlations in the data.

# Hierarchical Clustering is usually accompanied by a Dendrogram, which indicates both the similarity and the order
# that the clusters were formed. The shorter the dendrogram, the earlier it was created and having more similarity.

# Similarity can be determined by Euclidean distances or Manhattan distances.

# Ways to compare clusters:
# To identify in which cluster a new sample should be classified, we can compare that point to:
# 1. The centroid of each cluster
# 2. The closest point in each cluster (single-linkage)
# 3. The farthest point in each cluster (complete-linkage)
# ---------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------------------------------
# Mean Shift:
# ---------------------------------------------------------------------------------------------------------------------
# 1. Every data point in the feature set is a cluster center
# 2. A radius or bandwidth may be supplied which is applied to every data point. Bandwidth is the area around a
#    data point with a certain radius. Bandwidth may be applied at different levels with decreasing weights.
# 3. Take the first data point, identify the datapoints that lie within it's bandwidth (defined by the radius)
# 4. Take the mean of the data points that lie within the bandwidth of the first data point. This point becomes the
#    new cluster center.
# 5. Repeat steps 3 and 4 with the new cluster center that is identified in each iteration. Keep repeating, until
#    the cluster center does not move (i.e. convergence is reached).
# 6. Repeat steps 3, 4, and 5 with all the data points (m) in the feature set to identify m cluster centers.
# 7. Remove duplicates to identify unique cluster centers and thus identifying unique clusters.
# ---------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    # Create starting sample data
    centers = [[1, 1, 1], [5, 5, 5], [3, 10, 10]]
    # Generate isotropic Gaussian blobs for clustering of type numpy.ndarray
    X, _ = make_blobs(n_samples=100, centers=centers, cluster_std=1.5)

    clustering = MeanShift()
    clustering.fit(X)

    # cluster_centers_ gives us the n-dim cluster centers. labels_ gives us the cluster labels 0, 1, 2 etc.
    labels = clustering.labels_
    cluster_centers = clustering.cluster_centers_

    # The cluster centers will be close to our starting centers [1,1,1], [5,5,5], [3,10,10].
    # If you increase n_samples, the cluster centers will be closer to our starting
    print(cluster_centers)

    # Get number of clusters identified by Mean Shift algo
    n_clusters_ = len(np.unique(labels))
    print("Number of estimated clusters:", n_clusters_)

    # Plotting:
    colors = ['r', 'g', 'b', 'c', 'k', 'y', 'm']
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in range(len(X)):
        ax.scatter(X[i][0], X[i][1], X[i][2], c=colors[labels[i]], s=5, marker='o')

    ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], cluster_centers[:, 2],
                marker='x', color='k', s=50, zorder=10)

    plt.show()

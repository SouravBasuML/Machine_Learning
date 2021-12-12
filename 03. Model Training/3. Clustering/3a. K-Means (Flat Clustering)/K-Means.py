import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')


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
# K-Means:
# ---------------------------------------------------------------------------------------------------------------------
# 1. Algorithm randomly chooses K centroids (cluster centers) from among the feature set
# 2. Calculate the distance of each feature from the centroid and classify each feature to a cluster (centroid)
#    based on its distance from the centroid
# 3. Calculate the mean (center) of the data points classified into each cluster in Step 2, and move the centroid
#    to the new position
# 4. Keep repeating steps 2 and 3 until the new centroids do not move much.
# ---------------------------------------------------------------------------------------------------------------------
# K-Means has a 'tolerance' (default 1e-4) that tells the algorithm to stop iterating if the centroids' movement is
# smaller than the tolerance. It also has max_iter (default 300 in scikit learn) that stops the iterations when
# max_iter is reached, even if the algorithm has not found the optimum centroids.
# ---------------------------------------------------------------------------------------------------------------------
# Identify Optimal K: (Elbow Method)
# Start with K=2 and calculate the Sum of Squared Errors for each cluster. SSE is given by the sum of squared
# distances of every data point from its centroid in each cluster. Repeat this step for K = 3, 4, ... etc.
# Plot graph of SSE vs K. Identify optimal K where there is an elbow in the graph
# ---------------------------------------------------------------------------------------------------------------------
# Disadvantages:
# 1. K-Means tries to cluster data points into equal sized groups, hence if the actual clusters are of different sizes,
#    K-Means may fail to cluster correctly, e.g. the mouse dataset.
# 2. Scaling: Every point's distance has to be compared from the cluster centroid, hence K-Means does not scale
#    efficiently to very large datasets
# ---------------------------------------------------------------------------------------------------------------------
# K-Means is an example of semi-supervised algorithm. Once the cluster centers are found, we can use a supervised
# ML algorithm like SVM to classify new points based on those centroids; you don't need to train again.
# ---------------------------------------------------------------------------------------------------------------------

def plot_cluster():
    colors = ['g.', 'r.', 'b.']
    for i in range(len(X)):
        plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize=10)
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='x')
    plt.show()


def plot_elbow():
    k_range = range(1, 10)
    sum_squared_error = []
    for k in k_range:
        clf = KMeans(n_clusters=k)
        clf.fit(X)
        # clf.inertia_ gives the sum of squared distances of samples to their closest cluster center
        sum_squared_error.append(clf.inertia_)
    print(sum_squared_error)

    plt.plot(k_range, sum_squared_error)
    plt.show()


if __name__ == '__main__':
    X = np.array([[1, 0.6], [1, 2], [1.5, 1.8], [2, 2], [2, 3],
                  [6, 8], [8, 8], [7, 10], [8, 10], [9, 11],
                  [7, 3], [8, 1.5], [8, 3], [9, 4], [7, 4]])

    clf = KMeans(n_clusters=3)                          # default is 8. It must be <= number of data points
    clf.fit(X)

    # Array of coordinates of cluster centers
    centroids = clf.cluster_centers_

    # Array of labels of each point. Labels can be compared to y. First cluster will be label 0, second will be label 1
    # and so on... Labels vary with each execution. Label 0 in first execution may be classified as label 1 in
    # second execution e.g. labels [0 0 1 1 0 1] or [1 1 0 0 1 0]
    labels = clf.labels_

    print('Centroids: ', centroids)
    print('Labels: ', labels)

    plot_cluster()
    plot_elbow()

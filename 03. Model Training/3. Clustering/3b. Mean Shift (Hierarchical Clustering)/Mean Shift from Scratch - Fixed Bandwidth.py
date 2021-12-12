import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')


class Mean_Shift:

    # Radius = 4 works well in this example.
    # If radius = 40, the algorithm considers the entire dataset as only one cluster and finds only one centroid.
    # If radius = 2, data points that are spread apart and considered one cluster
    def __init__(self, radius=40):
        self.radius = radius

    def fit(self, data):

        # Initially every data point is a cluster centroid
        centroids = {}
        for i in range(len(data)):
            centroids[i] = data[i]

        while True:                                     # scikit learn MeanShift does not have max_iter
            new_centroids = []
            for i in centroids:
                in_bandwidth = []                       # Holds data points within the radius of the centroid

                # Populate in_bandwidth with all data points whose distance is < radius
                centroid = centroids[i]
                for featureset in data:
                    if np.linalg.norm(featureset - centroid) < self.radius:
                        in_bandwidth.append(featureset)

                # Calculate new centroid as the average of the data points that are within the radius
                new_centroid = np.average(in_bandwidth, axis=0)
                new_centroids.append(tuple(new_centroid))

            # At this point, new_centroids has a new centroid corresponding to each data point in the feature set.
            # Get the unique centriods from among the newly calculated centroids.
            # np.unique doesn't work here as it gives the unique of each value in the array not the unique of the array
            uniques = sorted(list(set(new_centroids)))

            # Store current centroids in prev centroids for comparison
            prev_centroids = dict(centroids)

            print('New Centroids: ', new_centroids)
            print('Uniq :', uniques)
            print('Prev Centroids :', prev_centroids)

            # Copy the unique centroids found above in this iteration to centroids dictionary
            centroids = {}
            for i in range(len(uniques)):
                centroids[i] = np.array(uniques[i])

            print('Centroids :', centroids)

            # Compare new and prev centroids to see if they have moved. If they are same, centroids have converged
            optimized = True
            for i in centroids:
                if not np.array_equal(centroids[i], prev_centroids[i]):
                    optimized = False
                if not optimized:           # even if one centroid has moved, centroids have not converged
                    break

            if optimized:                   # break the while loop
                break

        # Reset the centroids
        self.centroids = centroids


if __name__ == '__main__':

    X = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11], [8, 2], [10, 2], [9, 3]])
    colors = 10 * ["g", "r", "c", "b", "k"]

    clf = Mean_Shift()
    clf.fit(X)

    centroids = clf.centroids

    # scatter the feature set
    plt.scatter(X[:, 0], X[:, 1], s=25)
    # scatter the centroids
    for c in centroids:
        plt.scatter(centroids[c][0], centroids[c][1], color='k', marker='*', s=25)
    plt.show()

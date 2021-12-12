import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')


class K_Means:

    def __init__(self, k=2, tol=0.001, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, data):
        self.centroids = {}

        # Initialize the K centroids with the first two K points: {0: array([1., 2.]), 1: array([1.5, 1.8])}
        # you may shuffle the data points before initializing the centroids, but not necessary
        for i in range(self.k):
            self.centroids[i] = data[i]

        # Start the optimization process and loop till max_iter or if tolerance is reached
        for i in range(self.max_iter):
            # Dictionary to contain the K centroids (key) and the feature set classified to those centroids (values)
            # For every iteration, clear out the classifications as it will change based on the new centroids
            self.classifications = {}

            for j in range(self.k):
                self.classifications[j] = []                # {0: [], 1: []}

            # Calculate the distance of each data point from the centroids
            for feature_set in data:
                # 'distances' is a list of two distances of each data point from the two centroids, e.g.
                # Centroids: {0: [1., 2.], 1: [1.5, 1.8]}
                # first data point: [1. 2.]
                # distances: [0.0, 0.5385164807134504]
                distances = [np.linalg.norm(feature_set - self.centroids[centroid]) for centroid in self.centroids]

                # Assign the data point to the centroid with the minimum distance
                # 'classification' will give the key to the dictionary (either 0 or 1)
                # 'feature_set' appended to that key will give the value of the dictionary for that key
                classification = distances.index(min(distances))
                self.classifications[classification].append(feature_set)

            # Save the current centroid before calculating the new centroid
            prev_centroids = dict(self.centroids)

            # Find new centroids by averaging the data points assigned to the centroid in this iteration
            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)

            optimized = True

            # Check if the centroids moved more than the tolerance
            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid - original_centroid) / original_centroid * 100.0) > self.tol:
                    optimized = False

            # Break, if tolerance is reached
            if optimized:
                break

    def predict(self, data):
        # Calculate the distance from the final/trained centroids
        distances = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
        # Return the key of the minimum distance as the classification
        classification = distances.index(min(distances))
        return classification


if __name__ == '__main__':

    X = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
    clf = K_Means()
    clf.fit(X)

    for centroid in clf.centroids:
        plt.scatter(clf.centroids[centroid][0], clf.centroids[centroid][1],
                    marker="o", color="k", s=10, linewidths=5)

    colors = ["g", "r", "c", "b", "k"]
    for classification in clf.classifications:
        color = colors[classification]
        for featureset in clf.classifications[classification]:
            plt.scatter(featureset[0], featureset[1], marker='x', color=color, s=50)

    unknowns = np.array([[1, 3],
                         [8, 9],
                         [0, 3],
                         [5, 4],
                         [6, 4]])

    for unknown in unknowns:
        classification = clf.predict(unknown)
        plt.scatter(unknown[0], unknown[1], marker="*", color=colors[classification], s=25)

    plt.show()

import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.datasets import make_blobs
style.use('ggplot')


class MeanShift:

    # Radius is not hard coded. We are using a range of radii (0 to 99), but we will penalize the data points
    # that are further away from centroid
    def __init__(self, radius=None, radius_norm_step=100):
        self.radius = radius
        self.radius_norm_step = radius_norm_step

    def fit(self, data):

        # If no radius is provided, we need to define a radius. Find the centroid of all the data points, find its
        # magnitude (distance of this centroid from the origin), divide by the step to get the radius.
        if self.radius is None:
            all_data_centroid = np.average(data, axis=0)                # e.g. [-2.49  2.37]
            all_data_norm = np.linalg.norm(all_data_centroid)           # e.g. 3.43
            self.radius = all_data_norm / self.radius_norm_step         # e.g. 0.0343

        # Initially every data point is a cluster centroid
        centroids = {}
        for i in range(len(data)):
            centroids[i] = data[i]

        # Define weights as [99, 98, ... , 0]
        weights = [i for i in range(self.radius_norm_step)][::-1]

        while True:                                             # scikit learn MeanShift does not have max_iter
            new_centroids = []
            for i in centroids:
                in_bandwidth = []                               # holds data points within the radius of the centroid
                centroid = centroids[i]

                # Populate in_bandwidth with all data points whose distance is < radius
                for featureset in data:
                    distance = np.linalg.norm(featureset - centroid)
                    # Get the index to search the weight in the weights list [99, 98, ... , 0]
                    # weight_index gives the number of radius steps. The larger the weight_index the smaller the weight
                    weight_index = int(distance / self.radius)
                    # If the data point is further away than the largest radius, (if the distance > 99, assign it 99)
                    if weight_index > self.radius_norm_step - 1:
                        weight_index = self.radius_norm_step - 1

                    to_add = (weights[weight_index] ** 2) * [featureset]
                    in_bandwidth += to_add

                # Calculate new centroid as the average of the data points that are within the radius
                new_centroid = np.average(in_bandwidth, axis=0)
                new_centroids.append(tuple(new_centroid))

            # At this point, new_centroids has a new centroid corresponding to each data point in the feature set.
            # Get the unique centroids from among the newly calculated centroids.
            # np.unique doesn't work here as it gives the unique of each value in the array not the unique of the array
            uniques = sorted(list(set(new_centroids)))

            # Merge centroids that are very close to each other (i.e. whose distance from each other is <= radius)
            to_pop = []
            for i in uniques:
                for ii in [i for i in uniques]:
                    # Ignore the ones that are exactly equal, as we ignored them when we populated uniques
                    if i == ii:
                        pass
                    elif np.linalg.norm(np.array(i) - np.array(ii)) <= self.radius:
                        to_pop.append(ii)
                        break

            for i in to_pop:
                try:
                    uniques.remove(i)
                except:
                    pass

            # Store current centroids in prev centroids for comparison
            prev_centroids = dict(centroids)

            # Copy the unique centroids found above in this iteration to centroids dictionary
            centroids = {}
            for i in range(len(uniques)):
                centroids[i] = np.array(uniques[i])

            # Compare new and prev centroids to see if they have moved. If they are same, centroids have converged
            optimized = True
            for i in centroids:
                if not np.array_equal(centroids[i], prev_centroids[i]):
                    optimized = False
            if optimized:
                break

        self.centroids = centroids

        # Classify each data point into a dictionary with key as the cluster numbers 0, 1, 2 etc.
        self.classifications = {}
        for i in range(len(self.centroids)):
            self.classifications[i] = []

        for featureset in data:
            # Compare distance to all centroids and get the minimum to classify the data point to that cluster
            distances = [np.linalg.norm(featureset - self.centroids[centroid]) for centroid in self.centroids]
            classification = (distances.index(min(distances)))
            # featureset that belongs to that cluster
            self.classifications[classification].append(featureset)

    def predict(self, data):
        # Compare distance to all centroids and get the minimum to classify the data point to that cluster
        distances = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
        classification = (distances.index(min(distances)))
        return classification


if __name__ == '__main__':

    X, y = make_blobs(n_samples=20, centers=3, n_features=2)
    # X = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11], [8, 2], [10, 2], [9, 3]])

    clf = MeanShift()
    clf.fit(X)

    centroids = clf.centroids
    print(centroids)

    colors = 10 * ['r', 'g', 'b', 'c', 'k', 'y']

    for classification in clf.classifications:
        color = colors[classification]
        for featureset in clf.classifications[classification]:
            plt.scatter(featureset[0], featureset[1], marker='x', color=color, s=25, zorder=10)

    for c in centroids:
        plt.scatter(centroids[c][0], centroids[c][1], color='k', marker="*", s=25)

    plt.show()

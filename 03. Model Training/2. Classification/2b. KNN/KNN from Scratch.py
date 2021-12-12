from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import warnings
from collections import Counter
style.use('fivethirtyeight')


# ---------------------------------------------------------------------------------------------------------------------
# K - Nearest Neighbours (Classification)
# ---------------------------------------------------------------------------------------------------------------------
# https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
# ---------------------------------------------------------------------------------------------------------------------
# Classification: Create a model that best separates our data
# KNN: Find the 'K' closest points (nearest neighbours) to our data point and assign the data point to the
# class/cluster that has majority among those 'K' points. Chose 'K to be an odd number.
# You get accuracy of your model and a confidence for each data point. e.g. If K = 3 and the votes are - - +
# then there is a 67% confidence that the data point belongs to the '-' cluster.
# KNN works on both linear and non-linear data.
# KNN runs slow if the data set is very large and there are many dimensions (SVM is better at classification).
# To over this, we can find 'K' closest points to our data point within a certain radius and ignore the outliers.
# Also with KNN there is actually no training, we will have to measure the distance of our new data point everytime.
# ---------------------------------------------------------------------------------------------------------------------
# The Euclidean distance between two points in Euclidean space is the length of a line segment between the two points.
# It can be calculated from the Cartesian coordinates of the points using the Pythagorean theorem,
# therefore occasionally being called the Pythagorean distance.
# d(p,q)= sqrt((p1-q1)^2 + (p2-q2)^2 + ... + (p(i)-q(i))^2 + ... + (p(n)-q(n))^2)
# ---------------------------------------------------------------------------------------------------------------------

def k_nearest_neighbors(training_data, data_to_predict, k=3):
    if len(training_data) >= k:
        warnings.warn('K is set to a value less than the total voting groups (clusters)!')

    distances = []
    for group in training_data:
        for features in training_data[group]:
            # euclidean_distance =
            #   sqrt((features[0] - data_to_predict[0]) ** 2 + (features[1] - data_to_predict[1]) ** 2)
            #   np.sqrt(np.sum((np.array(features) - np.array(data_to_predict)) ** 2))
            euclidean_distance = np.linalg.norm(np.array(features) - np.array(data_to_predict))
            distances.append([euclidean_distance, group])

    # distances: [[6.4, 'k'], [5.0, 'k'], [6.32, 'k'], [2.23, 'r'], [2.0, 'r'], [3.16, 'r']]
    # once the distances are sorted, we need the top smallest distances up to k (in this case top 3 distances)
    # i[0] gives the distances, i[1] gives the group (in this case 'r' or 'k')
    votes = [i[1] for i in sorted(distances)[:k]]           # ['r', 'r', 'r']
    # Counter(votes).most_common(1) = [('r', 3)]            # Most voted class was 'r' and the vote was 3
    # Counter(votes).most_common(1)[0] = ('r', 3)
    # Counter(votes).most_common(1)[0][0] = r
    # Counter(votes).most_common(1)[0][1] = 3
    vote_result = Counter(votes).most_common(1)[0][0]       # 'r'
    return vote_result


if __name__ == '__main__':
    # Used names g and r to make plotting easy. g = green, r = red
    dataset = {'g': [[1, 2], [2, 3], [3, 1]], 'r': [[6, 5], [7, 7], [8, 6]]}
    new_features = [5, 7]

    result = k_nearest_neighbors(training_data=dataset, data_to_predict=new_features, k=3)    # r

    [[plt.scatter(j[0], j[1], color=i) for j in dataset[i]] for i in dataset]
    plt.scatter(new_features[0], new_features[1], s=100, color=result)
    plt.show()

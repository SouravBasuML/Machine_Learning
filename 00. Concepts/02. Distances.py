"""
-----------------------------------------------------------------------------------------------------------------------
Distance Metrics:
-----------------------------------------------------------------------------------------------------------------------
Distance metrics are used in both supervised and unsupervised learning (e.g. classification or clustering), to
calculate the similarity between data points. There are four types of distance metrics used in Machine Learning
1. Euclidean Distance
2. Manhattan Distance
3. Minkowski Distance
4. Hamming Distance
-----------------------------------------------------------------------------------------------------------------------

-----------------------------------------------------------------------------------------------------------------------
1. Euclidean Distance:
-----------------------------------------------------------------------------------------------------------------------
The Euclidean distance between two points in Euclidean space is the length of a line segment between the two points.
It represents the shortest distance between two points. It can be calculated from the Cartesian coordinates of the
points using the Pythagorean theorem, therefore occasionally being called the Pythagorean distance.
d(p,q)= sqrt((p1-q1)^2 + (p2-q2)^2 + ... + (p(i)-q(i))^2 + ... + (p(n)-q(n))^2)

euclidean_distance =
  sqrt((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2)
  np.sqrt(np.sum((np.array(p) - np.array(q)) ** 2))
  np.linalg.norm(np.array(p) - np.array(q))
-----------------------------------------------------------------------------------------------------------------------


-----------------------------------------------------------------------------------------------------------------------
Manhattan Distance:
-----------------------------------------------------------------------------------------------------------------------
The Manhattan Distance between two points is the sum of the absolute differences of their Cartesian coordinates.
i.e., it is the sum of absolute differences between points across all the dimensions
The Manhattan Distance is also known as the rectilinear distance, L1 distance, l1 norm, snake distance,
city block distance, taxicab distance, or Manhattan length.

d(p, q) = |p - q|
d(p, q) = |p1 - q1| + |p2 - q2| + ... + |pn - qn|

where p = (p1, p2, ... , pn)
and   q = (q1, q2, ... , qn)
e.g. In 2D, the Manhattan distance between (p1, p2) and (q1, q2) is |p1 - q1| + |p2 - q2|
-----------------------------------------------------------------------------------------------------------------------


-----------------------------------------------------------------------------------------------------------------------
Minkowski Distance:
-----------------------------------------------------------------------------------------------------------------------
Minkowski Distance is the generalized form of Euclidean and Manhattan Distance.
d(p,q)= ( (p1-q1)^2 + (p2-q2)^2 + ... + (p(i)-q(i))^2 + ... + (p(n)-q(n))^2 ) ^ (1/p)

where, p represents the order of the norm.
When p = 1, it is the Manhattan Distance
When p = 2, it is the Euclidean Distance
Intermediate values of 'p' provide a controlled balance between the two measures.
-----------------------------------------------------------------------------------------------------------------------


-----------------------------------------------------------------------------------------------------------------------
Hamming Distance:
-----------------------------------------------------------------------------------------------------------------------
Hamming Distance is used to measure similarity between categorical variables. It measures the similarity between two
strings of the same length. The Hamming Distance between two strings of the same length is the number of positions at
which the corresponding characters are different. The larger the Hamming Distance between two strings, more
dissimilar will those strings be (and vice versa).
-----------------------------------------------------------------------------------------------------------------------
"""

from scipy.spatial import distance


p = (1, 2, 3)
q = (4, 5, 6)

# Euclidean Distance:
euclidean_distance = distance.euclidean(p, q)
print('Euclidean Distance between points', p, 'and', q, 'is: ', euclidean_distance)


# Manhattan Distance
manhattan_distance = distance.cityblock(p, q)
print('Manhattan Distance between points', p, 'and', q, 'is: ', manhattan_distance)


# Minkowski Distance:
minkowski_distance = distance.minkowski(p, q, p=1)
print('Minkowski Distance between points', p, 'and', q, 'of order 1 is: ', minkowski_distance)

minkowski_distance = distance.minkowski(p, q, p=2)
print('Minkowski Distance between points', p, 'and', q, 'of order 2 is: ', minkowski_distance)

minkowski_distance = distance.minkowski(p, q, p=3)
print('Minkowski Distance between points', p, 'and', q, 'of order 3 is: ', minkowski_distance)


# Hamming Distance:
string_1 = 'euclidean'
string_2 = 'manhattan'
hamming_distance = distance.hamming(list(string_1), list(string_2))*len(string_1)
print('Hamming Distance between strings', string_1, 'and', string_2, 'is: ', hamming_distance)

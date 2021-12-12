import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np

style.use('ggplot')


# ---------------------------------------------------------------------------------------------------------------------
# Support Vector Machine:
# ---------------------------------------------------------------------------------------------------------------------
# Supervised Algorithm, Binary Classifier, Large Margin Classifier
# https://scikit-learn.org/stable/modules/svm.html
# https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
# https://medium.com/deep-math-machine-learning-ai/chapter-3-support-vector-machine-with-math-47d6193c82be
# ---------------------------------------------------------------------------------------------------------------------
# Classification: Create a model that best separates our data
# SVM is a binary classifier - It will divide the data points into two groups at a time
# The objective of SVM is to find the best separating hyperplane (or decision boundary) such that the distance
# between the hyperplane and the data that it separates is the greatest
# ---------------------------------------------------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------------------------------------------------
# https://scikit-learn.org/stable/modules/svm.html#svm-kernels
# Kernels are a similarity function that allow us to work with non-linear data by transforming the non-linear data
# in say X-space to Z-space by adding more dimensions and converting it to a linearly separable dataset.
# Kernels take inputs and outputs the similarity among them using their inner product. Inner product is a projection
# (or overlapping) of one vector onto another, and hence tells us the 'similarity'.
# This modification will have no effect on:
# 1. The classification algorithm y = sign(w.x + b), as w.x (with fewer dimensions) and w.z (with more dimensions)
#    will both return a scalar.
# 2. The constraints yi(xi.w + b) - 1 >= 0 and w = E(ai.yi.xi)
# Kernel Types: RBF (Radio Basis Function - Default kernel in Scikit Learn), Gaussian, Polynomial, Sigmoid
# ---------------------------------------------------------------------------------------------------------------------
# Soft Margin SVM:
# ---------------------------------------------------------------------------------------------------------------------
# Using kernels to go to higher dimensions may lead to over fitting. A soft margin SVM avoids over fitting by
# sacrificing hard separation. Fewer data points are classified as support vectors, and some data points may be
# misclassified (some data points may violate the decision boundary hyperplane).
# e.g. # if support vectors / # data points > 20%, there could be over fitting
# The softness of the margin is defined by parameter called 'slack' denoted by epsilon -> yi(xi.w + b) >= 1 - e
# e must be >=0. If e = 0, it is the hard margin. Regularization Parameter 'C' is used to control slack 'e'
# ---------------------------------------------------------------------------------------------------------------------
# OVR (One vs. Rest) vs OVO (One vs. One):
# OVR: Default. Each decision boundary hyperplane creates unbalanced clusters
# OVO: You will have to repeatedly evaluate the hyperplanes to find the right cluster
# ---------------------------------------------------------------------------------------------------------------------

class SupportVectorMachine:
    # Defining SVM as a class as we want to train and save the classifier as an object. Not required for KNN
    def __init__(self, visualization=True):
        self.visualization = visualization
        self.colors = {1: 'r', -1: 'b'}
        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1, 1, 1)

    # train
    def fit(self, training_data):
        self.data = training_data
        opt_dict = {}                                               # {||w||: [w,b]}
        transforms = [[1, 1], [-1, 1], [-1, -1], [1, -1]]           # Transform will be applied to w vector

        # Get max and min ranges
        all_data = []
        for yi in self.data:                                        # Class +1 or -1
            for featureset in self.data[yi]:                        #
                for feature in featureset:
                    all_data.append(feature)
        self.max_feature_value = max(all_data)
        self.min_feature_value = min(all_data)
        all_data = None                                             # So we are not holding all data points in memory

        # Support Vectors yi(xi.w + b) = 1. Keep stepping through the step sizes till you get yi(xi.w + b) close to 1
        # Step size from large to small for Gradient Descent. Step functions cannot be threaded or multi processed
        # In our example, max_feature_value = 8, step_sizes = [0.8, 0.08, 0.008]
        step_sizes = [self.max_feature_value * 0.1,                 # First ste size = 10% of the largest feature value
                      self.max_feature_value * 0.01,
                      self.max_feature_value * 0.001]               # point of expense

        b_range_multiple = 2                                        # Extremely expensive.
        b_multiple = 5                                              # No need to vary step size of b as we do with w
        latest_optimum = self.max_feature_value * 10                # First element in w vector e.g. 80

        for step in step_sizes:
            # # Start w at [latest_optimum, latest_optimum], e.g. [80, 80]
            w = np.array([latest_optimum, latest_optimum])
            optimized = False                                       # We can do this because ||w|| is convex
            while not optimized:
                # To get the maximum bias (b). This can be threaded
                for b in np.arange(-1 * (self.max_feature_value * b_range_multiple),    # e.g. (-16, 16, 4)
                                   self.max_feature_value * b_range_multiple,
                                   step * b_multiple):
                    for transformation in transforms:
                        # apply [1, 1], [-1, 1], [-1, -1], [1, -1] transformations to w. So w will go from -80 to +80
                        w_t = w * transformation
                        # Weakest link in the SVM fundamentally. Entire training data needs to be in memory.
                        # SMO attempts to fix this a bit
                        # #### add a break here later..
                        found_option = True
                        for i in self.data:
                            for xi in self.data[i]:
                                yi = i
                                if not yi * (np.dot(w_t, xi) + b) >= 1:                 # Constraint: yi(xi.w + b) >= 1
                                    # Even if one sample does not satisfy the constraint, that class can be ignored
                                    found_option = False
                                    # print(xi,':',yi*(np.dot(w_t,xi)+b))

                        if found_option:
                            # ||w|| : [w,b]
                            opt_dict[np.linalg.norm(w_t)] = [w_t, b]                    # get the magnitude of w_t

                # At this point we have run through all the w and b transformations
                if w[0] < 0:
                    optimized = True
                    print('Optimized a step.')
                else:
                    w = w - step

            norms = sorted([n for n in opt_dict])               # sort by magnitude if w to get the smallest ||w||
            # ||w|| : [w,b]
            opt_choice = opt_dict[norms[0]]                     # norms[0] has the smallest ||w||
            self.w = opt_choice[0]                              # Optimal w
            self.b = opt_choice[1]                              # Optimal b
            latest_optimum = opt_choice[0][0] + step * 2

        for i in self.data:
            for xi in self.data[i]:
                yi = i
                print(xi, ':', yi * (np.dot(self.w, xi) + self.b))

    def predict(self, features):
        # sign(x.w+b)
        classification = np.sign(np.dot(np.array(features), self.w) + self.b)
        if classification != 0 and self.visualization:
            self.ax.scatter(features[0], features[1], s=75, marker='*', c=self.colors[classification])
        return classification

    def visualize(self):
        [[self.ax.scatter(x[0], x[1], s=100, color=self.colors[i]) for x in data_dict[i]] for i in data_dict]

        # Embedded function to plot the decision boundary and support vector hyperplanes
        # Hyperplane = x.w+b
        # v = x.w+b (hyperplane values)
        # e.g. For positive and negative support vectors, v = 1 and -1, For decision boundary, v = 0
        def hyperplane(x, w, b, v):
            return (-w[0] * x - b + v) / w[1]

        # Adding 10% to min and max for plotting purpose
        datarange = (self.min_feature_value * 0.9, self.max_feature_value * 1.1)
        hyp_x_min = datarange[0]
        hyp_x_max = datarange[1]

        # Positive support vector hyperplane -> (w.x+b) = 1
        psv1 = hyperplane(hyp_x_min, self.w, self.b, 1)
        psv2 = hyperplane(hyp_x_max, self.w, self.b, 1)
        self.ax.plot([hyp_x_min, hyp_x_max], [psv1, psv2], 'k')

        # Negative support vector hyperplane -> (w.x+b) = -1
        nsv1 = hyperplane(hyp_x_min, self.w, self.b, -1)
        nsv2 = hyperplane(hyp_x_max, self.w, self.b, -1)
        self.ax.plot([hyp_x_min, hyp_x_max], [nsv1, nsv2], 'k')

        # Decision boundary hyperplane -> (w.x+b) = 0
        db1 = hyperplane(hyp_x_min, self.w, self.b, 0)
        db2 = hyperplane(hyp_x_max, self.w, self.b, 0)
        self.ax.plot([hyp_x_min, hyp_x_max], [db1, db2], 'y--')

        plt.show()


if __name__ == '__main__':
    data_dict = {-1: np.array([[1, 7], [2, 8], [3, 8]]),
                 1: np.array([[5, 1], [6, -1], [7, 3]])}

    svm = SupportVectorMachine()
    svm.fit(training_data=data_dict)

    predict_us = [[0, 10],
                  [1, 3],
                  [3, 4],
                  [3, 5],
                  [5, 5],
                  [5, 6],
                  [6, -5],
                  [5, 8]]

    for p in predict_us:
        svm.predict(p)

    svm.visualize()

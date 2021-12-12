"""
-----------------------------------------------------------------------------------------------------------------------
Gradient Descent:
-----------------------------------------------------------------------------------------------------------------------
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')


def gradient_descent(X, y):
    m_curr = b_curr = 0             # Start with initial m and b as 0
    iterations = 1000
    n = len(X)                      # number of data points in the feature set
    learning_rate = 0.005

    plot_line(x=X, m=m_curr, b=b_curr, color='r')                       # Plot the starting line in red

    for i in range(iterations):
        y_predicted = m_curr * X + b_curr                               # y = mx + b

        cost = (1/n) * sum([val**2 for val in (y-y_predicted)])         # Sum of squared error

        md = -(2/n) * sum(X * (y - y_predicted))                        # Partial derivative wrt m
        bd = -(2/n) * sum(y-y_predicted)                                # Partial derivative wrt b
        m_curr = m_curr - learning_rate * md                            # Take a step
        b_curr = b_curr - learning_rate * bd

        plot_line(x=X, m=m_curr, b=b_curr, color='g')                   # Plot (in green) as gradient descent runs
        # print("m {}, b {}, cost {} iteration {}".format(m_curr, b_curr, cost, i))

    plot_line(x=X, m=m_curr, b=b_curr, color='k')                       # Plot the best fit line in black


def plot_line(x, m, b, color):
    y = [m*x + b for x in range(1, 6)]
    plt.plot(x, y, color=color)


if __name__ == '__main__':
    X = np.array([1, 2, 3, 4, 5])
    y = np.array([5, 7, 9, 11, 13])

    gradient_descent(X, y)

    plt.scatter(X, y, s=50)
    plt.show()

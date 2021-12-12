from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random
style.use('ggplot')


# ---------------------------------------------------------------------------------------------------------------------
# Linear Regression
# ---------------------------------------------------------------------------------------------------------------------
# X and y must have some correlation
# ---------------------------------------------------------------------------------------------------------------------
def create_dataset(data_points, variance, step=2, correlation=True):
    # variance: how variable the dataset should be
    # step: how far on average to step up the y value per point
    # correlation: True: positive (step should be incremented), False: Negative (step should be decremented)
    val = 1     # initial value of y
    ys = []
    for i in range(data_points):
        ys.append(val + random.randrange(-variance, variance))
        if correlation and correlation == 'pos':
            val += step
        elif correlation and correlation == 'neg':
            val -= step
    xs = [i for i in range(data_points)]
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)


def best_fit_slope_and_intercept(xs, ys):
    # Function to calculate slope (m) and y-intercept (b)
    m = (mean(xs) * mean(ys) - mean(xs * ys)) / (pow(mean(xs), 2) - mean(pow(xs, 2)))
    b = mean(ys) - m * mean(xs)
    return m, b


def squared_error(ys_orig, ys_line):
    return sum((ys_line - ys_orig)**2)


def coefficient_of_determination(ys_orig, ys_line):
    # Coefficient of determination is 1 - (SE of regression line / SE of y-mean line)
    squared_error_regr = squared_error(ys_orig=ys_orig, ys_line=ys_line)
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    squared_error_y_mean = squared_error(ys_orig=ys_orig, ys_line=y_mean_line)
    return 1 - (squared_error_regr / squared_error_y_mean)


if __name__ == '__main__':
    xs, ys = create_dataset(data_points=40, variance=10, step=2, correlation='neg')
    print(xs, ys)

    # Build the prediction model (calculate m and b)
    m, b = best_fit_slope_and_intercept(xs=xs, ys=ys)
    print('Slope, y-intercept: ', m, b)

    regression_line = [(m*x) + b for x in xs]
    print('Regression Line: ', regression_line)

    # Predict
    predict_x = [i for i in range(40)]
    predict_y = [(m * x) + b for x in predict_x]

    # Accuracy (R-squared or coefficient of determination, calculated as squared error)
    # 1 - (SE of regression line / SE of y-mean line)
    # SE of regression line should ideally be close to 0 (as it is the best fit line)
    # R-squared should ideally be close to 1 for better fit
    r_squared = coefficient_of_determination(ys_orig=ys, ys_line=regression_line)
    print('Accuracy (R-Squared or Coefficient of Determination): ', r_squared)

    # Plot
    plt.scatter(xs, ys)
    plt.plot(xs, regression_line)
    plt.scatter(predict_x, predict_y, edgecolors='r')
    plt.show()

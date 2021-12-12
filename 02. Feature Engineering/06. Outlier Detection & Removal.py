"""
-----------------------------------------------------------------------------------------------------------------------
Outlier Detection and Removal:
-----------------------------------------------------------------------------------------------------------------------
1. Using Percentile
        Percentile is the value below which a percentage of data falls
        e.g. You can remove outliers that fall outside the 1st and 99th percentiles

2. Using Standard Deviation
        The Standard Deviation is a measure of how spread out numbers are
        e.g. You can remove outliers that fall outside +3 and -3 standard deviations

3. Using Z-Score
        Z-Score is how many standard deviations a value is from the mean
        e.g. You can remove outliers that fall outside Z-Score of +3 and -3 (or +3 and -3 standard deviations)

4. Using IQR
        The "Inter-Quartile Range" is from 25th percentile (Q1) to 75th percentile (Q3)
        e.g. You can remove outliers that fall outside of (Q1 - 1.5IQR) and (Q3 + 1.5IQR)
-----------------------------------------------------------------------------------------------------------------------
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from matplotlib import pyplot as plt

# ---------------------------------------------------------------------------------------------------------------------
# 1. Outlier detection and removal using Percentile:
# ---------------------------------------------------------------------------------------------------------------------
# if __name__ == '__main__':
#     df = pd.read_csv('heights.csv')                                                   # (10000, 2)
#     print(df.describe())
#
#     # get min and max thresholds at 0.1 and 99.9 percentiles:
#     min_threshold, max_threshold = df['height'].quantile([0.001, 0.999])
#     print(min_threshold, max_threshold)
#
#     # Print the rows that fall outside the threshold:
#     print(df[df.height > max_threshold])
#     print(df[df.height < min_threshold])
#
#     # Remove outliers:
#     df2 = df[(df.height < max_threshold) & (df.height > min_threshold)]         # (13172, 7)
#     print(df2.describe())


# ---------------------------------------------------------------------------------------------------------------------
# 2. Outlier detection and removal using Standard Deviation:
# ---------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    df = pd.read_csv('heights.csv')                                                 # (10000, 2)
    print(df.height.describe())

    # Plot the distribution:
    plt.figure(figsize=(11, 6))
    plt.tight_layout()

    # Histogram:
    plt.subplot(1, 2, 1)
    plt.hist(df.height, bins=20, color='#adad3b', rwidth=0.9, density=True)
    rng = np.arange(df.height.min(), df.height.max(), 0.1)
    plt.plot(rng, norm.pdf(rng, df.height.mean(), df.height.std()))
    plt.xlabel('Height (inches)')
    plt.ylabel('Count')
    # Boxplot:
    plt.subplot(1, 2, 2)
    df.boxplot(column='height', color='#5a7d9a')
    plt.show()

    # Define thresholds using 3-Standard Deviation:
    upper_limit = df.height.mean() + (3 * df.height.std())
    lower_limit = df.height.mean() - (3 * df.height.std())
    print(upper_limit, lower_limit)

    # Print samples that fall outside the thresholds:
    print(df[(df.height > upper_limit) | (df.height < lower_limit)])                # (7, 2)

    # Remove outliers:
    df2 = df[(df.height < upper_limit) & (df.height > lower_limit)]                 # (9993, 2)
    print(df2.describe())


# ---------------------------------------------------------------------------------------------------------------------
# 3. Outlier detection and removal using Z-Score:
# ---------------------------------------------------------------------------------------------------------------------
# if __name__ == '__main__':
#     df = pd.read_csv('heights.csv')                                                 # (10000, 2)
#
#     # Add a Z-Score column (it will have mean=0 and std=1):
#     df['z_score'] = (df.height - df.height.mean()) / df.height.std()
#     print(df.describe())
#
#     # Print samples that fall outside Z-scores (3 and -3):
#     print(df[(df.z_score > 3) | (df.z_score < -3)])                                 # (7, 3)
#
#     # Remove outliers:
#     df2 = df[(df.z_score < 3) & (df.z_score > -3)]                                  # (9993, 3)
#     print(df2.shape)
#     print(df2.describe())


# ---------------------------------------------------------------------------------------------------------------------
# 4. Outlier detection and removal using IQR:
# ---------------------------------------------------------------------------------------------------------------------

# if __name__ == '__main__':
#     df = pd.read_csv('heights.csv')                                                 # (10000, 2)
#     print(df.height.describe())
#
#     # Define Q1 (25th percentile) and Q3 (75th percentile):
#     Q1 = df.height.quantile(0.25)
#     Q3 = df.height.quantile(0.75)
#
#     # Calculate IQR
#     IQR = Q3 - Q1
#
#     # Define Upper and Lower limits using IQR
#     upper_limit = Q3 + 1.5*IQR
#     lower_limit = Q1 - 1.5*IQR
#     print(upper_limit, lower_limit)
#
#     # Print samples that fall outside the thresholds:
#     print(df[(df.height > upper_limit) | (df.height < lower_limit)])                # (8, 2)
#
#     # Remove outliers:
#     df2 = df[(df.height < upper_limit) & (df.height > lower_limit)]                 # (9992, 2)
#     print(df2.describe())

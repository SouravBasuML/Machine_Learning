"""
-----------------------------------------------------------------------------------------------------------------------
Mean, Mode, Median, Variance, Standard Deviation:
-----------------------------------------------------------------------------------------------------------------------
Mean (μ):
    Arithmetic Mean:
        Arithmetic Mean is the average (sum divided by the count)
    Geometric Mean:
        We multiply n numbers and take the nth root. It helps finding a value in between widely different values.
    Harmonic Mean:
        Harmonic mean is the reciprocal of the average of the reciprocals

Mode (Modal Value):
    Number that appears most often. The set of numbers can have two modes (bimodal) or multiple modes (multimodal)

Median:
    The Median is the "middle" of a sorted list of numbers. If there are two middle values, median is their average.

Variance:
    Variance is the average of the squared differences from the Mean. Calculate variance as follows:
        - Calculate the Mean (the simple average of the numbers)
        - Then for each number: subtract the Mean and square the result (the squared difference).
        - Then work out the average of those squared differences.
    When you have "N" data values that represent:
        - The Population (the entire dataset): divide by N when calculating Variance
        - A Sample (subset of the entire dataset): divide by N-1 when calculating Variance (this is done to compensate
            for measuring distances from the sample mean instead of the population mean)

Standard Deviation (σ):
    It is a measure of how spread out the measurements are around the mean. Its calculated as the sqrt of variance.
    Using standard Deviation, we can show which values are within one Standard Deviation of the Mean (i.e., we have a
    "standard" way of knowing what is normal, and what is extra large or extra small). We can expect about 68% of the
    measurements to be within plus-or-minus 1 SD, 95% within +/- 2 SD and so on.
        - 1 SD: 68.27%
        - 2 SD: 95.45%
        - 3 SD: 99.73%
        - 4 SD: 99.994%
        - 5 SD: 99.999_943%
        - 6 SD: 99.999_999_8%
        - 7 SD: 99.999_999_999_7%
-----------------------------------------------------------------------------------------------------------------------


-----------------------------------------------------------------------------------------------------------------------
Z-Score (Standard Score), Standardizing, Standard Normal Distribution:
-----------------------------------------------------------------------------------------------------------------------
Z-Score:
    Z-Score is how many standard deviations a measurement is from the mean. To convert a measurement to Standard Score:
        - first subtract the mean,
        - then divide by the standard deviation
                Z = (x - μ)/σ

Standardizing:
    The process of subtracting the mean from each measurement and dividing by the STD is called "Standardizing".
    Standardizing translates into 0 mean and unit STD; it helps us make better decisions about our data.

Standard Normal Distribution:
    A Normal Distribution with 0 mean and unit STD.
-----------------------------------------------------------------------------------------------------------------------


-----------------------------------------------------------------------------------------------------------------------
Histograms:
-----------------------------------------------------------------------------------------------------------------------
Stacking measurements into equal-sized bins gives us a Histogram. We can also draw a curve over the histogram to better
approximate how the probabilities of measurements are distributed.
We can use them to predict the probability of future measurements.
    - Using Histogram - number of samples in a range / total samples
    - Using Distribution Curve - Area under the curve for the range
-----------------------------------------------------------------------------------------------------------------------


-----------------------------------------------------------------------------------------------------------------------
Population and Estimated parameters:
-----------------------------------------------------------------------------------------------------------------------
Population represents all the samples. Parameters that represents a distribution are called population parameters.
e.g.
    - For normal distribution, population parameters are Population Mean and Population SD
    - For exponential distribution, population parameters are Population Rate
    - For gamma distribution, population parameters are Population Shape and Population Rate

'Sample' is a sample from the entire population. For population samples we estimate population parameters, (e.g.,
estimated/sample mean, estimated/sample SD etc.), which ensures that the results drawn from our experiment are
reproducible.
-----------------------------------------------------------------------------------------------------------------------


-----------------------------------------------------------------------------------------------------------------------
Statistical Distributions:
-----------------------------------------------------------------------------------------------------------------------
Statistical Distributions show us how the probabilities of measurements are distributed. Histograms and the Curve over
a Histogram are examples of Distributions.

Normal Distribution:
    Data can be "distributed" (spread out) in different ways. It can be spread out more on the left or right or jumbled
    up. But there are many cases where the data tends to be around a central value (mean) with no bias left or right,
    and it gets close to a "Normal Distribution" like a "Bell Curve".
        - High variance in data - curve will be wide and short
        - Low variance in data - curve will be narrow and tall
    The Normal Distribution has:
        - mean = median = mode
        - symmetry about the center
        - 50% of values less than the mean and 50% greater than the mean
        - area under the curve = 1
    Examples that follow normal distribution:
        - heights of people
        - size of things produced by machines
        - errors in measurements
        - blood pressure
        - marks on a test
        - commuting times
-----------------------------------------------------------------------------------------------------------------------


-----------------------------------------------------------------------------------------------------------------------
Percentile:
-----------------------------------------------------------------------------------------------------------------------
Percentile is the value below which a percentage of data falls.
e.g.
    1. The 20th percentile is the value (or score) below which 20% of the observations may be found.
    2. You are the fourth tallest person in a group of 20, i.e., 80% of people are shorter than you. That means you
        are at the 80th percentile. If your height is 1.85m then "1.85m" is the 80th percentile height in that group.
    3. You have scored 68% and you got 100 percentile. That means, you are the topper; there's no one who's scored more

When the data is grouped:
    Add up all percentages below the score, plus half the percentage at the score.
    e.g.
        In the test 12% got D, 50% got C, 30% got B and 8% got A. You got a B. Your percentile is 77% (12 + 50 + 15).
        That means, you did as well or better than 77% of the class

Deciles:
    Split the data into 10% groups:
        The 1st decile is the 10th percentile (the value that divides the data so 10% is below it)
        The 2nd decile is the 20th percentile (the value that divides the data so 20% is below it) etc.

Quartiles:
    Split the data into quarters or 25% groups:
        Quartile 1 (Q1) is the 25th percentile
        Quartile 2 (Q2) is the 50th percentile
        Quartile 3 (Q3) is the 75th percentile

Inter-Quartile Range (IQR):
    The "Inter-Quartile Range" is from Q1 (25th percentile) to Q3 (75th percentile).
    IQR = Q3 - Q1
-----------------------------------------------------------------------------------------------------------------------


-----------------------------------------------------------------------------------------------------------------------
Covariance and Correlation:
-----------------------------------------------------------------------------------------------------------------------
Covariance:
    Covariance is a measure to indicate the extent to which two random variables change in tandem. It signifies the
    direction of the linear relationship between the two variables. Covariance can vary between -∞ and +∞

Correlation:
    Correlation is a measure used to represent how strongly two random variables are related to each other. Correlation
    ranges between -1 and +1
-----------------------------------------------------------------------------------------------------------------------

"""
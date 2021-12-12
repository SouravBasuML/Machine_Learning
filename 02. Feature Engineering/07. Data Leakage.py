"""
-----------------------------------------------------------------------------------------------------------------------
Data Leakage:
-----------------------------------------------------------------------------------------------------------------------
Data leakage happens when your training data contains information about the target, but similar data will not be
available when the model is used for prediction. This leads to high performance on the training set (and possibly even
the validation data), but the model will perform poorly in production. There are two main types of leakage:
1. Target leakage and
2. Train-test contamination
3. Time-Series (Temporal) Data

-----------------------------------------------------------------------------------------------------------------------
1. Target Leakage:
-----------------------------------------------------------------------------------------------------------------------
Target leakage occurs when your predictors include data that will not be available at the time you make predictions.
It is important to think about target leakage in terms of the timing or chronological order that data becomes available,
not merely whether a feature helps make good predictions.
e.g.
    - medicine usage data included in disease prediction dataset
    - credit card limit/utilization (from the new credit card issued) included in training data
To prevent this, any variable updated (or created) after the target value is realized should be excluded.
-----------------------------------------------------------------------------------------------------------------------


-----------------------------------------------------------------------------------------------------------------------
2. Train-Test Contamination:
-----------------------------------------------------------------------------------------------------------------------
This type of leakage occurs when you aren't careful to distinguish training data from validation data. Validation is
meant to be a measure of how the model does on data that it hasn't considered before. You can corrupt this process in
subtle ways if the validation data affects the preprocessing behavior.
e.g.
     running preprocessing (like fitting an imputer for missing values, scaling) before calling train_test_split()

If your validation is based on a simple train-test split, exclude the validation data from any type of fitting,
including the fitting of preprocessing steps:
    - use fit_transform() on training data
    - use only transform() on testing/validation data

This is easier if you use scikit-learn pipelines. When using cross-validation, it's even more critical that you do
your preprocessing inside the pipeline!
-----------------------------------------------------------------------------------------------------------------------


-----------------------------------------------------------------------------------------------------------------------
3. Time-Series (Temporal) Data:
-----------------------------------------------------------------------------------------------------------------------
In time-series data there is a time-order of the samples; each sample is dependent on the previous sample in time.
Data leakage will happen if we randomly select train and test samples from the entire time-series data, as we are
removing the time-order from the samples.

Instead, we should select a particular point in time (t). Samples before 't' should be considered in training and
samples after 't' should be used for testing.
-----------------------------------------------------------------------------------------------------------------------
"""

"""
-----------------------------------------------------------------------------------------------------------------------
Gradient Boosting:
-----------------------------------------------------------------------------------------------------------------------
Gradient boosting is a machine learning technique used in regression and classification tasks, among others. It gives
a prediction model in the form of an ensemble of weak prediction models, which are typically decision trees.
When a decision tree is the weak learner, the resulting algorithm is called gradient-boosted trees; it usually
outperforms random forest.

Gradient Boost for Regression: When Gradient Boost is used to predict a continuous value.
Gradient Boost for Classification: When Gradient Boost is used to predict discreet values.

-----------------------------------------------------------------------------------------------------------------------
Summary:
-----------------------------------------------------------------------------------------------------------------------
Gradient boosting is a method that goes through cycles to iteratively add models into an ensemble. Ensemble methods
combine the predictions of several models (e.g., several trees, in the case of random forests).

1. It begins by initializing the ensemble with a single model, whose predictions can be pretty naive. (Even if its
    predictions are wildly inaccurate, subsequent additions to the ensemble will address those errors.)
2. Then, we start the cycle:
    a. First, we use the current ensemble to generate predictions for each observation in the training dataset. To make
        a prediction for a training sample, we add the predictions from all models in the ensemble.
    b. These predictions are used to calculate a loss function (e.g. mean squared error).
    c. Then, we use the loss function to fit a new model that will be added to the ensemble. Specifically, we determine
        model parameters so that adding this new model to the ensemble will reduce the loss. (Side note: The "gradient"
        in "gradient boosting" refers to the fact that we'll use gradient descent on the loss function to determine the
        parameters in this new model.)
    d. Finally, we add the new model to the ensemble, and repeat!
-----------------------------------------------------------------------------------------------------------------------


-----------------------------------------------------------------------------------------------------------------------
Gradient Boosting Algorithm for Regression:
-----------------------------------------------------------------------------------------------------------------------
1. Get Initial Prediction:
    Start by making a single leaf (prediction) instead of a tree or stump. This leaf represents an initial predicted
    guess for all the samples, e.g. the average value of the labels for a continuous output (regression).

2. Calculate Pseudo Residuals (using a Differentiable Loss Function):
    The errors that the previous tree (the initial leaf in this case) made, are the differences between the Actual
    label values and the Predicted values. This error is also called the Pseudo Residual. The residual is calculated
    using a Loss Function that is differentiable. The most common Loss Function for Gradient Boost is:

            Loss Function = 1/2 * (Observed - Predicted) ^ 2

    Note:
        In Linear Regression, the difference between the Observed and the Predicted values results in Residual
        In Gradient Boost, this difference is called Pseudo Residual

3. Build a first tree:
    Build a tree based on the errors of the first tree to predict the residuals. In case of the first iteration, the
    first tree is the initial leaf. Unlike AdaBoost, this tree is larger than a stump but Gradient Boost restricts the
    size of the tree (e.g., max leaf nodes between 8 and 32). If multiple rows end up in the same leaf of the tree
    (because of Tree size restriction), replace the leaf node value by their average.

4. Now combine the original leaf with the new tree, to make new predictions for the training data.
        New label (for the training data) = Original leaf prediction + Prediction from the New Tree

5. Scale the first Tree:
    Multiply the output of the tree with a 'Learning Rate' ('nu', a number between 0 and 1). This scaling of the output
    is done so that the model does not over fit the training data. With the learning rate, the prediction is not as good
    as the one without the learning rate, but little better than the previous prediction. This results in a small step
    in the right direction. This will help in low variance with test data.
        New label (for the training data) = Original leaf prediction + (Learning Rate * Prediction from the New Tree)

6. Build the next tree:
    a. Calculate new Pseudo Residual =  Observed label values - Latest prediction
            The new pseudo residuals will be smaller than the previous residuals
    b. Build a new tree to predict the new residuals. Note: The new tree will be of the same size as the first tree
        (i.e., it will have the same number of leaves) but it's branches may be different.
    c. If multiple rows end up in the same leaf, replace the leaf node value by their average.
    c. Scale the new tree by the same learning rate

7. Make new predictions for the training data:
    New label = Original leaf prediction + (LR * 1st Tree Prediction) + (LR * 2nd Tree Prediction)

8. Keep building fixed sized trees until the number of trees asked for (e.g. 50 or 100) or additional trees fail to
    improve the fit (i.e. additional trees does not significantly reduce the residuals)

9. Make Predictions:
    Prediction = Original leaf prediction + (LR * 1st Tree Prediction) + (LR * 2nd Tree Prediction) + ...
-----------------------------------------------------------------------------------------------------------------------


-----------------------------------------------------------------------------------------------------------------------
Gradient Boosting Algorithm for Classification:
-----------------------------------------------------------------------------------------------------------------------
1. Get Initial Prediction:

    Start by making a single leaf that represents an initial prediction for all the samples. In Gradient Boost for
    Classification, the initial prediction is the log(odds).
    e.g. if x samples have cancer and y samples do not, then log(odds) that someone has cancer is log(x/y)

    To use the log(odds) for classification, we convert it into a probability using the Logistic Function:
    e.g. Prob(someone has cancer) = e^log(odds) / (1 + e^log(odds))

    If the initial probability is > threshold (e.g. 0.5),
        we can classify that everyone in the dataset has cancer (this is the initial prediction)

2. Calculate Pseudo Residuals (using a Differentiable Loss Function):

        Loss Function = log(Likelihood of the observed data given the prediction)

        Residual = Observed probability - Predicted probability
            where:
                Observed values = 1 or 0 (the actual classifications)
                Predicted values = probability calculated in step 1

3. Build the first tree:
        Build a tree to predict the residuals. Since the size of the tree is restricted, multiple residuals may end up
        in a single leaf. Output value for a leaf is calculated as follows:
                E Residual(i) / E [Prev Prob(i) * (1 - Prev Prob(i))]

4. Scale the new tree and combine the original leaf with the new tree, to make new predictions for the training data.
        a. Get new log(odds) Prediction:
                New log(odds) Prediction = Original leaf prediction + (LR * Prediction from the new tree)
        b. Convert the new log(odds) Prediction into Probability. This new prob is a small step in the right direction
                Prob = e^log(odds) / (1 + e^log(odds))

5. Build the next tree:
    a. Calculate new Pseudo Residual =  Observed probabilities (1 or 0) - Latest predicted probabilities
    b. Build a new tree to predict the new residuals.
    c. Calculate the output value for a leaf:
            E Residual(i) / E [Prev Prob(i) * (1 - Prev Prob(i))]
    d. Scale the new tree by the same learning rate
    e. Make new predictions for the training data (Step 4)

7. Keep building fixed sized trees until the number of trees asked for or additional trees fail to
    improve the fit (i.e. additional trees does not significantly reduce the residuals)

9. Make Predictions:
    a. log(odds) Prediction = Original leaf prediction + (LR * 1st Tree Prediction) + (LR * 2nd Tree Prediction) + ...
    b. Prob = e^log(odds) / (1 + e^log(odds))
    c. Make classification based on whether the probability is more or less than the threshold
-----------------------------------------------------------------------------------------------------------------------
"""

"""
-----------------------------------------------------------------------------------------------------------------------
Bias:
-----------------------------------------------------------------------------------------------------------------------
How accurately a model can capture a pattern in a training dataset. It is the inability for an ML algorithm to
capture the  true relationship between its independent and dependent variables.

High Bias: High train error (under fit), Low variance
           Such model may give good predictions, not great predictions; but the predictions will be consistent
Low Bias: Low train error, high test error (over fit), High variance
-----------------------------------------------------------------------------------------------------------------------


-----------------------------------------------------------------------------------------------------------------------
Variance:
-----------------------------------------------------------------------------------------------------------------------
Variance is the difference in fits between datasets.

High variance: High test error and it varies greatly based on the selection of the test dataset. It may not perform
               well on future data.
Low variance: Test error remains mostly same for different test sets.
-----------------------------------------------------------------------------------------------------------------------


-----------------------------------------------------------------------------------------------------------------------
Bias-Variance Tradeoff:
-----------------------------------------------------------------------------------------------------------------------
Fitting the training data well but making poor predictions is called the Bias-Variance Tradeoff.
Techniques to find the best model (low bias and low variance)
1. Regularization
2. Boosting
3. Bagging
-----------------------------------------------------------------------------------------------------------------------


-----------------------------------------------------------------------------------------------------------------------
Bulls Eye Diagram:
-----------------------------------------------------------------------------------------------------------------------
Low Bias, Low Variance - Data points are concentrated and are near the bulls eye
Low Bias, High Variance - Data points are spread out around, but are near the bulls eye
High Bias, Low Variance - Data points are concentrated, but are away from the bulls eye
High Bias, High Variance - Data points are spread out, and are away from the bulls eye
-----------------------------------------------------------------------------------------------------------------------
"""

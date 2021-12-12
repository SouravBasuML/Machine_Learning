"""
-----------------------------------------------------------------------------------------------------------------------
Adaptive Boosting (AdaBoost):
-----------------------------------------------------------------------------------------------------------------------
AdaBoost algorithm, short for Adaptive Boosting, is a Boosting technique used as an Ensemble Method in Machine Learning.
It is called Adaptive Boosting as the weights are re-assigned to each instance, with higher weights assigned to
incorrectly classified instances. AdaBoost is most commonly used with Decision Trees.

-----------------------------------------------------------------------------------------------------------------------
Concepts:
-----------------------------------------------------------------------------------------------------------------------
1. AdaBoost creates a "Forest of Stumps" i.e. trees with a root node and leaf nodes. Stumps are "Weak Learners", which
   avoid over-fitting. AdaBoost combines multiple weak learners to make predictions ( in contrast, Random forest is a
   forest of full sized trees).
2. In a Forest of Stumps made with AdaBoost, some stumps get more say in the final classification than others
    (in contrast, in RF, each tree has an equal vote on the final classification).
3. In a Forest of Stumps made with AdaBoost, the order of the stumps is important. The errors made by the first stump
   influence how the second stump is made, and so on (in contrast, in RF, each DT is made independently of the others).
-----------------------------------------------------------------------------------------------------------------------

-----------------------------------------------------------------------------------------------------------------------
AdaBoost Algorithm Steps (Create a Forest of Stumps):
-----------------------------------------------------------------------------------------------------------------------
1. Assign Sample Weights:
    Each sample in the feature set is given a weight. The weight indicates how important it is to be correctly
    classified. Initially all samples get the same weight (1/number of samples), i.e., initially all samples are
    equally important. However, after we make the first stump, these weights will change in order to guide how the next
    stump is created.

2. Create the first stump:
    a. Create one stump for each feature.
    b. Calculate the Gini index for each stump
    c. Select the feature as the root of the first stump that does the best job classifying the samples, i.e., the
       stump with the least Gini index (impurity)

3. Calculate "Amount of Say" for the first stump:
    Determine how much say the first stump will have in the final classification, based on how well it classified the
    samples:

    a. Calculate the "Total Error" for the first stump as the sum of weights associated with the incorrectly classified
       samples.
            Total Error = 0 (for a perfect stump)
            Total Error = 1 (for the most imperfect stump)

    b. Calculate the "Amount of Say" this stump has in the final classification:

            Amount of Say = 1/2 loge((1 - Total Error) / Total Error)

        If Total Error ~ 0:
            Stump consistently gives the correct classification
            Amount of Say is a large positive value
        If Total Error = 0.5:
            Half the samples are correctly classified by the stump and half are incorrectly classified
            Amount of Say = 0
        If Total Error ~ 1:
            Stump consistently gives the opposite classification
            Amount of Say is a large negative value

        Since we selected the stump that has the least Gini index, the Total Error for this stump will be the least and
        the Amount of Say will be the most, among all other stumps created from the other features.

4. Modify the weights (so that the next stump takes into account the errors that the current stump made):
    Increase the weight of the sample that was incorrectly classified by the first stump, and decrease the sample
    weights of the other samples that were correctly classified.

    a. New Sample Weight (for incorrectly classified samples):

                Old sample weight * e ^ (Amount of Say)

            If Amount of Say was large (last stump did a good job in classifying the samples):
                    New sample weight will be much larger than the old one
            If Amount of Say was small (last stump did not do a good job in classifying the samples):
                    New sample weight will be little larger than the old one

    b. New Sample Weight (for correctly classified samples):

                Old sample weight * e ^ (-Amount of Say)

            If Amount of Say was large (last stump did a good job in classifying the samples):
                    New sample weight will be very small
            If Amount of Say was small (last stump did not do a good job in classifying the samples):
                    New sample weight will be little smaller than the old one

5. Normalize the new sample weights by dividing the new sample weights by the sum of all the new sample weights
    (so they add up to one)

6. Create a new feature set (of the same size as the original feature set), that contains duplicate copies of the
    samples with the largest sample weights
    a. Pick a random number between 0 and 1
    b. Using the modified weights as distribution, pick samples from the original feature set into the new feature set.
        Feature samples that have larger weights (that were incorrectly classified by the first stump), will have a
        greater probability of getting selected into the new feature set.

7. Create the second stump in the forest:
    Use the new feature set, assign them initial weights (1/number of samples) and repeat the entire procedure.

    The duplicate copies of the samples (which are the misclassified samples using the first stump) will be treated as
    a block, creating a large penalty for being misclassified. This is how the errors that one stump makes, influence
    how the next stump is made.

8. The procedure is repeated 'n_estimators' times or until AdaBoost has a perfect fit.

-----------------------------------------------------------------------------------------------------------------------
Make Predictions:
-----------------------------------------------------------------------------------------------------------------------
1. Run the new sample through all the stumps
2. Add up the "Amount of Say" of the stumps for each type of classification and
3. Select the classification that has the largest sum of Amounts of Say.
-----------------------------------------------------------------------------------------------------------------------
"""

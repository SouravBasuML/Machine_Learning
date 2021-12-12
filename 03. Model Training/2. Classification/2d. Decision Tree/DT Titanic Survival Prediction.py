import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import model_selection

# ---------------------------------------------------------------------------------------------------------------------
# Decision Tree:
# ---------------------------------------------------------------------------------------------------------------------
# Characteristics:
# 1. True is generally on the left side of the node, while False is on the right.
# 2. Same tree may have combination data types in the branches (numeric, boolean etc.)
# 3. Numeric thresholds can be different for the same data.
# 4. Final classifications (leaf nodes) can be repeated.
# ---------------------------------------------------------------------------------------------------------------------
# Node Impurity:
# Measure of the homogeneity of the labels at the node. An impure node contains a mixture of the labels. If a node
# classifies one label fully, it is a pure node. Methods to quantify Impurity:
# 1. Gini Impurity
# 2. Entropy or Information Gain
# ---------------------------------------------------------------------------------------------------------------------
# Gini Impurity:
# Gini Impurity of a Yes/No Node: 1 - (prob of YES)^2 - (prob of NO)^2
# Total Gini Impurity: Weighted Average of Gini Impurities
# Gini Impurity for a numeric column is calculated by sorting the column, getting the average of adjacent values and
# calculating Gini Impurity for each of the average values. The lowest impurity is picked.
# ---------------------------------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------------------------------
# Classification Tree: When a Decision Tree classifies samples into categories
# ---------------------------------------------------------------------------------------------------------------------
# The first step is to determine the root node. The Gini Impurity for all the features (columns) is calculated and
# the feature with the lowest Gini Impurity is placed at the root node. This step is repeated to identify the next
# levels of the decision tree. The process stops when a node does not have any impurity (it becomes the leaf node) or
# if its Gini Impurity is lower than its children (it becomes the leaf node). If a feature does not reduce impurity
# score, we will not use that feature to build the decision tree.

# If a pure node has very few samples for a particular label, the Tree is over-fit. To overcome over-fitting:
# 1. We can prune the decision tree
# 2. We can put limits on samples per leaf node which can be determined using cross-validation
# 3. We can put limits on impurity reduction to determine if we need to create children or designate the node as leaf

# Missing Data:
# 1. Use the majority value or mean/median for the column
# 2. Identify another column with the highest correlation with the column with missing data and use that as a guide to
#    update missing data.

# ---------------------------------------------------------------------------------------------------------------------
# Regression Tree:
# ---------------------------------------------------------------------------------------------------------------------
# When a Decision Tree predicts numeric values (each leaf represents a numeric value).
# https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
#
# Root Node is selected by selecting different thresholds for each feature, calculating the sum of squared
# residuals at each step, and picking the feature with a threshold that gives the minimum sum of squared residuals.
# The process stops when a node has less than a minimum number of observations (e.g. 20), in which case the node is
# designated as a leaf node; otherwise the process is repeated to split the remaining observations to identify
# intermediate nodes.
# ---------------------------------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------------------------------
# Pruning Regression Trees: Cost Complexity Pruning aka Weakest Link Pruning
# ---------------------------------------------------------------------------------------------------------------------
# Pruning is used to prevent over fitting a regression tree by removing some leaf nodes and replacing the split with
# a leaf that is the average of a large number of observations. Cost Complexity Pruning provides a way to decide to
# what level to prune the tree for optimal fit. This is achieved by:
# 1. Calculating the Sum of Squared Residuals for each pruned tree, starting with the full sized tree. The SSR keeps
#    increasing as the tree is pruned.
# 2. Calculating Tree Score for each tree.
#    Tree Score = SSR + (alpha * Tree Complexity Penalty)
#    Tree Complexity Penalty is a function of the number of leaves or Terminal nodes in a tree or subtree, and it
#    compensates for the difference in the number of leaves. More the leaves, larger the tree complexity penalty
#    'alpha' is a tuning parameter that can be found using cross-validation.
# 3. Pick the sub tree (pruned tree) that has the lowest Tree Score
# ---------------------------------------------------------------------------------------------------------------------


if __name__ == '__main__':
    df = pd.read_csv('titanic.csv')
    df.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked'], axis='columns', inplace=True)

    X = df.drop('Survived', axis='columns')
    y = df['Survived']

    # Encode Sex column to numeric
    X['Sex'] = X['Sex'].map({'male': 1, 'female': 2})
    # Replace missing ages by mean of the Age column
    X['Age'] = X['Age'].fillna(X['Age'].mean())

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    print('Accuracy: ', clf.score(X=X_test, y=y_test))

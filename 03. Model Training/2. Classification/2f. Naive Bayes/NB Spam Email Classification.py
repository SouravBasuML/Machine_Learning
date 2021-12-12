import pandas as pd
from sklearn import model_selection
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------------------------------------------------------------------
# Naive Bayes
# ---------------------------------------------------------------------------------------------------------------------
# Naive Bayes methods are a set of supervised learning algorithms based on applying Bayes’ theorem with the “naive”
# assumption of conditional independence between every pair of features given the value of the class variable (label).
# They require a small amount of training data. Naive Bayes learners and classifiers can be extremely fast compared
# to more sophisticated methods. On the flip side, although naive Bayes is known as a decent classifier, it is known
# to be a bad estimator, so the probability outputs from predict_proba are not to be taken too seriously.
# ---------------------------------------------------------------------------------------------------------------------
# Bayes Theorem: P(A/B) = (P(B/A) * P(A))/ P(B)
# ---------------------------------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------------------------------
# Multinomial Naive Bayes:
# ---------------------------------------------------------------------------------------------------------------------
# Used when we have discrete data which can be represented in terms of their frequency of occurrence
# e.g. word count, movie rating etc.
# Naive Bayes does not differentiate between the order between every pair of features (words). e.g. the probability
# score for "Dear Friend" is same as "Friend Dear". NB ignores all language rules and word dependencies and treats
# every word in a language the same. NB has high bias (as it ignores dependencies  between words) and low variance
# (as it works well in practice)
# ---------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------------------------------
# Gaussian Naive Bayes:
# ---------------------------------------------------------------------------------------------------------------------
# Used where the features are continuous, i.e. features cannot be represented in terms of their occurrences.
# Also the feature set forms a normal distribution e.g. Iris dataset
# ---------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------------------------------
# Bernoulli Naive Bayes:
# ---------------------------------------------------------------------------------------------------------------------
# Used when all our feature values are binary (0s and 1s)
# ---------------------------------------------------------------------------------------------------------------------


if __name__ == '__main__':
    df = pd.read_csv('email spam.csv')
    print(df.groupby(['category']).count())

    # Convert 'category' column to numeric. 1: Spam, 0: Ham
    # df['spam'] = df['category'].map({'spam': 1, 'ham': 0})
    df['spam'] = df['category'].apply(lambda x: 1 if x == 'spam' else 0)
    print(df.head())

    X = df['message']
    y = df['spam']

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

    # Vectorize the 'message' column
    vectorizer = CountVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train.values)
    # print(X_train_vectorized.toarray()[: 3])

    # Train:
    clf = MultinomialNB()
    clf.fit(X_train_vectorized, y_train)

    # Predict:
    emails = [
        'Hey mohan, can we get together to watch football game tomorrow?',
        'Upto 20% discount on parking, exclusive offer just for you. Dont miss this reward!'
    ]
    emails_vectorized = vectorizer.transform(emails)
    print('Prediction: ', clf.predict(emails_vectorized))

    # Accuracy:
    X_test_vectorized = vectorizer.transform(X_test)
    print('Accuracy : ', clf.score(X_test_vectorized, y_test))

    # Pipeline: Create a Pipeline of transforms with a final estimator
    clf = Pipeline([
        ('vectorizer', CountVectorizer()),                  # transformer
        ('nb', MultinomialNB())                             # estimator
    ])

    # We can now train directly on un-vectorized text data, as the pipeline will internally convert the text data
    # to vector and then apply the Naive Bayes classifier
    clf.fit(X_train, y_train)
    print('Prediction: ', clf.predict(emails))
    print('Accuracy : ', clf.score(X_test, y_test))

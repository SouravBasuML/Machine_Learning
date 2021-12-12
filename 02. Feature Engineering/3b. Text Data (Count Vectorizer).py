"""
-----------------------------------------------------------------------------------------------------------------------
Text Count Vectorizer:
-----------------------------------------------------------------------------------------------------------------------
sklearn.feature_extraction.text.CountVectorizer
-----------------------------------------------------------------------------------------------------------------------
Convert a collection of text documents to a matrix of token counts. If you do not provide an a-priori dictionary and
you do not use an analyzer that does some kind of feature selection then the number of features will be equal to the
vocabulary size found by analyzing the data.
-----------------------------------------------------------------------------------------------------------------------

"""

from sklearn.feature_extraction.text import CountVectorizer


corpus = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
    'Is this the first document?',
]
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)

print(vectorizer.get_feature_names_out())
# ['and', 'document', 'first', 'is', 'one', 'second', 'the', 'third', 'this']

# Prints a 4x9 matrix. 4 items in corpus list, 9 unique words in corpus
print(X.toarray())

vectorizer2 = CountVectorizer(analyzer='word', ngram_range=(2, 2))
X2 = vectorizer2.fit_transform(corpus)

print(vectorizer2.get_feature_names_out())
# ['and this', 'document is', 'first document', 'is the', 'is this', 'second document',
#  'the first', 'the second', 'the third', 'third one', 'this document', 'this is', 'this the']

print(X2.toarray())

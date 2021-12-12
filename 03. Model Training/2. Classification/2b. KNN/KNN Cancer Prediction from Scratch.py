import pandas as pd
import numpy as np
import warnings
import random
from collections import Counter


def k_nearest_neighbors(training_data, data_to_predict, k=3):
    if len(training_data) >= k:
        warnings.warn('K is set to a value less than total voting groups!')

    distances = []
    for group in training_data:
        for features in training_data[group]:
            euclidean_distance = np.linalg.norm(np.array(features) - np.array(data_to_predict))
            distances.append([euclidean_distance, group])

    votes = [i[1] for i in sorted(distances)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]           # most voted class
    confidence = Counter(votes).most_common(1)[0][1] / k        # number of votes / k

    return vote_result, confidence


if __name__ == '__main__':

    df = pd.read_csv('breast-cancer-wisconsin.data')
    df.replace('?', -99999, inplace=True)           # Model will treat -99999 as an outlier and will ignore
    df.drop(['id'], axis=1, inplace=True)           # If you don't drop id, accuracy drops to below 60%
    full_data = df.astype(float).values.tolist()    # To convert everything into float and df into a list of lists
    random.shuffle(full_data)                       # Shuffle

    test_size = 0.2
    # train_set and test_set are dictionaries as our version of KNN needs input as dictionaries
    train_set = {2: [], 4: []}
    test_set = {2: [], 4: []}
    train_data = full_data[: -int(test_size*len(full_data))]    # everything up to the last 20%
    test_data = full_data[-int(test_size*len(full_data)):]      # the last 20%

    # Populate the train and test data dictionaries
    # i[-1] is the last element (column) of the samples, i.e. the result (2: benign, 4: malignant)
    # append(i[:-1] will append all elements up to the last element
    for i in train_data:
        train_set[i[-1]].append(i[:-1])

    for i in test_data:
        test_set[i[-1]].append(i[:-1])

    correct = total = 0                         # To calculate Accuracy

    for group in test_set:
        for data in test_set[group]:
            # k = 5 is the default in sklearn for KNN
            vote, confidence = k_nearest_neighbors(training_data=train_set, data_to_predict=data, k=5)
            if group == vote:
                correct += 1
            else:
                # If 1.0 that means all the 5 votes were incorrect
                print('Confidence when votes are incorrect: ', confidence)
            total += 1

    print('Accuracy:', correct/total)

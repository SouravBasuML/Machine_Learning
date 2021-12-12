import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import preprocessing

# ---------------------------------------------------------------------------------------------------------------------
# Titanic Dataset Metadata:
# ---------------------------------------------------------------------------------------------------------------------
# pclass: Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)
# survival: Survival (0 = No; 1 = Yes)
# name: Name
# sex: Sex
# age: Age
# sibsp: Number of Siblings/Spouses Aboard
# parch: Number of Parents/Children Aboard
# ticket: Ticket Number
# fare: Passenger Fare (British pound)
# cabin: Cabin
# embarked: Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)
# boat: Lifeboat
# body: Body Identification Number
# home.dest: Home/Destination
# ---------------------------------------------------------------------------------------------------------------------


def handle_non_numerical_data(df):
    # For non-numeric data, get the unique values in the column in a set, assign integer values to them 0, 1, 2, ...
    # Replace non-numeric data with its integer equivalent
    columns = df.columns.values
    for column in columns:
        text_digit_vals = {}

        def convert_to_int(val):
            return text_digit_vals[val]                             # e.g. this will return 0 for 'female' etc.

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()            # convert the entire column into a list
            unique_elements = set(column_contents)                  # convert the list to a set to get unique values

            # Convert the unique values to a dictionary e.g. {'female': 0, 'male': 1}
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x += 1

            # Map the non-numeric value in the column to numeric value in the dict using map function
            df[column] = list(map(convert_to_int, df[column]))

    return df


if __name__ == '__main__':
    df = pd.read_csv('titanic.csv')
    pd.set_option('display.max_columns', 14)

    # Drop columns that does not add value to modelling
    df.drop(['body', 'name', 'ticket', 'boat'], axis=1, inplace=True)

    # Replace unavailable columns values with 0
    df.fillna(0, inplace=True)

    # Convert non-numeric data to numeric
    df = handle_non_numerical_data(df)

    X = np.array(df.drop(['survived'], axis=1).astype(float))       # convert values to float
    X = preprocessing.StandardScaler().fit_transform(X)
    y = np.array(df['survived'])

    clf = KMeans(n_clusters=2)
    clf.fit(X)

    print('Actual Survival Results (y):', y)
    print('Predicted Survival Results (clf.labels_): ', clf.labels_)

    # Accuracy will oscillate between say 30% and 70% depending on the centroids chosen arbitrarily by the algorithm
    # In either case accuracy will  be the higher % in this case 70%

    # Accuracy can be calculated in two ways:
    # 1. By comparing y and clf.labels_
    correct = 0
    for i in range(len(y)):
        if y[i] == clf.labels_[i]:
            correct += 1
    print('Accuracy: ', correct/len(y))

    # # 2. By using clf.predict on X
    # correct = 0
    # for i in range(len(X)):
    #     predict_me = np.array(X[i].astype(float))
    #     predict_me = predict_me.reshape(-1, len(predict_me))
    #     prediction = clf.predict(predict_me)
    #     if prediction[0] == y[i]:
    #         correct += 1
    # print('Accuracy Method 1: ', correct/len(X))

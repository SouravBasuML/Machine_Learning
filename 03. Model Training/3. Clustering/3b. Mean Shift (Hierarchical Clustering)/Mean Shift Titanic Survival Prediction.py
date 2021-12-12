import numpy as np
import pandas as pd
from sklearn.cluster import MeanShift
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

    # Create of copy of the data frame to textually compare the output of Mean Shift
    original_df = pd.DataFrame.copy(df)

    # Drop columns that does not add value to modelling
    df.drop(['body', 'name'], axis=1, inplace=True)
    # df.drop(['body', 'name', 'ticket', 'boat'], axis=1, inplace=True)

    # Replace unavailable columns values with 0
    df.fillna(0, inplace=True)

    # Convert non-numeric data to numeric
    df = handle_non_numerical_data(df)

    X = np.array(df.drop(['survived'], axis=1).astype(float))       # convert values to float
    X = preprocessing.StandardScaler().fit_transform(X)
    y = np.array(df['survived'])

    clf = MeanShift()
    clf.fit(X)

    labels = clf.labels_
    cluster_centers = clf.cluster_centers_
    n_clusters_ = len(np.unique(labels))                            # 3
    print('Unique Cluster Labels: ', np.unique(labels))             # [0 1 2]

    # Add labels as a new column to the original df
    original_df['cluster_group'] = np.nan
    for i in range(len(labels)):
        original_df['cluster_group'].iloc[i] = labels[i]

    # Dictionary of {cluster group: survival rate}
    survival_rates = {}
    for i in range(n_clusters_):
        # Create a temp df to hold data for each cluster group (or label) identified by i
        temp_df = original_df[(original_df['cluster_group'] == float(i))]

        # Within a cluster group identify the survivors
        survival_cluster = temp_df[(temp_df['survived'] == 1)]

        # Survival rate: number of people survived in each cluster / total number of people in each cluster
        survival_rate = len(survival_cluster) / len(temp_df)
        survival_rates[i] = survival_rate

    print('Survival Rates of each cluster: ', survival_rates)

    # Mean Shift mostly classifies the data set into 3 groups of survivors. Each group roughly translates to
    # the class the passenger was travelling. e.g.
    # 1st Class: Survival rate: 88%
    # 2nd Class: Survival rate: 38% (e.g. mean(pclass) = 2.3, which implies most are from 2nd class)
    # 3rd Class: Survival rate: 10%

    print(original_df[(original_df['cluster_group'] == 0)].describe())      # 2nd class
    print(original_df[(original_df['cluster_group'] == 1)].describe())      # 1st class
    print(original_df[(original_df['cluster_group'] == 2)].describe())      # 3rd class

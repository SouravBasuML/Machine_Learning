"""
-----------------------------------------------------------------------------------------------------------------------
Exploratory Data Analysis (EDA) using Pandas:
-----------------------------------------------------------------------------------------------------------------------
Missing Values
Numerical Variables
Distribution of the Numerical Variables
Categorical Variables
Cardinality of Categorical Variables
Outliers
Relationship between independent and dependent features
-----------------------------------------------------------------------------------------------------------------------
"""

import pandas as pd
import numpy as np

"""
-----------------------------------------------------------------------------------------------------------------------
Dataframe:
-----------------------------------------------------------------------------------------------------------------------
"""
df = pd.DataFrame()
pd.set_option('display.max_columns', 10)
print(df.dtypes)                                        # Get the datatype of all the columns
df.describe()
df['col name'].describe()
df.astype(float)                                        # Convert all values to float
df['col name'].values.tolist()                          # Create a list from data frame column
df.drop(['col name'], axis='columns', inplace=True)     # Drop a column


"""
-----------------------------------------------------------------------------------------------------------------------
Identify Uniques:
-----------------------------------------------------------------------------------------------------------------------
Do this for all columns to see if column contains spaces/NA/? etc.
-----------------------------------------------------------------------------------------------------------------------
"""
df.nunique()                                            # Get number of unique values in each column
df['col name'].unique()                                 # Gets a list of unique values in the column


"""
-----------------------------------------------------------------------------------------------------------------------
Label Imbalance:
-----------------------------------------------------------------------------------------------------------------------
If data is imbalanced, use train_test_split(stratify=y) to preserve the balance in train and test datasets
-----------------------------------------------------------------------------------------------------------------------
"""
df['col name'].value_counts()                           # Count of each unique value in the column (Imbalance in data)
df['col name'].value_counts().to_dict()                 # Save it in a dictionary. Useful later for mapping
# Find the top 10 most frequent categories for the feature:
df['col name'].value_counts().sort_values(ascending=False).head(10)


"""
-----------------------------------------------------------------------------------------------------------------------
Aggregation:
-----------------------------------------------------------------------------------------------------------------------
count(), sum(), min(), max(), avg()
-----------------------------------------------------------------------------------------------------------------------
"""
df['new col'] = df['col names'].sum(axis=1)             # Sum up all the col names into a new column
df['new col'] = df['col names'].gt(0).sum(axis=1)       # Sum up all the col names whose value is > 0 into a new column


"""
-----------------------------------------------------------------------------------------------------------------------
Group Transforms (Group-by column name):
-----------------------------------------------------------------------------------------------------------------------
"""
df.groupby(['col name']).count()
df.groupby(['col name']).mean()
df.groupby(['col name']).describe()
df.groupby(['col1'])['col2'].mean()                     # col1: Grouping feature; col2: aggregated feature
df.groupby(['col'])['col'].count() / df['col'].count()  # Calculate frequency

X_train = pd.DataFrame()
X_test = pd.DataFrame()

# Apply aggregate function on train data:
X_train['new col'] = X_train.groupby('col1')['col2'].transform('mean')

# Merge the values into the validation set
X_test = X_test.merge(X_train[['col1', 'new col']].drop_duplicates(), on='col1', how='left')


"""
-----------------------------------------------------------------------------------------------------------------------
NaN/Missing Data preprocessing:
-----------------------------------------------------------------------------------------------------------------------
"""
print(df.columns[df.isna().any()])                      # Check if any columns have NA, NaN, None
df.isnull().any()                                       # Prints columns that have null
df.isnull().sum()                                       # Prints total number of rows in each column that have null
df.isna().any()                                         # Prints columns that have NaN
df.isna().sum()                                         # Prints total number of rows in each column that have NaN
df['col name'] = np.nan                                 # Add a column to the data frame initialized with NaN

# Handling NaN/Missing Data:
df.dropna(axis='columns', inplace=True)                 # Drop columns that have NaN
df.dropna(axis=0, subset=['col name'], inplace=True)    # Drop rows with missing data
df.fillna(-99999, inplace=True)                         # Replace NaN with -99999
df.replace('?', -99999, inplace=True)                   # Replace ? with -99999

# Replace NaN with the value that comes directly after it in the same column, then replace all the remaining NA's with 0
df.fillna(method='bfill', axis=0).fillna(0)

# Identify columns with missing data (NaN):
cols_with_nan = [col for col in df.columns if df[col].isnull().any()]

# Replace missing data in a column with average value of the same column (Imputation):
df['col name'] = df['col name'].fillna(df['col name'].mean())


"""
-----------------------------------------------------------------------------------------------------------------------
Percentile:
-----------------------------------------------------------------------------------------------------------------------
"""
df['col name'].quantile(0.9)                            # Returns a value which is at 90th percentile
df['col name'].quantile([0.25, 0.5, 0.75])              # Returns values at 25th, 50th, and 75th percentiles


"""
-----------------------------------------------------------------------------------------------------------------------
Correlation:
-----------------------------------------------------------------------------------------------------------------------
Find correlation of set of features with the target
-----------------------------------------------------------------------------------------------------------------------
"""
df['col names'].corrwith(df['target col'])


"""
-----------------------------------------------------------------------------------------------------------------------
Columns with Text Data:
-----------------------------------------------------------------------------------------------------------------------
"""
# Convert text column to numeric using map()/apply(). 1: spam, 0: ham
df['new col'] = df['old col'].apply(lambda x: 1 if x == 'spam' else 0)
df['new col'] = df['old col'].map({'spam': 1, 'ham': 0})

# Identify columns with text:
s = (df.dtypes == 'object')
object_cols = list(s[s].index)

# Remove columns with string/text data types:
df = df.select_dtypes(exclude=['object'])


"""
-----------------------------------------------------------------------------------------------------------------------
Apply Transforms:
-----------------------------------------------------------------------------------------------------------------------
If the feature has 0.0 values, use np.log1p (i.e., log(1+x)) instead of np.log (i.e. log(x) '''
-----------------------------------------------------------------------------------------------------------------------
"""
df['col name'] = np.log(df['col name'])                                 # Logarithmic Transformation
df['col name'] = df['col name'].apply(np.log1p)

# Split a text column into multiple columns:
df[['col1', 'col2']] = df['col name'].str.split(' ', expand=True)        # String Transform

# Join multiple text columns into one column:
df['new col'] = df['col1'] + '_' + df['col2']


"""
-----------------------------------------------------------------------------------------------------------------------
Parsing Dates:
-----------------------------------------------------------------------------------------------------------------------
Help Python recognize dates as composed of day, month, and year. https://strftime.org/
-----------------------------------------------------------------------------------------------------------------------
"""
# Converts dtypes 'Object' to 'datetime64'
df['date col'] = pd.to_datetime(df['date col'], format='%m/%d/%y')

# Lets pandas determine the correct date format in case there are multiple date formats in the input feature set.
# However, this method is not efficient compared to specifying the format
df['date col'] = pd.to_datetime(df['date col'], infer_datetime_format=True)

# Extract day:
day = df['date col'].dt.day

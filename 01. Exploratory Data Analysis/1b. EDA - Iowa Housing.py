"""
-----------------------------------------------------------------------------------------------------------------------
Exploratory Data Analysis (EDA) of Iowa Housing Price Dataset:
-----------------------------------------------------------------------------------------------------------------------
1. Missing Values
2. Relationship between independent and dependent features
3. Numerical Features
        Cardinality of numeric features
        Distribution of continuous numeric features (numeric features having high cardinality)
4. Temporal Features (Date, Month, Year, TS)
5. Categorical Features
        Cardinality of categorical features
6. Outliers

-----------------------------------------------------------------------------------------------------------------------
"""

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
plt.style.use('ggplot')


df = pd.read_csv('iowa_housing_train.csv', index_col='Id')                  # (1460, 81)
pd.set_option('display.max_columns', None)

"""
-----------------------------------------------------------------------------------------------------------------------
Missing values:
-----------------------------------------------------------------------------------------------------------------------
"""

""" Get list of columns with missing data """
# features_with_na = [col for col in df.columns if df[col].isnull().sum() > 0]    # 19

""" Get percentage of missing data in each feature """
# null_features_dict = {}
# for feature in features_with_na:
#     null_features_dict[feature] = round(df[feature].isnull().mean() * 100, 2)

""" Sort features with missing data by % of missing values """
# sorted_null_features_dict = {}
# sorted_keys = sorted(null_features_dict, key=null_features_dict.get, reverse=True)
# for key in sorted_keys:
#     sorted_null_features_dict[key] = null_features_dict[key]
#     print(key, 'has', null_features_dict[key], '% missing values')

# sorted_features_with_na = [
#     key + ' (' + str(value) + '% missing data)'
#     for key, value in sorted_null_features_dict.items()
# ]


"""
Identify if features with missing values have any relationship with the label (SalePrice):
    Plot bar graphs to show median house price for each feature where data is missing (1) and present (0)
    If bar heights differ, features with missing values are related to the target variable and they cannot be dropped.
"""
# plt.figure(figsize=(11, 7))
# for i, feature in enumerate(sorted_features_with_na, start=1):
#     df1 = df.copy()
#     df1.rename(columns={feature.split()[0]: feature}, inplace=True)
#     df1 = df1[[feature, 'SalePrice']]

#     # Change the feature to 1 if the value is missing, else 0
#     df1[feature] = np.where(df1[feature].isnull(), 1, 0)

#     # Calculate the mean SalePrice where the information for the feature is missing (1) or present (0):
#     plt.subplot(4, 5, i)
#     plt.tight_layout()
#     df1.groupby(feature)['SalePrice'].median().plot.bar(fontsize=8, color=['#5a7d9a', '#adad3b'])
# plt.show()


"""
-----------------------------------------------------------------------------------------------------------------------
Numerical Features:
-----------------------------------------------------------------------------------------------------------------------
"""

""" Get the list of Numerical Variables """
numeric_features = [col for col in df.columns if df[col].dtypes in ['int64', 'float64']]        # 37

""" Get the list of temporal features (features having date/year) """
year_feature = [col for col in numeric_features if 'Yr' in col or 'Year' in col]                # 4


"""
Identify if there is a relation between the temporal variables with 'SalePrice':
    Plot line graphs to show how median house price changes in time for each of the temporal variables
    We see that the median price increases as the year increases, except 'YrSold', where the median price reduces
"""
# plt.figure(figsize=(9, 5))
# for i, feature in enumerate(year_feature, start=1):
#     df1 = df.copy()
#     df1 = df1[[feature, 'SalePrice']]
#
#     plt.subplot(2, 2, i)
#     plt.tight_layout()
#     df1.groupby(feature)['SalePrice'].median().plot(fontsize=8, color='#adad3b')
# plt.show()


"""
Compare the 'difference between YrSold and year features' with SalePrice
    We see that as the difference (age) increases, house price reduces
"""
# plt.figure(figsize=(13, 4))
# for i, feature in enumerate(year_feature, start=1):
#     if feature != 'YrSold':
#         df2 = df.copy()
#         df2 = df2[[feature, 'YrSold', 'SalePrice']]
#
#         # Capture the difference between the year variable and the year the house was sold:
#         df2[feature] = df2['YrSold'] - df2[feature]
#
#         plt.subplot(1, 3, i)
#         plt.tight_layout()
#         plt.scatter(df2[feature], df2['SalePrice'], color='#adad3b')
#         plt.xlabel(feature, fontsize=8, fontweight='bold')
#         plt.ylabel('Sale Price', fontsize=8, fontweight='bold')
# plt.show()


"""
Get Cardinality (number of unique entries) of Numeric features:
"""
# print(df[numeric_features].nunique())
low_cardinality_features = [col for col in numeric_features
                            if len(df[col].unique()) < 25 and col not in year_feature]              # 17
high_cardinality_features = [col for col in numeric_features
                             if len(df[col].unique()) >= 25 and col not in year_feature]            # 16
#
# print('Low Cardinality Features: ', low_cardinality_features)
# print('Number of Low Cardinality Features: ', len(low_cardinality_features))
# print('High Cardinality Features: ', high_cardinality_features)
# print('Number of High Cardinality Features: ', len(high_cardinality_features))


"""
Plot the median sale price for numeric features with low cardinality (discrete numeric features):
"""
# plt.figure(figsize=(15, 8))
# for i, feature in enumerate(low_cardinality_features, start=1):
#     df3 = df.copy()
#     df3 = df3[[feature, 'SalePrice']]
#     plt.subplot(4, 5, i)
#     plt.tight_layout()
#     df3.groupby(feature)['SalePrice'].median().plot.bar(fontsize=8, color='#adad3b')
# plt.show()


"""
Plot histogram to identify the distribution of high cardinality features (continuous numeric features):
    We see that most of the features have a skewed distribution (and not normal distribution)
"""
# plt.figure(figsize=(15, 8))
# for i, feature in enumerate(high_cardinality_features, start=1):
#     df4 = df.copy()
#     df4 = df4[[feature]]
#     plt.subplot(4, 4, i)
#     plt.tight_layout()
#     df4[feature].hist(bins=25, rwidth=0.9, color='#adad3b')
#     plt.xlabel(feature, fontsize=8, fontweight='bold')
#     plt.ylabel('Count', fontsize=8, fontweight='bold')
# plt.show()


"""
Use Logarithmic transform to transform the numeric features with high cardinality to Gaussian distribution:
    - Plot histogram to plot the distribution after transformation
            We see that the distribution is more Gaussian than before 
    - Plot scatter plots of the transformed feature with SalePrice
            We see a positive correlation for the features with Sale Price
"""
# plt.figure(figsize=(15, 8))
# for i, feature in enumerate(high_cardinality_features, start=1):
#     df5 = df.copy()
#     df5[feature] = np.log1p(df5[feature])
#     df5['SalePrice'] = np.log1p(df5['SalePrice'])
#     plt.subplot(4, 4, i)
#     plt.tight_layout()
#
#     # Histogram:
#     df5[feature].hist(bins=25, rwidth=0.9, color='#adad3b')
#     plt.xlabel(feature, fontsize=8, fontweight='bold')
#     plt.ylabel('Count', fontsize=8, fontweight='bold')
#
#     # Scatter Plot:
#     # plt.scatter(df5[feature], df5['SalePrice'], color='#adad3b')
#     # plt.xlabel(feature, fontsize=8, fontweight='bold')
#     # plt.ylabel('Count', fontsize=8, fontweight='bold')
# plt.show()


"""
Outliers (for numeric features with high cardinality):
    Plot BoxPlot for numeric features with high cardinality to see outliers (does not work for other cases):
"""
# plt.figure(figsize=(15, 8))
# for i, feature in enumerate(high_cardinality_features, start=1):
#     df6 = df.copy()
#     df6[feature] = np.log1p(df6[feature])
#     plt.subplot(4, 4, i)
#     plt.tight_layout()
#     df6.boxplot(column=feature)
# plt.show()


"""
-----------------------------------------------------------------------------------------------------------------------
Categorical Features:
-----------------------------------------------------------------------------------------------------------------------
"""
categorical_features = [col for col in df.columns if df[col].dtypes == 'O']             # 43

""" Get cardinality of categorical features """
for feature in categorical_features:
    print(feature, df[feature].nunique())

"""
Find the relationship between categorical variables and dependent feature SalesPrice 
    Plot bar graphs to show median house price for each feature
"""
plt.figure(figsize=(15, 8))
for i, feature in enumerate(categorical_features, start=1):
    df1 = df.copy()
    df1 = df1[[feature, 'SalePrice']]
    plt.subplot(5, 9, i)
    plt.tight_layout()
    df1.groupby(feature)['SalePrice'].median().plot.bar(fontsize=6, color=['#adad3b'])
plt.show()

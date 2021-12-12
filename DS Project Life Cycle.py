"""
-----------------------------------------------------------------------------------------------------------------------
DATA SCIENCE PROJECT LIFE CYCLE:
-----------------------------------------------------------------------------------------------------------------------
-> EDA & Visualization
    -> Feature Engineering
        -> Feature Selection
            -> Model Training
                -> Hyper-Parameter Tuning
                    -> Model Deployment
                        -> Incremental Learning


-----------------------------------------------------------------------------------------------------------------------
EDA & VISUALIZATION:
-----------------------------------------------------------------------------------------------------------------------
1. Relationship between independent and dependent features
2. Missing Values
3. Numerical Features
        Cardinality of numeric features
        Distribution of continuous numeric features (numeric features having high cardinality)
4. Temporal Features (Date, Month, Year, TS)
5. Categorical Features
        Cardinality of categorical features
6. Outliers


-----------------------------------------------------------------------------------------------------------------------
FEATURE ENGINEERING:
-----------------------------------------------------------------------------------------------------------------------
0. Train-Test Split:
        from sklearn.model_selection import train_test_split

1. Mutual Information:
        from sklearn.feature_selection import mutual_info_regression, mutual_info_classif

2. Handling missing data (Imputation):
        from sklearn.impute import SimpleImputer

3. Handling categorical features:
    a. Ordinal/Label Encoding:
            Pandas factorize()
            from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
    b. One-Hot Encoding:
            Pandas get_dummies()
            from sklearn.preprocessing import OneHotEncoder
    c. Target Guided Ordinal Encoding, Mean Encoding:
            from category_encoders import MEstimateEncoder 
    d. Text Count Vectorizer:
            from sklearn.feature_extraction.text import CountVectorizer
    e. Rare Categorical Features
            Remove categorical variable labels that are present less than 1% of the observations

4. Handling Temporal features:
        Convert dtypes of date columns to 'datetime64' from 'Object'
                pd.to_datetime()
        Year feature may be converted to age etc.

5. Handling Imbalanced Dataset:
        from sklearn.model_selection import train_test_split(stratify=y)

6. Scaling and Normalization:
        a. Standardization:
                from sklearn.preprocessing import StandardScaler, RobustScaler
        b. Normalization:
                from sklearn.preprocessing import MinMaxScaler

7. Handle Different Character Encodings:
        import chardet

8. Handle Inconsistent Data Entry:
        from fuzzywuzzy import process

9. Applying Transforms:
    a. Mathematical Transforms
    b. Gaussian Transforms (Normalization)
    c. Aggregating Transforms
    d. Group Transforms
    e. String Transforms
    f. Column Transformer
            from sklearn.compose import ColumnTransformer

10. Outlier Detection and Removal
        Percentile, Std Deviation, Z-Score, IQR

11. Data Leakage:
        Target Leakage, Train-Test Contamination, Time-Series (Temporal) Data

12. Dimensionality Reduction using PCA:
        from sklearn.decomposition import PCA

13. Feature Discovery using Unsupervised Learning Algorithms:
        a. Clustering
                from sklearn.cluster import KMeans
        b. Principal Component Analysis
                from sklearn.decomposition import PCA


-----------------------------------------------------------------------------------------------------------------------
FEATURE SELECTION:   ###################
-----------------------------------------------------------------------------------------------------------------------
1. Remove features with low variance:
        from sklearn.feature_selection import VarianceThreshold

2. Uni-variate feature selection:
        from sklearn.feature_selection import SelectKBest, chi2

3. Recursive feature elimination
        from sklearn.feature_selection import RFE, RFECV

4. Select features based on importance weights:
        from sklearn.feature_selection import SelectFromModel

5. Sequential Feature Selection
        from sklearn.feature_selection import SequentialFeatureSelector


-----------------------------------------------------------------------------------------------------------------------
MODEL TRAINING:
-----------------------------------------------------------------------------------------------------------------------
1. Regression:
    a. Linear Regression:
            from sklearn.linear_model import LinearRegression
    b. Ridge Regression:
            from sklearn.linear_model import Ridge
    c. Lasso Regression:
            from sklearn.linear_model import Lasso
    d. Elastic Net Regression:
            from sklearn.linear_model import Lasso, Ridge, ElasticNet
    e. KNN Regression:
            from sklearn.neighbors import KNeighborsRegressor
    f. Decision Tree Regression:
            from sklearn.tree import DecisionTreeRegressor
    g. Random Forest Regressor:
            from sklearn.ensemble import RandomForestRegressor
    i. Support Vector Regression:
            from sklearn.svm import SVR

2. Classification:
    a. Logistic Regression:
            from sklearn.linear_model import LogisticRegression
    b. K-Nearest Neighbours:
            from sklearn.neighbors import KNeighborsClassifier
    c. Support Vector Machine:
            from sklearn.svm import SVC
    d. Decision Tree:
            from sklearn.tree import DecisionTreeClassifier
    e. Random Forest:
            from sklearn.ensemble import RandomForestClassifier
    f. Naive Bayes:
            from sklearn.naive_bayes import GaussianNB, MultinomialNB

3. Clustering:
    a. K-Means Clustering:
            from sklearn.cluster import KMeans
    b. Mean-Shift Clustering:
            from sklearn.cluster import MeanShift
    c. Density-Based Spatial Clustering of Applications with Noise (DBSCAN):
            from sklearn.cluster import DBSCAN


-----------------------------------------------------------------------------------------------------------------------
MODEL IMPROVEMENT:
-----------------------------------------------------------------------------------------------------------------------
1. Pipeline:
        from sklearn.pipeline import Pipeline

2. K-fold Cross-Validation:
        from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

3. Hyper Parameter Tuning:
        from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

4. Bagging:
        from sklearn.ensemble import BaggingClassifier, BaggingRegressor

5. Boosting:
    a. AdaBoosting
            from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
    b. Gradient Boosting
            from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
    c. XGBoost:
            from xgboost import XGBClassifier, XGBRegressor


-----------------------------------------------------------------------------------------------------------------------
MEASURING MODEL QUALITY:
-----------------------------------------------------------------------------------------------------------------------
1. Mean Absolute Error (MAE):
        from sklearn.metrics import mean_absolute_error

2. Mean Squared Error (MSE):
        from sklearn.metrics import mean_squared_error

"""

from df_reducer import minimize_by_feature_importance
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from matplotlib.pyplot import axis
from numpy import NaN, generic
import pandas as pd
import numpy as np
import os

# Reading csv file
dataFrame = pd.read_csv(
    os.getcwd() + '\\dataset\\UNSW_NB15_testing-set.csv',
    encoding='utf-8'
)

# Save dataset features
FEATURES = dataFrame.columns.values
FEATURES = np.append(FEATURES[0:2], FEATURES[5:-2])
N_FEATURES = int(len(FEATURES))

# Create matrix X
X = dataFrame.iloc[:, np.append([0, 1], np.arange(5, N_FEATURES + 3))].values

# Create vector y
y = dataFrame.iloc[:, N_FEATURES + 1].values

# Standardize value of matrix X
X = StandardScaler().fit_transform(X)

# Split X and y for testing and training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Create and fit the random forest classifier
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

# Test random forest
y_pred = clf.predict(X_test)

# Extract sub dataframe with only first ten feature
minDataFrame = minimize_by_feature_importance(clf, FEATURES,
                                              os.getcwd() + '\\dataset\\UNSW_NB15_testing-set.csv', .9)

# Building generics dataframe
generics = dataFrame[dataFrame["attack_cat"] == "Generic"]

# Deleting generics from dataframe
dataFrame = dataFrame.set_index("attack_cat")
dataFrame = dataFrame.drop("Generic", axis = 0)

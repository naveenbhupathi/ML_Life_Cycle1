import pandas as pd
import numpy as np

data = pd.read_csv("credit_card_data.csv")

original_file = data.copy()
original_file.head()

### Defaulter Distribution
original_file['Defaulter'].value_counts(normalize=True)

## Find data types
original_file.dtypes

## Find statistics of the data
original_file.describe()

###### Observed negative values in Monthly income and Total Outstanding loans
#1. Negative outstanding loans can be loan company has to pay back money to borrower
#2. Negative Monthly income means borrower doesnt have any earnings

import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x="Defaulter", data=original_file)
# Data is balanced w.r.t target attribute 'Defaulter'

## finding missing values if any
original_file.isnull().sum()
# No missing values found

## Find correlation between features
sns.heatmap(original_file.corr())
# No correlation observed between features

## Find data distribution in the features
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10,5))
sns.distplot(original_file['Monthly_Income'], ax=ax1)
sns.distplot(original_file['Total_outstanding_loans'], ax=ax2)

print(original_file['Monthly_Income'].skew())
print(original_file['Total_outstanding_loans'].skew())
# Observed left skewed for skewthly_Income and Total_outstanding_loans
#-> There are negative and zero observations in the features

sns.scatterplot(x='Monthly_Income', y='Total_outstanding_loans', data=original_file)
# We can observe outliers in the data

#pip install pyod
import pyod
from pyod.models.iforest import IForest
from collections import Counter

X = original_file.drop('Defaulter', axis=1)
y = original_file['Defaulter']
outlier_model = IForest(contamination=0.02, random_state=0)
outlier_model.fit(X)

Counter(outlier_model.labels_)
sns.scatterplot(x='Monthly_Income', y='Total_outstanding_loans', hue= outlier_model.labels_, data=X)

inliers_index = np.argwhere(outlier_model.labels_==0)
inliers_index = inliers_index.reshape(-1,)
inliers_index

print(len(inliers_index))

X_1 = X[X.index.isin(inliers_index)]
y_1 = y[y.index.isin(inliers_index)]
print(X_1.shape)
print(y_1.shape)

## Calculate skewness after removing outliers
X_1.skew()
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10,5))
sns.distplot(X_1['Monthly_Income'], ax=ax1)
sns.distplot(X_1['Total_outstanding_loans'], ax=ax2)

skewed_feats = X_1.apply(lambda x:x.dropna().skew()).sort_values()
skewed_feats

skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness

print("Pre: There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))
print("Pre", abs(skewness.Skew).mean())

from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer(method='yeo-johnson').fit(X_1)
X_2 = pd.DataFrame(pt.transform(X_1), index=X_1.index, columns=X_1.columns)
X_2

skewed_feats_transformed = X_2.apply(lambda x:x.dropna().skew()).sort_values()
skewed_feats_transformed

skewness_transformed = pd.DataFrame({'Skew' :skewed_feats_transformed})
skewness_transformed

print("Post: There are {} skewed numerical features to Box Cox transform".format(skewness_transformed.shape[0]))
print("Post", abs(skewness_transformed.Skew).mean())

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10,5))
sns.distplot(X_2['Monthly_Income'], ax=ax1)
sns.distplot(X_2['Total_outstanding_loans'], ax=ax2)

print(X_2.shape)
print(y_1.shape)

# Find uncommon items
set(X_2.index) ^ set(y_1.index)

## Model Build
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, classification_report

X_train, X_test, y_train, y_test = train_test_split(X_2, y_1, test_size=0.2, random_state=0)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

model = LogisticRegressionCV(n_jobs=-1, cv=10)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
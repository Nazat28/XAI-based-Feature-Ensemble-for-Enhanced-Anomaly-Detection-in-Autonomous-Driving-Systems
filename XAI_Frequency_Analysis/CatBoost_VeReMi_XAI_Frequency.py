#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np



df=pd.read_csv('modified_data.csv')



df.head()



df.shape




# drop the columns 'pos_x', 'pos_y', 'pos_z', 'spd_x' ,'spd_y', 'spd_z'
df = df.drop(['pos_z', 'spd_z', 'spd_x' ,'spd_y'], axis=1)
data_df=df
df.head()



from sklearn.preprocessing import StandardScaler

# define the feature columns and target variable
feature_cols = ['pos_x', 'pos_y']
X = data_df[feature_cols] # Features
y = data_df.attackerType # Target variable

from sklearn.model_selection import train_test_split


# split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# create a StandardScaler object and fit it to the training data
scaler = StandardScaler()
scaler.fit(X_train)

# transform the training and testing data using the scaler object
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)



from catboost import CatBoostClassifier 

# Create and train the CatBoostClassifier
model = CatBoostClassifier(learning_rate=0.03, depth=10)


# Fit the model on the training data
model.fit(X_train_scaled, y_train, cat_features=None, eval_set=(X_test_scaled, y_test))

# Make predictions on the testing data
predictions = model.predict(X_test_scaled)

# Calculate accuracy score

from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))


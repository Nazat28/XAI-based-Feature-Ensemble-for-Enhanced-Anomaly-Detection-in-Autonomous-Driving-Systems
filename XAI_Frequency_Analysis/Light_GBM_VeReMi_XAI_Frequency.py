#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install --upgrade numexpr bottleneck


# In[2]:


pip install --upgrade pandas


# In[3]:


pip install --upgrade dask


# In[1]:


#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_csv('modified_data.csv')

# drop the columns 'pos_x', 'pos_y', 'pos_z', 'spd_x' ,'spd_y', 'spd_z'
df = df.drop(['pos_z', 'spd_z', 'spd_x' ,'spd_y'], axis=1)
data_df=df
df.head()



from sklearn.preprocessing import StandardScaler

# define the feature columns and target variable
feature_cols = ['pos_x', 'pos_y']
X = data_df[feature_cols]
y = data_df.attackerType

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Scale the features
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert the dataset into LightGBM format
d_train = lgb.Dataset(X_train_scaled, label=y_train)

# Specify parameters
params = {
    'learning_rate': 0.5,
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',
    'max_depth': 10
}

# Train the model
clf = lgb.train(params, d_train, 500)

# Make predictions
predictions = clf.predict(X_test_scaled)

# Convert probabilities to binary predictions
binary_predictions = (predictions >= 0.5).astype(int)

# Print the classification report
print(classification_report(y_test, binary_predictions))


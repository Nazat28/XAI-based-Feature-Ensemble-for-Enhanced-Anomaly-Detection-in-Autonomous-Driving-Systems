#!/usr/bin/env python
# coding: utf-8

# In[3]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np





df=pd.read_csv('labeled_dataset_CPSS.csv')


# In[12]:


df.head()


# In[13]:


df.shape


# In[15]:


# drop the columns 'Formality', 'Location', 'Lane Alignment', 'Protocol', 'Plausibility','Frequency','Consistency', 'Speed', 'Correlation','Headway Time'
df = df.drop(['Plausibility','Frequency','Formality', 'Speed','Headway Time'], axis=1)
data_df=df
df.head()


# In[4]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

# define the feature columns and target variable
feature_cols = ['Location', 'Lane Alignment', 'Protocol','Consistency', 'Correlation']
X = df[feature_cols] # Features
y = df.Label # Target variable


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a StandardScaler object and fit it to the training data
scaler = StandardScaler()
scaler.fit(X_train)

# Transform the training and testing data using the scaler object
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# Make predictions on the testing data
predictions = model.predict(X_test_scaled)

# Calculate and print the classification report
print(classification_report(y_test, predictions))


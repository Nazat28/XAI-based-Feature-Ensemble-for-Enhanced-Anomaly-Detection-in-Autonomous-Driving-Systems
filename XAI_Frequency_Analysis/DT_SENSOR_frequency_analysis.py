#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

# In[30]:


import pandas as pd
import numpy as np


df=pd.read_csv('labeled_dataset_CPSS.csv')



df.head()


# In[31]:


# drop the columns 'Formality', 'Location', 'Lane Alignment', 'Protocol', 'Plausibility','Frequency','Consistency', 'Speed', 'Correlation','Headway Time'
data_df=df
# df = df.drop(['Consistency','Correlation', 'Location', 'Lane Alignment','Formality', 'Protocol','Frequency', 'Speed'], axis=1)
df.head()


# In[32]:


from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

# define the feature columns and target variable
feature_cols = ['Formality', 'Location', 'Lane Alignment', 'Protocol', 'Plausibility','Frequency','Consistency', 'Speed', 'Correlation','Headway Time']
X = df[feature_cols] # Features
y = df.Label # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# create a StandardScaler object and fit it to the training data
scaler = StandardScaler()
scaler.fit(X_train)

# transform the training and testing data using the scaler object
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)



from sklearn.tree import DecisionTreeClassifier
# train the decision tree classifier on the normalized training data
clf = DecisionTreeClassifier(criterion="gini", max_depth=50, min_samples_leaf=4)
clf.fit(X_train_scaled, y_train)


predictions=clf.predict(X_test_scaled)
predictions


# In[22]:


from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))


# In[3]:


pip install shap


# In[4]:


pip install pandas seaborn tensorflow numpy matplotlib


# In[3]:


# SHAP EXPLAINER
# ------------------------------>

import shap
import numpy as np
import matplotlib.pyplot as plt
# Assuming 'feature_cols' contains the names of your feature columns
feature_cols = ['Formality', 'Location', 'Lane Alignment', 'Protocol', 'Plausibility','Frequency','Consistency', 'Speed', 'Correlation','Headway Time']

# Get the class names
class_names = ['Benign', 'Anomalous']

# Create a TreeExplainer
explainer = shap.TreeExplainer(clf)

start_index = 0
end_index = 1000
shap_values = explainer.shap_values(X_test_scaled[start_index:end_index])
shap_obj = explainer(X_test_scaled[start_index:end_index])
shap.summary_plot(shap_values = shap_values,
                  features=feature_cols,
                      class_names=class_names,
                      show=False)


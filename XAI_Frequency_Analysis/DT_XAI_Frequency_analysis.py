#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np


df=pd.read_csv('modified_data.csv')


df.head()


# In[2]:



# drop the columns 'pos_x', 'pos_y', 'pos_z', 'spd_x' ,'spd_y', 'spd_z'
df = df.drop(['pos_z', 'spd_z', 'spd_y'], axis=1)
data_df=df
df.head()


# In[15]:


from sklearn.preprocessing import StandardScaler

# define the feature columns and target variable
feature_cols = ['pos_x', 'pos_y' ,'spd_x']
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


# In[5]:


# see how the scaled data looks
#print(len(X_train_scaled))
#print(len(X_test_scaled))

from sklearn.tree import DecisionTreeClassifier
# train the decision tree classifier on the normalized training data
clf = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=50, min_samples_leaf=20)
clf.fit(X_train_scaled, y_train)


# In[21]:


predictions=clf.predict(X_test_scaled)
predictions


# In[22]:


from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))




# In[2]:


# SHAP EXPLAINER
# ------------------------------>

import shap
import numpy as np
import matplotlib.pyplot as plt
# Assuming 'feature_cols' contains the names of your feature columns
feature_cols = ['pos_x', 'pos_y', 'pos_z', 'spd_x', 'spd_y', 'spd_z']

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


# In[3]:


# LIME EXPLAINER
# ------------------------------>


import lime
import lime.lime_tabular
import numpy as np
import matplotlib.pyplot as plt

# # Assuming 'feature_cols' contains the names of your feature columns
# feature_cols = ['pos_x', 'pos_y', 'pos_z', 'spd_x', 'spd_y', 'spd_z']
# # Get the class names
# class_names=['Benign','Anomalous']

# # Get the feature names
# feature_names = list(feature_cols)

# explainer = lime.lime_tabular.LimeTabularExplainer(np.array(X_train_scaled),
#                                                     feature_names=feature_names, 
#                                                     class_names=class_names,                          
#                                                     verbose=True, mode='classification')

# # Set the index of the sample you want to explain
# sample_index = 1

# # Get the explanation for the specified sample
# exp = explainer.explain_instance(np.array(X_test_scaled[sample_index]), 
#                                  clf.predict_proba, 
#                                  num_features=len(feature_cols))

# # Plot the Lime explanation
# fig = exp.as_pyplot_figure()
# plt.show()



# In[18]:


import time
print('---------------------------------------------------------------------------------')
print('Generating LIME explanation')
print('---------------------------------------------------------------------------------')
print('')



# test.pop ('Label')
print('------------------------------------------------------------------------------')

#START TIMER MODEL
start = time.time()
train =  X_train_scaled
test = X_test_scaled
test2 = test
# test = test.to_numpy()
samples = 1000

explainer = lime.lime_tabular.LimeTabularExplainer(train, feature_names= feature_cols, class_names=class_names , discretize_continuous=True)


#creating dict 
feat_list = feature_cols
# print(feat_list)

feat_dict = dict.fromkeys(feat_list, 0)
# print(feat_dict)
c = 0

num_columns = data_df.shape[1] - 1
feature_name = feature_cols
feature_name=list(feature_name)
feature_name.sort()
# print('lista',feature_name)
feature_val = []

for i in range(0,num_columns): feature_val.append(0)

for i in range(0,samples):

# i = sample
    # exp = explainer.explain_instance(test[i], rf.predict_proba)
    
    exp = explainer.explain_instance(test[i], clf.predict_proba, num_features=num_columns, top_labels=num_columns)
    # exp.show_in_notebook(show_table=True, show_all=True)
    
    #lime list to string
    lime_list = exp.as_list()
    for i in range(0,len(lime_list)):
    #---------------------------------------------------
    #fix
        my_string = lime_list[i][0]
        for index, char in enumerate(my_string):
            if char.isalpha():
                first_letter_index = index
                break  # Exit the loop when the first letter is found

        my_string = my_string[first_letter_index:]
        modified_tuple = list(lime_list[i])
        modified_tuple[0] = my_string
        lime_list[i] = tuple(modified_tuple)

    #---------------------------------------------------
    
    
    lime_list.sort()
    # print(lime_list)
    for j in range (0,num_columns): feature_val[j]+= abs(lime_list[j][1])
    # print ('debug here',lime_list[1][1])

    # lime_str = ' '.join(str(x) for x in lime_list)
    # print(lime_str)


    #lime counting features frequency 
    # for i in feat_list:
    #     if i in lime_str:
    #         #update dict
    #         feat_dict[i] = feat_dict[i] + 1
    
    c = c + 1 
    print ('progress',100*(c/samples),'%')

# Define the number you want to divide by
divider = samples

# Use a list comprehension to divide all elements by the same number
feature_val = [x / divider for x in feature_val]

# for item1, item2 in zip(feature_name, feature_val):
#     print(item1, item2)


# Use zip to combine the two lists, sort based on list1, and then unzip them
zipped_lists = list(zip(feature_name, feature_val))
zipped_lists.sort(key=lambda x: x[1],reverse=True)

# Convert the sorted result back into separate lists
sorted_list1, sorted_list2 = [list(x) for x in zip(*zipped_lists)]

# print(sorted_list1)
# print(sorted_list2)
print('----------------------------------------------------------------------------------------------------------------')

for item1, item2 in zip(sorted_list1, sorted_list2):
    print(item1, item2)

for k in sorted_list1:
#     with open(output_file_name, "a") as f:print("df.pop('",k,"')", sep='', file = f)
    print("df.pop('",k,"')", sep='')

print('---------------------------------------------------------------------------------')

# # print(feat_dict)
# # Sort values in descending order
# for k,v in sorted(feat_dict.items(), key=lambda x: x[1], reverse=True):
#   print(k,v)

# for k,v in sorted(feat_dict.items(), key=lambda x: x[1], reverse=True):
#   print("df.pop('",k,"')", sep='')

print('---------------------------------------------------------------------------------')


end = time.time()
print('ELAPSE TIME LIME GLOBAL: ',(end - start)/60, 'min')
print('---------------------------------------------------------------------------------')


# In[8]:


pip install dalex


# In[11]:


import dalex as dx

import pandas as pd
import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import xgboost as xgb

import warnings
warnings.filterwarnings('ignore')

import plotly
plotly.offline.init_notebook_mode()


# In[9]:


# Create a Dalex explainer
explainer = dx.Explainer(clf, X_test_scaled, y_test, feature_names=feature_cols)

# Calculate and plot feature importance
feature_importance = explainer.model_parts()
feature_importance.plot()


# In[12]:


# Step 1: Import necessary libraries
import dalex as dx
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import warnings

warnings.filterwarnings('ignore')

# Step 2: Load and prepare your data
df = pd.read_csv('modified_data.csv')
data_df = df

# Define the feature columns and target variable
feature_cols = ['pos_x', 'pos_y', 'pos_z', 'spd_x', 'spd_y', 'spd_z']
X = data_df[feature_cols]  # Features
y = data_df.attackerType  # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Create a StandardScaler object and fit it to the training data
scaler = StandardScaler()
scaler.fit(X_train)

# Transform the training and testing data using the scaler object
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the Decision Tree classifier on the normalized training data
clf = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=50, min_samples_leaf=20)
clf.fit(X_train_scaled, y_train)

# Make predictions
predictions = clf.predict(X_test_scaled)

# Print classification report
print(classification_report(y_test, predictions))

# Step 3: Create a Dalex explainer
explainer = dx.Explainer(clf, X_train_scaled, y_train, label="Decision Tree Classifier")

# Step 4: Calculate and plot feature importance
feature_importance = explainer.model_parts()
feature_importance.plot()


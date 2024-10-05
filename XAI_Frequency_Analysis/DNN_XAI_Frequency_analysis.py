#!/usr/bin/env python
# coding: utf-8

# In[2]:




# In[5]:


import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import shap


# In[2]:


import pandas as pd
import numpy as np


# In[3]:


df=pd.read_csv('modified_data.csv')

df.shape


# In[6]:

# drop the columns 'pos_x', 'pos_y', 'pos_z', 'spd_x' ,'spd_y', 'spd_z'
df = df.drop(['pos_z', 'spd_z', 'spd_x'], axis=1)
data_df=df
df.head()


# In[15]:


from sklearn.preprocessing import StandardScaler

# define the feature columns and target variable
feature_cols = ['pos_x', 'pos_y' ,'spd_y']
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

import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dropout

# Set random seeds for reproducibility
random_seed = 42
np.random.seed(random_seed)
random.seed(random_seed)
tf.random.set_seed(random_seed)



import time


# In[77]:


# define the keras model
model = tf.keras.Sequential()
model.add(tf.keras.Input(shape=(3,)))
model.add(Dropout(0.1))
model.add(tf.keras.layers.Dense(16,activation='relu'))
model.add(tf.keras.layers.Dense(16,activation='relu'))
model.add(tf.keras.layers.Dense(2,activation=tf.keras.activations.softmax))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()


#START TIMER MODEL
start = time.time()
model.fit(X_train_scaled, y_train, epochs=5, batch_size=100)
end = time.time()


# In[91]:


model.summary()


# In[78]:


predictions =model.predict(X_test_scaled)
predictions


# In[79]:


predictions = predictions.argmax(axis=1)


# In[80]:

from sklearn.metrics import precision_score, f1_score, recall_score

precision = precision_score(y_test, predictions)
f1 = f1_score(y_test, predictions)
recall= recall_score(y_test, predictions)


# In[81]:


from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))


# In[4]:


import shap
# Fits the explainer
explainer = shap.Explainer(model.predict, X_test_scaled)
start_index = 0
end_index = 1000
# Calculates the SHAP values - It takes some time
shap_values = explainer(X_test_scaled[start_index:end_index])


# In[8]:


import time
import shap

# start time
start=time.time()

start_index =0
end_index = 100
# Assuming 'feature_cols' contains the names of your feature columns
feature_cols = ['pos_x', 'pos_y', 'pos_z', 'spd_x', 'spd_y', 'spd_z']

explainer = shap.DeepExplainer(model, X_test_scaled[start_index:end_index].astype('float'))
shap_values = explainer.shap_values(X_test_scaled[start_index:end_index].astype('float'))


# In[10]:


import shap
import time
import matplotlib.pyplot as plt

# Assuming shap_values and other variables are defined before this point

# Start time
start = time.time()

# Generate a summary plot
shap.summary_plot(shap_values=shap_values,
                  features=X_test_scaled[0:1000],
                  class_names=['Benign', 'Anomalous'],
                  feature_names=feature_cols,  # pass feature names as a list
                  show=False)

# Waterfall plot
shap.waterfall_plot(shap.Explanation(values=shap_values[0], base_values=explainer.expected_value[1], data=X_test_scaled[0], feature_names=feature_cols), max_display=6)

# End time
end = time.time()
print('SHAP time for DNN:', (end - start), 'sec')

# Show the plots
plt.show()


# In[11]:


import lime
import lime.lime_tabular
import numpy as np
import matplotlib.pyplot as plt

# Assuming 'feature_cols' contains the names of your feature columns
feature_cols = ['pos_x', 'pos_y', 'pos_z', 'spd_x', 'spd_y', 'spd_z']
# Get the class names
class_names=['Benign','Anomalous']

# Get the feature names
feature_names = list(feature_cols)

explainer = lime.lime_tabular.LimeTabularExplainer(np.array(X_train_scaled),
                    feature_names=feature_names, 
                    class_names=class_names,                          
                    verbose=True, mode='classification')


# In[12]:


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
feature_name.sort()
# print('lista',feature_name)
feature_val = []

for i in range(0,num_columns): feature_val.append(0)

for i in range(0,samples):

# i = sample
    # exp = explainer.explain_instance(test[i], rf.predict_proba)
    
    exp = explainer.explain_instance(test[i], model.predict_proba, num_features=num_columns, top_labels=num_columns)
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


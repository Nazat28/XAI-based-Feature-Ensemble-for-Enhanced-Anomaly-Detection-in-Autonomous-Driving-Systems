#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install dalex


# In[2]:


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


# In[4]:


# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names


# In[ ]:


X


# In[ ]:


y


# In[ ]:


feature_names


# In[11]:


import dalex as dx

import pandas as pd
import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

import warnings
warnings.filterwarnings('ignore')

import plotly
plotly.offline.init_notebook_mode()


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Neural Network model
model = MLPClassifier(random_state=42, max_iter=300)
model.fit(X_train, y_train)

# Create a Dalex explainer
explainer = dx.Explainer(model, X_train, y_train, feature_names=feature_names)

# Calculate and plot feature importance
feature_importance = explainer.model_parts()
feature_importance.plot()


# In[12]:


import dalex as dx

import pandas as pd
import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

import warnings
warnings.filterwarnings('ignore')

import plotly
plotly.offline.init_notebook_mode()

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Neural Network model
model = MLPClassifier(random_state=42, max_iter=300)
model.fit(X_train, y_train)

# Create a Dalex explainer
explainer = dx.Explainer(model, X_train, y_train)

# Calculate and plot feature importance
feature_importance = explainer.model_parts()
feature_importance.plot()


# In[1]:


# -------------
# DT CLASSIFIER
# -------------


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

df=pd.read_csv('merged_modified.csv')


df.head()


# In[33]:


# drop the columns 'Formality', 'Location', 'Lane Alignment', 'Protocol', 'Plausibility','Frequency','Consistency', 'Speed', 'Correlation','Headway Time'
data_df=df
# df = df.drop(['Location','Consistency', 'Correlation','Lane Alignment','Formality', 'Protocol', 'Speed','Headway Time'], axis=1)
df.head()

# In[34]:


from sklearn.preprocessing import StandardScaler


# In[20]:


from sklearn.model_selection import train_test_split

# define the feature columns and target variable
feature_cols = ['pos_x', 'pos_y', 'pos_z', 'spd_x' ,'spd_y', 'spd_z']
X = data_df[feature_cols] # Features
y = data_df.attackerType # Target variable



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

# Create a Dalex explainer
explainer = dx.Explainer(clf, X_train_scaled, y_train, label="DT Classifier")

# Update the explainer's data attribute to include feature names
explainer.data = pd.DataFrame(X_train_scaled, columns=feature_cols)

# Calculate and plot feature importance
feature_importance = explainer.model_parts()
feature_importance.plot()


# In[2]:


import pandas as pd
import numpy as np


df=pd.read_csv('merged_modified.csv')


df.head()


# In[33]:


# drop the columns 'Formality', 'Location', 'Lane Alignment', 'Protocol', 'Plausibility','Frequency','Consistency', 'Speed', 'Correlation','Headway Time'
data_df=df
# df = df.drop(['Location','Consistency', 'Correlation','Lane Alignment','Formality', 'Protocol', 'Speed','Headway Time'], axis=1)
df.head()

# In[34]:


from sklearn.preprocessing import StandardScaler


# In[20]:


from sklearn.model_selection import train_test_split

# define the feature columns and target variable
feature_cols = ['pos_x', 'pos_y', 'pos_z', 'spd_x' ,'spd_y', 'spd_z']
X = data_df[feature_cols] # Features
y = data_df.attackerType # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# create a StandardScaler object and fit it to the training data
scaler = StandardScaler()
scaler.fit(X_train)

# transform the training and testing data using the scaler object
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[13]:


#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

clf=RandomForestClassifier(criterion="gini",max_depth=50,n_estimators=100)
clf.fit(X_train_scaled,y_train)


predictions=clf.predict(X_test_scaled)


from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))

# Create a Dalex explainer
explainer = dx.Explainer(clf, X_train_scaled, y_train, label="RF Classifier")

# Update the explainer's data attribute to include feature names
explainer.data = pd.DataFrame(X_train_scaled, columns=feature_cols)

# Calculate and plot feature importance
feature_importance = explainer.model_parts()
feature_importance.plot()


# In[16]:


# Create a Dalex explainer without the features argument
explainer = dx.Explainer(clf, X_train_scaled, y_train, label="Decision Tree Classifier")

# Update the explainer's data attribute to include feature names
explainer.data = pd.DataFrame(X_train_scaled, columns=feature_cols)

# Calculate and plot feature importance
feature_importance = explainer.model_parts()
feature_importance.plot()


# In[17]:


# -------------
# RF CLASSIFIER
# -------------


#!/usr/bin/env python
# coding: utf-8

# Import necessary libraries
import pandas as pd
import numpy as np
import dalex as dx
import warnings
warnings.filterwarnings('ignore')

# Load and prepare your data
df = pd.read_csv('modified_data.csv')
df.head()
print(df.shape)

# Drop unnecessary columns if needed (commented out in this case)
# df = df.drop(['pos_z','pos_x', 'spd_x','pos_y'], axis=1)
data_df = df
df.head()

# Define the feature columns and target variable
feature_cols = ['pos_x', 'pos_y', 'pos_z', 'spd_x', 'spd_y', 'spd_z']
X = data_df[feature_cols]  # Features
y = data_df.attackerType  # Target variable

from sklearn.model_selection import train_test_split

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

# Create a StandardScaler object and fit it to the training data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

# Transform the training and testing data using the scaler object
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(criterion="gini", max_depth=50, n_estimators=100, min_samples_leaf=4)
clf.fit(X_train_scaled, y_train)

# Make predictions
predictions = clf.predict(X_test_scaled)

# Print classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))

# Create a Dalex explainer
explainer = dx.Explainer(clf, X_train_scaled, y_train, label="Random Forest Classifier")

# Update the explainer's data attribute to include feature names
explainer.data = pd.DataFrame(X_train_scaled, columns=feature_cols)

# Calculate and plot feature importance
feature_importance = explainer.model_parts()
feature_importance.plot()


# In[3]:


# -------------
# KNN CLASSIFIER
# -------------

#!/usr/bin/env python
# coding: utf-8

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import dalex as dx
import warnings
warnings.filterwarnings('ignore')


df=pd.read_csv('merged_modified.csv')


df.head()


# In[33]:


# drop the columns 'Formality', 'Location', 'Lane Alignment', 'Protocol', 'Plausibility','Frequency','Consistency', 'Speed', 'Correlation','Headway Time'
data_df=df
# df = df.drop(['Location','Consistency', 'Correlation','Lane Alignment','Formality', 'Protocol', 'Speed','Headway Time'], axis=1)
df.head()

# In[34]:


from sklearn.preprocessing import StandardScaler


# In[20]:


from sklearn.model_selection import train_test_split

# define the feature columns and target variable
feature_cols = ['pos_x', 'pos_y', 'pos_z', 'spd_x' ,'spd_y', 'spd_z']
X = data_df[feature_cols] # Features
y = data_df.attackerType # Target variable

from sklearn.model_selection import train_test_split

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

# Create a StandardScaler object and fit it to the training data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

# Transform the training and testing data using the scaler object
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the K-Nearest Neighbors Classifier
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier()
clf.fit(X_train_scaled, y_train)

# Make predictions
predictions = clf.predict(X_test_scaled)

# Print classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))

# Create a Dalex explainer
explainer = dx.Explainer(clf, X_train_scaled, y_train, label="KNN Classifier")

# Update the explainer's data attribute to include feature names
explainer.data = pd.DataFrame(X_train_scaled, columns=feature_cols)

# Calculate and plot feature importance
feature_importance = explainer.model_parts()
feature_importance.plot()


# In[4]:


# -------------
# AdaBoost CLASSIFIER
# -------------


#!/usr/bin/env python
# coding: utf-8

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dalex as dx
import warnings
warnings.filterwarnings('ignore')

df=pd.read_csv('merged_modified.csv')


df.head()


# In[33]:


# drop the columns 'Formality', 'Location', 'Lane Alignment', 'Protocol', 'Plausibility','Frequency','Consistency', 'Speed', 'Correlation','Headway Time'
data_df=df
# df = df.drop(['Location','Consistency', 'Correlation','Lane Alignment','Formality', 'Protocol', 'Speed','Headway Time'], axis=1)
df.head()

# In[34]:


from sklearn.preprocessing import StandardScaler


# In[20]:


from sklearn.model_selection import train_test_split

# define the feature columns and target variable
feature_cols = ['pos_x', 'pos_y', 'pos_z', 'spd_x' ,'spd_y', 'spd_z']
X = data_df[feature_cols] # Features
y = data_df.attackerType # Target variable

from sklearn.model_selection import train_test_split

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

# Create a StandardScaler object and fit it to the training data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

# Transform the training and testing data using the scaler object
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the AdaBoost Classifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
base_estimator = DecisionTreeClassifier(max_depth=4)
clf = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=50)
clf.fit(X_train_scaled, y_train)

# Make predictions
predictions = clf.predict(X_test_scaled)

# Print classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))

# Create a Dalex explainer
explainer = dx.Explainer(clf, X_train_scaled, y_train, label="AdaBoost Classifier")

# Update the explainer's data attribute to include feature names
explainer.data = pd.DataFrame(X_train_scaled, columns=feature_cols)

# Calculate and plot feature importance
feature_importance = explainer.model_parts()
feature_importance.plot()


# In[15]:


# -------------
# AdaBoost CLASSIFIER
# -------------

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dalex as dx
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv('labeled_dataset_CPSS.csv')
df.head()

# Prepare the data
data_df = df

# Define the feature columns and target variable
feature_cols = ['Formality', 'Location', 'Lane Alignment', 'Protocol', 'Plausibility', 'Frequency', 'Consistency', 'Speed', 'Correlation', 'Headway Time']
X = df[feature_cols]  # Features
y = df['Label']  # Target variable

# Split the dataset into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

# Create a StandardScaler object and fit it to the training data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

# Transform the training and testing data using the scaler object
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the AdaBoost Classifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
base_estimator = DecisionTreeClassifier(max_depth=4)
clf = AdaBoostClassifier(estimator=base_estimator, n_estimators=50)
clf.fit(X_train_scaled, y_train)

# Make predictions
predictions = clf.predict(X_test_scaled)

# Print classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))

# Create a Dalex explainer
explainer = dx.Explainer(clf, X_train_scaled, y_train, label="AdaBoost Classifier")

# Update the explainer's data attribute to include feature names
explainer.data = pd.DataFrame(X_train_scaled, columns=feature_cols)

# Calculate and plot feature importance
feature_importance = explainer.model_parts()
feature_importance.plot()


# In[5]:


# -------------
# SVM CLASSIFIER
# -------------


#!/usr/bin/env python
# coding: utf-8

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dalex as dx
import warnings
warnings.filterwarnings('ignore')

df=pd.read_csv('merged_modified.csv')


df.head()


# In[33]:


# drop the columns 'Formality', 'Location', 'Lane Alignment', 'Protocol', 'Plausibility','Frequency','Consistency', 'Speed', 'Correlation','Headway Time'
data_df=df
# df = df.drop(['Location','Consistency', 'Correlation','Lane Alignment','Formality', 'Protocol', 'Speed','Headway Time'], axis=1)
df.head()

# In[34]:


from sklearn.preprocessing import StandardScaler


# In[20]:


from sklearn.model_selection import train_test_split

# define the feature columns and target variable
feature_cols = ['pos_x', 'pos_y', 'pos_z', 'spd_x' ,'spd_y', 'spd_z']
X = data_df[feature_cols] # Features
y = data_df.attackerType # Target variable

# Split the dataset into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

# Create a StandardScaler object and fit it to the training data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

# Transform the training and testing data using the scaler object
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train an SVM model using a smaller subset of the data
from sklearn.svm import SVC
subset_size = 1000
X_train_subset = X_train_scaled[:subset_size]
y_train_subset = y_train[:subset_size]
clf = SVC(kernel='rbf', C=1, gamma='auto', probability=True, random_state=100)
clf.fit(X_train_subset, y_train_subset)

# Make predictions
predictions = clf.predict(X_test_scaled)

# Print classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))

# Create a Dalex explainer
explainer = dx.Explainer(clf, X_train_scaled, y_train, label="SVM Classifier")

# Update the explainer's data attribute to include feature names
explainer.data = pd.DataFrame(X_train_scaled, columns=feature_cols)

# Calculate and plot feature importance
feature_importance = explainer.model_parts()
feature_importance.plot()


# In[6]:


#!/usr/bin/env python
# coding: utf-8

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, f1_score, recall_score, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
import dalex as dx
import random
import time

# Load and prepare your data
# Load the dataset
df=pd.read_csv('merged_modified.csv')


df.head()


# In[33]:


# drop the columns 'Formality', 'Location', 'Lane Alignment', 'Protocol', 'Plausibility','Frequency','Consistency', 'Speed', 'Correlation','Headway Time'
data_df=df
# df = df.drop(['Location','Consistency', 'Correlation','Lane Alignment','Formality', 'Protocol', 'Speed','Headway Time'], axis=1)
df.head()

# In[34]:


from sklearn.preprocessing import StandardScaler


# In[20]:


from sklearn.model_selection import train_test_split

# define the feature columns and target variable
feature_cols = ['pos_x', 'pos_y', 'pos_z', 'spd_x' ,'spd_y', 'spd_z']
X = data_df[feature_cols] # Features
y = data_df.attackerType # Target variable

# Convert target variable to one-hot encoded format
y_one_hot = to_categorical(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.3, random_state=100)

# Create a StandardScaler object and fit it to the training data
scaler = StandardScaler()
scaler.fit(X_train)

# Transform the training and testing data using the scaler object
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Set random seeds for reproducibility
random_seed = 42
np.random.seed(random_seed)
random.seed(random_seed)
tf.random.set_seed(random_seed)

# Define the Keras model
model = tf.keras.Sequential([
    tf.keras.Input(shape=(6,)),
    Dropout(0.1),
    Dense(16, activation='relu'),
    Dense(16, activation='relu'),
    Dense(2, activation=tf.keras.activations.softmax)
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model
start = time.time()
model.fit(X_train_scaled, y_train, epochs=5, batch_size=100)
end = time.time()

# Make predictions
predictions = model.predict(X_test_scaled)
predictions = predictions.argmax(axis=1)
y_test_labels = y_test.argmax(axis=1)  # Convert one-hot encoded y_test to labels

# Evaluate the model
precision = precision_score(y_test_labels, predictions)
f1 = f1_score(y_test_labels, predictions)
recall = recall_score(y_test_labels, predictions)

print(classification_report(y_test_labels, predictions))

# Convert y_train back to one-dimensional array for Dalex
y_train_labels = y_train.argmax(axis=1)

# Create a Dalex explainer
explainer = dx.Explainer(model, X_train_scaled, y_train_labels, label="Neural Network Classifier")

# Update the explainer's data attribute to include feature names
explainer.data = pd.DataFrame(X_train_scaled, columns=feature_cols)

# Calculate and plot feature importance
feature_importance = explainer.model_parts()
feature_importance.plot()


# In[7]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import dalex as dx

# Load the data
df = pd.read_csv('merged_modified.csv')
data_df = df

# Define the feature columns and target variable
feature_cols = ['pos_y', 'pos_x', 'spd_x', 'pos_z', 'spd_z', 'spd_y']
X = data_df[feature_cols]  # Features
y = data_df.attackerType   # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

# Create a StandardScaler object and fit it to the training data
scaler = StandardScaler()
scaler.fit(X_train)

# Transform the training and testing data using the scaler object
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("1. Done!")


# Define and train the MLP classifier
clf_mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, activation='relu', solver='adam', random_state=1)

print("2. Done!")

clf_mlp.fit(X_train_scaled, y_train)

predictions_mlp = clf_mlp.predict(X_test_scaled)

print("MLP Classifier Report:")
print(classification_report(y_test, predictions_mlp))

# Create a Dalex explainer for MLP
explainer_mlp = dx.Explainer(clf_mlp, X_train_scaled, y_train, label="DNN Classifier")

# Update the explainer's data attribute to include feature names
explainer_mlp.data = pd.DataFrame(X_train_scaled, columns=feature_cols)

# Calculate and plot feature importance for MLP
feature_importance_mlp = explainer_mlp.model_parts()
feature_importance_mlp.plot()


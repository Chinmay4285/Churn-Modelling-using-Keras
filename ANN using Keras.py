
# coding: utf-8

# ### Business Problem
# Our basic aim is to predict customer churn for a certain bank i.e. which customer is going to leave this bank service. Dataset is small(for learning purpose) and contains 10000 rows with 14 columns. 

# In[7]:


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[8]:


import keras


# In[ ]:





# In[9]:


# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')


# In[10]:


dataset.columns


# In[11]:


dataset.head()


# In[12]:


dataset.info()


# In[13]:


X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values


# In[14]:


dataset.iloc[:, 3:13].head()


# In[15]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])


# In[16]:


X


# In[17]:


onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]


# In[18]:


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


# In[19]:


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[20]:


# Importing the Keras libraries and packages
import keras as kr


# In[21]:


dir(kr)


# In[22]:


import keras.models as km


# In[23]:


from keras.models import Sequential
from keras.layers import Dense


# In[24]:


#Initializing Neural Network
classifier = Sequential()


# In[25]:


# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))
# Adding the second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))


# In[26]:


# Compiling Neural Network
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# In[27]:


# Fitting our model 
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 10)


# In[29]:


# Predicting the Test set results
y_pred = classifier.predict(X_test)


# In[30]:


y_pred


# In[31]:


y_pred = (y_pred > 0.2)


# In[32]:


y_pred


# In[33]:


# Creating the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# In[34]:


import sklearn.metrics as metrics


# In[35]:


print(metrics.classification_report(y_test, y_pred))


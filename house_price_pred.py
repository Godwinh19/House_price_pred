#!/usr/bin/env python
# coding: utf-8

# In[1]:


#this is Simple Linear Regression model that predict the price of a land knowing the surface


# In[2]:


# Importing the libraries
#import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
#get_ipython().magic('pylab inline')


# In[3]:


# Importing the dataset
dataset = pd.read_csv('house.csv')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:, 0].values


# In[4]:


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 40)


# In[5]:


#Fitting simple linear regression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# In[6]:


#Predicting the test set results
y_pred = regressor.predict(X_test)


# In[7]:


# Visualize the training set results
#pylab.rcParams['figure.figsize'] = (15, 6)
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Price vs Surface (Trainig set)')
plt.xlabel('Surface')
plt.ylabel('Price $')
plt.show()


# In[8]:


#Visualize the test set results
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Price vs Surface (Test set)')
plt.xlabel('Surface')
plt.ylabel('Price $')
plt.show()


# In[10]:


print('Accuracy of our model on the training data is {:.2f} out of 1'.format(regressor.score(X_train,y_train)))
print('Accuracy of our model on the test data is {:.2f} out of 1'.format(regressor.score(X_test,y_test)))


# In[ ]:





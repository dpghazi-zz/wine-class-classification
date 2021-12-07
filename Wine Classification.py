#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')
from sklearn.datasets import make_blobs
from sklearn import linear_model, datasets, metrics
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv("data.csv")
df = df.ix[:,1:14]
df


# In[3]:


plt.scatter(df['12'],df['0'])
plt.show()


# In[4]:


df.groupby('12')['0'].plot(kind='density')
plt.legend(labels=['Red','White'])
plt.show()


# In[5]:


df.groupby('12')['1'].plot(kind='density')
plt.legend(labels=['Red','White'])
plt.show()


# In[6]:


df.groupby('12')['2'].plot(kind='density')
plt.legend(labels=['Red','White'])
plt.show()


# In[7]:


df.groupby('12')['4'].plot(kind='density')
plt.legend(labels=['Red','White'])
plt.show()


# In[8]:


df.groupby('12')['5'].plot(kind='density')
plt.legend(labels=['Red','White'])
plt.show()


# In[9]:


labels = df.ix[:,12:13]


# In[10]:


features = df.ix[:, 0:12]

training_proportion = (6000 / 100) * 80
training_features = features.ix[:training_proportion]
validation_features = features.ix[training_proportion:]
training_labels = labels.ix[:training_proportion]
validation_labels = labels.ix[training_proportion:]


# In[11]:


def modelTraining(training_features, training_labels, validation_features, validation_labels, n, regression_type):
    x = training_features[:n]
    y = training_labels[:n]
    lrm = linear_model.LogisticRegression(penalty=regression_type)
    lrm.fit(x, y)
    
    pred = lrm.predict(validation_features)
    score = metrics.accuracy_score(validation_labels, pred)
    
    return score


# In[12]:


sample_points = [100, 200, 500, 1000, 2000, 4800]

L1Scores = []

for x in sample_points:
    L1Scores.append(modelTraining(training_features, 
                                  training_labels, 
                                  validation_features, 
                                  validation_labels,
                                  x,
                                  "l1"))
    
L1Scores


# In[13]:


plt.plot(sample_points, L1Scores, 'ro')
plt.xlabel('n training samples')
plt.ylabel('validation accuracy')
plt.show()


# In[14]:


L2Scores = []

for x in sample_points:
    L2Scores.append(modelTraining(training_features, 
                                  training_labels, 
                                  validation_features, 
                                  validation_labels,
                                  x,
                                  "l2"))
    
L2Scores


# In[15]:


plt.plot(sample_points, L2Scores, 'ro')
plt.xlabel('n training samples')
plt.ylabel('validation accuracy')
plt.show()


# L1 performed better than L2, as can be seen by the higher validation accuracy in the first graph for all data points

# In[16]:


test = pd.read_csv("./test.csv")
test = test.ix[:,1:14]
test_features = test.ix[:, 0:12]

lrm2 = linear_model.LogisticRegression(penalty='l1')
lrm2.fit(features, labels)
results = lrm2.predict(test)

np.savetxt("./submission.csv", results, delimiter=',')


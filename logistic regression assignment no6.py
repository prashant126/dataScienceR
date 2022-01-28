#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sb
from sklearn.linear_model import LogisticRegression


# In[3]:


app_log = pd.read_csv("C:/Users/Prashant/Downloads/bank-full (1).csv",sep=';')


# In[4]:


app_log.tail()


# In[5]:


app_log.tail()


# In[6]:


# select columns
columns = ['age', 'balance', 'duration', 'campaign', 'y']
app_log_sel = app_log[columns]
app_log_sel.info()


# In[7]:


pd.crosstab(app_log_sel.age,app_log_sel.y).plot(kind="line")


# In[8]:


sb.boxplot(data =app_log_sel,orient = "v")


# In[9]:


app_log_sel['outcome'] = app_log_sel.y.map({'no':0, 'yes':1})
app_log_sel.tail(10)


# In[10]:


app_log_sel.boxplot(column='age', by='outcome')


# In[11]:


#probably not a great feature since lot of outliers


# In[12]:


feature_col=['age','balance','duration','campaign']
output_target=['outcome']
X = app_log_sel[feature_col]
Y = app_log_sel[output_target]


# In[13]:


classifier = LogisticRegression()


# In[14]:


classifier.fit(X,Y)


# In[15]:


classifier.coef_ # coefficients of features 


# In[16]:


classifier.predict_proba (X) # Probability values 


# In[17]:


y_pred = classifier.predict(X)


# In[18]:


y_pred


# In[19]:


from sklearn.metrics import confusion_matrix


# In[20]:


confusion_matrix = confusion_matrix(Y,y_pred)


# In[21]:


print (confusion_matrix)


# In[22]:


import matplotlib.pyplot as plt


# In[23]:


sb.heatmap(confusion_matrix, annot=True)
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')


# In[ ]:





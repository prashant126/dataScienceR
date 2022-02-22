#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


df = pd.read_csv("C:/Users/Prashant/Downloads/my_movies.csv")
df


# In[5]:


#Replacing NA Values with Blank Spaces


# In[6]:


df= df.replace(np.nan,'',regex=True)
df


# In[7]:


df=df.iloc[:,5:]
df.head()


# In[8]:


frequent_itemset = apriori(df,min_support=0.2,use_colnames=True)
frequent_itemset


# In[9]:


rules = association_rules(frequent_itemset,metric="confidence",min_threshold=0.6)
rules


# In[10]:


rules.sort_values('lift',ascending=False)


# In[11]:


rules[ (rules['lift'] >=1) &
       (rules['confidence'] >= 0.5) ]


# In[12]:


plt.scatter(rules['support'], rules['confidence'])
plt.xlabel('support')
plt.ylabel('confidence')
plt.title('Support vs Confidence')
plt.show()


# In[13]:


plt.scatter(rules['support'], rules['lift'])
plt.xlabel('support')
plt.ylabel('lift')
plt.title('Support vs Lift')
plt.show()


# In[14]:


fit = np.polyfit(rules['lift'], rules['confidence'], 1)
fit_fn = np.poly1d(fit)
plt.plot(rules['lift'], rules['confidence'], 'yo', rules['lift'],
fit_fn(rules['lift']))
plt.xlabel('Lift Ratio')
plt.ylabel('Condifence')
plt.title('Confidence vs Lift')


# In[ ]:





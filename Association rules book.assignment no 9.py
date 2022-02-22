#!/usr/bin/env python
# coding: utf-8

# In[19]:


import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import matplotlib.pyplot as plt
import seaborn as sns


# In[20]:


df = pd.read_csv(r"C:\Users\Prashant\Downloads\book.csv")
df


# In[21]:


df.isnull().sum()


# In[22]:


a=df['ChildBks'].value_counts()
b=df['YouthBks'].value_counts()
c=df['CookBks'].value_counts()
d=df['DoItYBks'].value_counts()
e=df['RefBks'].value_counts()
f=df['ArtBks'].value_counts()
g=df['GeogBks'].value_counts()
h=df['ItalCook'].value_counts()
i=df['ItalAtlas'].value_counts()
j=df['ItalArt'].value_counts()
k=df['Florence'].value_counts()


# In[23]:


list1=[a[1]/2000,b[1]/2000,c[1]/2000,d[1]/2000,e[1]/2000,f[1]/2000,g[1]/2000,h[1]/2000,i[1]/2000,j[1]/2000,k[1]/2000]


# In[24]:


print(list1)


# In[25]:


#Minimum Support for this Problem should be around 0.2


# In[26]:


frequent_itemset = apriori(df,min_support=0.1,use_colnames=True)
frequent_itemset


# In[27]:


rules = association_rules(frequent_itemset,metric="confidence",min_threshold=0.7)
rules


# In[28]:


rules.sort_values('lift',ascending=False)


# In[29]:


rules[ (rules['lift'] >=1) &
       (rules['confidence'] >= 0.7) ]


# In[30]:


plt.scatter(rules['support'], rules['confidence'])
plt.xlabel('support')
plt.ylabel('confidence')
plt.title('Support vs Confidence')
plt.show()


# In[31]:


plt.scatter(rules['support'], rules['lift'])
plt.xlabel('support')
plt.ylabel('lift')
plt.title('Support vs Lift')
plt.show()


# In[32]:


fit = np.polyfit(rules['lift'], rules['confidence'], 1)
fit_fn = np.poly1d(fit)
plt.plot(rules['lift'], rules['confidence'], 'yo', rules['lift'],
fit_fn(rules['lift']))
plt.xlabel('Lift Ratio')
plt.ylabel('Condifence')
plt.title('Confidence vs Lift')


# In[ ]:





# In[ ]:





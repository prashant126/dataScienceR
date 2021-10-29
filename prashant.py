
# coding: utf-8

# In[22]:


get_ipython().system('pip')


# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv("https://raw.githubusercontent.com/Apress/data-analysis-and-visualization-using-python/master/Ch07/Salaries.csv")


# In[3]:


df.head()


# In[5]:


df.columns


# In[6]:


df.dtypes


# In[14]:


df.describe()


# In[15]:


df.min()


# In[16]:


df.max()


# In[30]:


df.size


# In[25]:


print (df.dtypes)


# In[31]:


df.mean()


# In[32]:


df.discipline.value_counts()


# In[33]:


df.std()


# In[36]:


df.dropna()


# In[38]:


df.phd.describe()


# In[39]:


df.phd.count()


# In[40]:


df.phd.mean()


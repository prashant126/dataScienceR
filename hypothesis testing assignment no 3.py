#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import norm


# In[5]:


# Load the dataset
data=pd.read_csv('C:/Users/Prashant/Downloads/Cutlets.csv')
data.head()


# In[6]:


unitA=pd.Series(data.iloc[:,0])
unitA


# In[7]:


unitB=pd.Series(data.iloc[:,1])
unitB


# In[8]:


# 2-sample 2-tail ttest:   stats.ttest_ind(array1,array2)     # ind -> independent samples
p_value=stats.ttest_ind(unitA,unitB)
p_value


# In[9]:


p_value[1]     # 2-tail probability 


# In[10]:


#A F&B manager wants to determine whether there is any significant difference in the diameter of the cutlet between two units. A randomly selected sample of cutlets was collected from both units and measured? Analyze the data and draw inferences at 5% significance level. Please state the assumptions and tests that you carried out to check validity of the assumptions. Cutlets.csv

#Assume Null hyposthesis as Ho: μ1 = μ2 (There is no difference in diameters of cutlets between two units).

#thus Alternate hypothesis as Ha: μ1 ≠ μ2 (There is significant difference in diameters of cutlets between two units) 2 Sample 2 Tail test applicable


# In[ ]:





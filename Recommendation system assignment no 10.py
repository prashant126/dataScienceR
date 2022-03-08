#!/usr/bin/env python
# coding: utf-8

# In[1]:


##Recommendation System
##Problem statement

##Build a recommender system by using cosine simillarties score.

##Importing Required Libraries


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


# In[6]:


book = pd.read_csv("C:/Users/Prashant/Downloads/book (2).csv")


# In[7]:


book.head()


# In[8]:


book.tail()


# In[9]:


book.shape


# In[10]:


book.info()


# In[11]:


book.isnull().sum()


# In[12]:


book.drop(book.columns[[0]],axis=1,inplace =True)
book


# In[13]:


book.nunique()


# In[14]:


#Renaming the colums
book.columns = ["UserID","BookTitle","BookRating"]


# In[15]:


book


# In[16]:


book =book.sort_values(by=['UserID'])


# In[17]:


#number of unique users in the dataset
len(book.UserID.unique())


# In[18]:


#Unique movies
len(book.BookTitle.unique())


# In[19]:


book.loc[book["BookRating"] == 'small', 'BookRating'] = 0
book.loc[book["BookRating"] == 'large', 'BookRating'] = 1


# In[20]:


book.BookRating.value_counts()


# In[21]:


plt.figure(figsize=(20,6))
sns.distplot(book.BookRating)


# In[22]:


book_df = book.pivot_table(index='UserID',
                   columns='BookTitle',
                   values='BookRating').reset_index(drop=True)


# In[23]:


book_df.fillna(0,inplace=True)


# In[24]:


book_df


# In[25]:


##AVERAGE RATING OF BOOKS


# In[26]:


AVG = book['BookRating'].mean()
print(AVG)


# In[27]:


AVG = book['BookRating'].mean()
print(AVG)


# In[28]:


# Calculate the minimum number of votes required to be in the chart, 
minimum = book['BookRating'].quantile(0.90)
print(minimum)


# In[29]:


# Filter out all qualified Books into a new DataFrame
q_Books = book.copy().loc[book['BookRating'] >= minimum]
q_Books.shape


# In[30]:


##Calculating Cosine Similarity between Users


# In[31]:


from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine,correlation


# In[32]:


user_sim=1-pairwise_distances(book_df.values,metric='cosine')


# In[33]:


user_sim


# In[34]:


user_sim_df=pd.DataFrame(user_sim)


# In[35]:


user_sim_df


# In[36]:


#Set the index and column names to user ids 
user_sim_df.index = book.UserID.unique()
user_sim_df.columns = book.UserID.unique()


# In[37]:


user_sim_df


# In[38]:


np.fill_diagonal(user_sim,0)
user_sim_df


# In[39]:


#Most Similar Users
print(user_sim_df.idxmax(axis=1)[1348])
print(user_sim_df.max(axis=1).sort_values(ascending=False).head(50))


# In[40]:


##1 represents that the two user ID have read the same books
##1348 has highest correlation with 2576 UserID


# In[41]:


reader = book[(book['UserID']==1348) | (book['UserID']==2576)]
reader


# In[42]:


reader1=book[(book['UserID']==1348)] 
reader1


# In[43]:


reader2=book[(book['UserID']==2576)] 
reader2


# In[44]:


##Result : BookTitle Stardust whose UserID 2576 has BookRating 10


# In[ ]:





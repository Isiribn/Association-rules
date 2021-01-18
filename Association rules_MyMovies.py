#!/usr/bin/env python
# coding: utf-8

# In[20]:


import pandas as pd
data=pd.read_csv("my_movies.csv")
data.head()


# In[21]:


data.shape


# In[22]:


data.tail()


# In[25]:


data=data.drop(["V1","V2","V3","V4","V5"],axis=1)


# In[26]:


data.head()


# In[27]:


from mlxtend.frequent_patterns import apriori, association_rules


# In[28]:


freq1=apriori(data,min_support=0.005,max_len=3, use_colnames=True)
freq1.sort_values("support", ascending = False, inplace = True)


# In[32]:


rules1 = association_rules(freq1, metric = "lift", min_threshold= 1)


# In[33]:


rules1.head()


# In[34]:


rules1.shape


# In[35]:


print(len(rules1))


# In[36]:


rules1.sort_values('lift', ascending = False).head(20)


# In[38]:


freq2=apriori(data,min_support=0.007,max_len=5,use_colnames=True)
freq2.sort_values("support",ascending=True)


# In[43]:


freq2.sort_values("support",ascending=False)


# In[44]:


rules2=association_rules(freq2,"lift",min_threshold=2)


# In[45]:


rules2.head()


# In[46]:


print(len(rules2))


# In[47]:


rules2.shape


# In[ ]:





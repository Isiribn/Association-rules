#!/usr/bin/env python
# coding: utf-8

# In[28]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[29]:


data=pd.read_csv("book.csv")


# In[30]:


data.head()


# In[31]:


data.shape


# In[32]:


data.tail()


# In[33]:


from collections import Counter
item_frequencies = Counter(data)
item_frequencies = sorted(item_frequencies.items(),key = lambda x:x[1])


# In[34]:


# Storing frequencies and items in separate variables 
frequencies = list(reversed([i[1] for i in item_frequencies]))
items = list(reversed([i[0] for i in item_frequencies]))


# In[35]:


from mlxtend.frequent_patterns import association_rules,apriori


# In[36]:


import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

plt.bar(x=list(range(0,11)),height = frequencies[0:11],color='rgbkymc')
plt.xticks(list(range(0,11),),items[0:11])
plt.xlabel("items")
plt.ylabel("Count")


# In[37]:


frequent_books = apriori(data, min_support = 0.005, max_len = 3, use_colnames = True)
frequent_books.sort_values("support", ascending = False, inplace = True)


# In[38]:


rules = association_rules(frequent_books, metric = "lift", min_threshold= 1)


# In[39]:


rules.head(10)


# In[40]:


print(len(rules))


# In[41]:


rules.sort_values('lift', ascending = False).head(20)


# In[42]:


frequent_books_2 = apriori(data, min_support = 0.005, max_len = 5, use_colnames = True)
rules2 = association_rules(frequent_books_2, metric = "lift", min_threshold= 1)


# In[43]:


rules2.head(10)


# In[44]:


print(len(rules2))


# In[45]:


rules_r = rules.sort_values('lift', ascending = False)


# In[46]:


rules_r


# In[47]:


print(len(rules_r))


# In[48]:


frequent_books_3 = apriori(data, min_support = 0.075, max_len = 4, use_colnames = True)
rules3 = association_rules(frequent_books_2, metric = "lift", min_threshold= 1)


# In[49]:


print(len(rules3))


# In[50]:


rules3.head(25)


# In[ ]:





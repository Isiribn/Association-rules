#!/usr/bin/env python
# coding: utf-8

# # Association rules for Grocery dataset

# In[2]:


get_ipython().system('pip install mlxtend')


# In[1]:


import pandas as pd
from mlxtend.frequent_patterns import apriori,association_rules
grocery=[]


# In[2]:


with open("groceries.csv") as f:
    grocery=f.read()


# In[3]:


#Split the transaction data with "\n"
grocery=grocery.split("\n")
grocery_list=[]
for i in grocery:
    grocery_list.append(i.split(","))


# In[4]:


all_grocery_list = [i for item in grocery_list for i in item]


# In[5]:


from collections import Counter
item_frequencies = Counter(all_grocery_list)
item_frequencies = sorted(item_frequencies.items(),key = lambda x:x[1])


# In[6]:


# Storing frequencies and items in separate variables 
frequencies = list(reversed([i[1] for i in item_frequencies]))
items = list(reversed([i[0] for i in item_frequencies]))


# In[7]:


# barplot of top 10 

import matplotlib.pyplot as plt

plt.bar(x=list(range(0,11)),height = frequencies[0:11],color='rgbkymc')
plt.xticks(list(range(0,11),),items[0:11])
plt.xlabel("items")
plt.ylabel("Count")


# In[8]:


# Creating Data Frame for the transactions data 
grocery_series  = pd.DataFrame(pd.Series(grocery_list))
#grocery_series = grocery_series.iloc[:9835,:] # removing the last empty transaction

grocery_series.columns = ["transactions"]

# creating a dummy columns for the each item in each transactions ... Using column names as item name
X = grocery_series['transactions'].str.join(sep='*').str.get_dummies(sep='*')

frequent_itemsets = apriori(X, min_support=0.005, max_len=3,use_colnames = True)


# In[9]:


# Most Frequent item sets based on support 
frequent_itemsets.sort_values('support',ascending = False,inplace=True)
plt.bar(x = list(range(1,11)),height = frequent_itemsets.support[1:11],color='rgmyk')
plt.xticks(list(range(1,11)),frequent_itemsets.itemsets[1:11])
plt.xlabel('item-sets')
plt.ylabel('support')


# In[12]:


rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules.sort_values(by='confidence',ascending = False,inplace=True)


# In[13]:


rules.head(10)


# In[ ]:





# In[15]:


get_ipython().system('pip install apyori')


# In[16]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from apyori import apriori


# In[20]:


groceries=[]
with open("groceries.csv") as f:
    groceries=f.read()


# In[ ]:


#Split the transaction data with "\n"
grocery=grocery.split("\n")
grocery_list=[]
for i in grocery:
    grocery_list.append(i.split(","))


# In[25]:


association_rules = apriori(grocery_list, min_support=0.0045, min_confidence=0.2, min_lift=3, min_length=2)
association_results = list(association_rules)


# In[32]:


print(len(association_results))


# In[28]:


association_results


# In[30]:


print(association_results[0])


# In[31]:


for item in association_results:

    # first index of the inner list
    # Contains base item and add item
    pair = item[0] 
    items = [x for x in pair]
    print("Rule: " + items[0] + " -> " + items[1])

    #second index of the inner list
    print("Support: " + str(item[1]))

    #third index of the list located at 0th
    #of the third index of the inner list

    print("Confidence: " + str(item[2][0][2]))
    print("Lift: " + str(item[2][0][3]))
    print("=====================================")


# In[ ]:


plt.scatter()


# In[33]:


association_rules1 = apriori(grocery_list, min_support=0.0060, min_confidence=0.3, min_lift=4, min_length=3)
association_results1 = list(association_rules1)


# In[34]:


print(len(association_results1))


# In[35]:


association_results1


# In[38]:


for item in association_results1:

    # first index of the inner list
    # Contains base item and add item
    pair = item[0] 
    items = [x for x in pair]
    print("Rule: " + items[0] + " -> " + items[1]+","+items[2]+","+items[3])

    #second index of the inner list
    print("Support: " + str(item[1]))

    #third index of the list located at 0th
    #of the third index of the inner list

    print("Confidence: " + str(item[2][0][2]))
    print("Lift: " + str(item[2][0][3]))
    print("=====================================")


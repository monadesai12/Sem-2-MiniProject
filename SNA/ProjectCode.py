#!/usr/bin/env python
# coding: utf-8

# In[3]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import math
import warnings
warnings.filterwarnings('ignore') # Hides warning
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore",category=UserWarning)
sns.set_style("whitegrid") # Plotting style
np.random.seed(7) # seeding random number generator

df = pd.read_csv('amazon.csv')
print(df.head())


# In[4]:


# Describing the Dataset

data = df.copy()
data.describe()


# In[5]:


#  Information : column name with data type
data.info()


# In[8]:


#We need to clean  up the name column by referencing review_id (unique products)
data["review_id"].unique()


# In[9]:


review_id_unique = len(data["review_id"].unique())
print("Number of Unique Review ID: " + str(review_id_unique))


# In[10]:


#Visualizing the distributions of numerical variables:

data.hist(bins=50, figsize=(20,15))
plt.show()


# In[12]:


#we will split it into training set and test sets. Our goal is to train a sentiment analysis classifier.

#we will need to do a stratified split on the reviews score  (star rating)

from sklearn.model_selection import StratifiedShuffleSplit
print("Before {}".format(len(data)))
dataAfter = data.dropna(subset=["star_rating"])
# Removes all NAN in star.rating
print("After {}".format(len(dataAfter)))
dataAfter["star_rating"] = dataAfter["star_rating"].astype(int)

split = StratifiedShuffleSplit(n_splits=5, test_size=0.2)
for train_index, test_index in split.split(dataAfter,
                                           dataAfter["star_rating"]):
    strat_train = dataAfter.reindex(train_index)
    strat_test = dataAfter.reindex(test_index)


# In[13]:


#We need to see if train and test sets were stratified proportionately in comparison to raw data:

print(len(strat_train))
print(len(strat_test))
print(strat_test["star_rating"].value_counts()/len(strat_test))


# In[14]:


#We will use regular expressions to clean out any unfavorable characters in the dataset

reviews = strat_train.copy()
reviews.head()


# In[21]:


print(len(reviews["product_parent"].unique()), len(reviews["review_id"].unique()))
print(reviews.info())


# In[22]:


#Entire training dataset average rating

print(reviews["star_rating"].mean())
asins_count_ix = reviews["product_title"].value_counts().index
plt.subplots(2,1,figsize=(16,12))
plt.subplot(2,1,1)
reviews["product_parent"].value_counts().plot(kind="bar", title="Product Parent Frequency")
plt.subplot(2,1,2)
sns.pointplot(x="product_parent", y="star_rating",  data=reviews)
plt.xticks(rotation=90)
plt.show()


# In[23]:


# Using the features in place, we will build a classifier that can determine a review’s sentiment.

def sentiments(rating):
    if (rating == 5) or (rating == 4):
        return "Positive"
    elif rating == 3:
        return "Neutral"
    elif (rating == 2) or (rating == 1):
        return "Negative"
    
# Add sentiments to the data
strat_train["Sentiment"] = strat_train["star_rating"].apply(sentiments)
strat_test["Sentiment"] = strat_test["star_rating"].apply(sentiments)
print(strat_train["Sentiment"][:20])


# In[ ]:





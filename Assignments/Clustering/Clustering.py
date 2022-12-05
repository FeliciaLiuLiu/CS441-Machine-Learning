#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from scipy.cluster import hierarchy
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# # *Assignment Summary

# You can find a dataset dealing with European employment in 1979 at http://www.dm.unibo.it/~simoncin/EuropeanJobs.html. This dataset gives the percentage of people employed in each of a set of areas in 1979 for each of a set of European countries. Notice this dataset contains only 26 data points. That's fine; it's intended to give you some practice in visualization of clustering.
# 
# 1. Use agglomerative clustering to cluster this data. Produce a dendrogram of this data for each of single link, complete link, and group average clustering. You should label the countries on the axis. What structure in the data does each method expose? You should see dendrograms that "make sense" (at least if you remember some European history), and have interesting differences.
# 
# 2. Using k-means, cluster this dataset. What is a good choice of k for this data and why?

# # 0. Data

# ## 0.1 Description

# You can find a dataset dealing with European employment in 1979 at http://www.dm.unibo.it/~simoncin/EuropeanJobs.html. This dataset gives the percentage of people employed in each of a set of areas in 1979 for each of a set of European countries. Notice this dataset contains only 26 data points. That's fine; it's intended to give you some practice in visualization of clustering.

# ## 0.2 Loading

# In[2]:


df = pd.read_csv("../Clustering-lib/EuropeanJobs.dat", sep='\t', header=0)


# In[3]:


df


# Here is a description of the columns in the data:
# - Country: name of the country
# - Agr: percentage employed in agriculture
# - Min: percentage employed in mining
# - Man: percentage employed in manufacturing
# - PS: percentage employed in power supply industries 
# - Con: percentage employed in construction
# - SI: percentage employed in service industries
# - Fin: percentage employed in finance
# - SPS: percentage employed in social and personal services
# - TC: percentage employed in transport and communications

# In[4]:


feature_cols = ['Agr','Min','Man','PS','Con','SI','Fin','SPS','TC'] 
X = df[feature_cols].values
Y = df['Country'].tolist()


# # 1. Agglomerative Clustering

# # <span style="color:blue">Task 1</span>

# Write a function `single_linkage` that produces a single-link agglomerative clustering. This function should take the data matrix `X` as input, which is a numpy array with the shape of $(N,d)$ where $N$ is the number of samples and $d$ is the number of features. The output of the function should be a linkage matrix. Use the Euclidean distance as a metric. 
# 
# You may find scipy's hierarchical clustering methods (https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html) useful here. The utilization of the `optimal_ordering` option makes interpretations of the resulting trees an easier job.

# In[5]:


def single_linkage(X):
    """
    Produce a single-link agglomerative clustering.
    
        Parameters:
                X (np.array): A numpy array of the shape (N,d) where N is the number of samples and d is the number of features.
                
        Returns:
                single_link (np.array): The single-link agglomerative clustering of X encoded as a linkage matrix.
    """
    
    # your code here
    # raise NotImplementedError
    
    single_link = hierarchy.linkage(X, optimal_ordering=True)
    
    return single_link


# In[6]:


single_link = single_linkage(X)
assert single_link[:,2].min().round(3) == 4.234


# In[7]:


# Next, we will plot the dendogram for the first task.


# In[8]:


single_link = single_linkage(X)
plt.figure(figsize=(12,6), dpi=90)
plt.ylabel("Distance")
plt.title("Agglomerative Clustering of European Jobs - Single Link")
dn_single = hierarchy.dendrogram(single_link, labels=Y)


# # <span style="color:blue">Task 2</span>

# Write a function `complete_linkage` that produces a complete-link agglomerative clustering. This function should take the data matrix `X` as input, which is a numpy array with the shape of $(N,d)$ where $N$ is the number of samples and $d$ is the number of features. The output of the function should be a linkage matrix. Use the Euclidean distance as a metric. 
# 
# You may find scipy's hierarchical clustering methods (https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html) useful here. The utilization of the `optimal_ordering` option makes interpretations of the resulting trees an easier job.

# In[9]:


def complete_linkage(X):
    """
    Produce a complete-link agglomerative clustering.
    
        Parameters:
                X (np.array): A numpy array of the shape (N,d) where N is the number of samples and d is the number of features.
                
        Returns:
                comp_link (np.array): The complete-link agglomerative clustering of X encoded as a linkage matrix.
    """
    
    # your code here
    # raise NotImplementedError
    comp_link = hierarchy.linkage(X, method='complete' , optimal_ordering=True)
    
    return comp_link


# In[10]:


comp_link = complete_linkage(X)
assert comp_link[:,2].max().round(3) == 72.278


# In[11]:


# Next, we will plot the dendogram for the second task.


# In[12]:


complete_link = complete_linkage(X)
plt.figure(figsize=(12,6), dpi=90)
plt.ylabel("Distance")
plt.title("Agglomerative Clustering of European Jobs - Complete Link")
dn_complete = hierarchy.dendrogram(complete_link,labels=Y)


# # <span style="color:blue">Task 3</span>

# Write a function `group_avg_linkage` that produces an average-link agglomerative clustering. This function should take the data matrix `X` as input, which is a numpy array with the shape of $(N,d)$ where $N$ is the number of samples and $d$ is the number of features. The output of the function should be a linkage matrix. Use the Euclidean distance as a metric. 
# 
# You may find scipy's hierarchical clustering methods (https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html) useful here. The utilization of the `optimal_ordering` option makes interpretations of the resulting trees an easier job.

# In[13]:


def group_avg_linkage(X):
    """
    Produce an average-link agglomerative clustering.
    
        Parameters:
                X (np.array): A numpy array of the shape (N,d) where N is the number of samples and d is the number of features.
                
        Returns:
                avg_link (np.array): The average-link agglomerative clustering of X encoded as a linkage matrix.
    """
    
    # your code here
    # raise NotImplementedError
    avg_link = hierarchy.linkage(X, method='average' , optimal_ordering=True)
    
    return avg_link


# In[14]:


avg_link = group_avg_linkage(X)
assert avg_link[:,2].max().round(3) == 44.172


# In[15]:


# Next, we will plot the dendogram for the third task.


# In[ ]:


average_link = group_avg_linkage(X)
plt.figure(figsize=(12,6), dpi=90)
plt.ylabel("Distance")
plt.title("Agglomerative Clustering of European Jobs - Group Average")
dn_average = hierarchy.dendrogram(average_link,labels=Y)


# # 2. K-Means Clustering

# In this part, we perform the K-Means clustering algorithm on the dataset, and evalute the effect of the parameter $k$ (the number of clusters) on the outcome.
# 
# For this, we use the `KMeans` class from `sklearn.cluster`. You should familiarize yourself with this class. You can find the documentation of it here: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html 
# 
# In the following code, we run the K-Means algorithm for $2 \leq k \leq 25$ clusters.
# 
# <font color='red'><b>Attention:</font> Although you are not implementing this part, for the follow-up quiz of this assignment, you will need to come back here, write some code and do some calculations to get the answer of some questions in the quiz. For now, try to read the documentation for the `KMeans` class and try to understand what the following code is doing.

# In[ ]:


k_list = list(range(2,26))
k_inertias = []
k_scores = []
model_list = []
for k in k_list:
    model = KMeans(n_clusters=k, random_state=12345).fit(X)
    model_list.append(model)
    cluster_assignments = model.labels_
    score = silhouette_score(X, cluster_assignments, metric='euclidean')
    inertia = model.inertia_
    k_scores.append(score)
    k_inertias.append(inertia)


# Now, we plot the sum of square distances of samples to their closest cluster center as a function of $k$, the numebr of clusters. 

# In[ ]:


plt.figure(figsize=(8,4), dpi=120)
plt.title('The Elbow Plot')
plt.xlabel('Number of Clusters')
plt.ylabel('Sum of Square Distances')
_=plt.plot(k_list, k_inertias,'bo--')


# Look at the above *Elbow plot*. Based on this plot, what do you think is a reasonable choice of $k$?

# Next, we plot the so called "*silhouette*" score for the result of the K-Means clustering algorithm for the values of $k$ we implemented above. The silhouette score is a measure of how similar an object is to its cluster compared to other clusters. 
# 
# - Try to learn how the silhouette score is defined. For instance, you can look at this Wikipedia page: [https://en.wikipedia.org/wiki/Silhouette_(clustering)](https://en.wikipedia.org/wiki/Silhouette_%28clustering%29).
# 
# - What is the range of the silhouette score?
# - Is a larger value of the silhouette score better or worse?

# In[ ]:


plt.figure(figsize=(8,4), dpi=120)
plt.title('Silhouette Score vs. Number of clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
_=plt.plot(k_list, k_scores,'bo--')


# Based on the silhouette measure, what do you think is a reasonable value for $k$? Is this the same value that the above elbow plot suggests? Why do think so?

# In[ ]:





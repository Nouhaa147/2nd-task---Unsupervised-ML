#!/usr/bin/env python
# coding: utf-8

# In[84]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.cluster import KMeans


# In[85]:


data = pd.read_csv('Iris.csv')


# In[86]:


data


# In[87]:


data.isnull ==True


# In[88]:


data.describe()


# In[89]:


x = data.iloc[:,1:-1]


# In[90]:


x


# In[91]:


wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', 
                    max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1, 11), wcss)
plt.title('The elbow method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS') 
plt.show()        


# In[92]:


kmeans = KMeans(n_clusters = 3, init = 'k-means++',
                max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(x)


# In[93]:


y_kmeans = np.where(y_kmeans==0, 'Iris-setosa', y_kmeans)
y_kmeans = np.where(y_kmeans=='1', 'Iris-versicolour', y_kmeans)
y_kmeans = np.where(y_kmeans=='2', 'Iris-virginica', y_kmeans)


# In[94]:


data_with_clusters = x.copy()
data_with_clusters["Cluster"] = y_kmeans
print(data_with_clusters.head(5))


# In[95]:


sns.set_style('darkgrid')
sns.barplot(x = data_with_clusters["Cluster"] .unique(),
            y = data_with_clusters["Cluster"] .value_counts(),
            palette=sns.color_palette(["#e74c3c", "#34495e", "#2ecc71"]));


# In[ ]:



     


# In[ ]:





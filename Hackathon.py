# -*- coding: utf-8 -*-
"""
Created on Fri May 31 04:33:43 2019

@author: Dell
"""
# K-Means Clustering

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
data = pd.read_csv('QA_DUMP.csv')

data1 = data.iloc[:, [1,2,3,4,5,6,7]].values
df1 = pd.DataFrame(data1)

ohe = preprocessing.OneHotEncoder(categories='auto')
Y = ohe.fit_transform(df1).toarray()

from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(Y)
    wcss.append(kmeans.inertia_)

# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(Y)

# Visualising the clusters
plt.scatter(Y[y_kmeans == 0, 0], Y[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(Y[y_kmeans == 1, 0], Y[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(Y[y_kmeans == 2, 0], Y[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

new_data_1 = pd.DataFrame(y_kmeans)
new_data_2 = new_data_1.rename(columns = {0: "priority"}) 
PriorityCount=new_data_2.groupby('priority').priority.agg(['count'])
PriorityCountPercent=(new_data_2.groupby('priority').size()/new_data_2['priority'].count())*100

data['priority']=y_kmeans
val1=PriorityCount.iloc[0]['count']
val2=PriorityCount.iloc[1]['count']
val3=PriorityCount.iloc[2]['count']

if val1>val2>val3:
   data['priority']=  data['priority'].replace({0: 'high', 1: 'medium', 2: 'low'})
if val2>val1>val3:
  data['priority']=  data['priority'].replace({0: 'medium', 1: 'high', 2: 'low'})
if val2>val3>val1:
  data['priority']=  data['priority'].replace({0: 'low', 1: 'high', 2: 'medium'})
if val1>val3>val2:
  data['priority']=   data['priority'].replace({0: 'high', 1: 'low', 2: 'medium'})
if val3>val2>val1:
   data['priority']=  data['priority'].replace({0: 'low', 1: 'medium', 2: 'high'})
if val3>val1>val2:
  data['priority']=   data['priority'].replace({0: 'medium', 1: 'low', 2: 'high'})
  

export_excel = data.to_excel ('Classified_QA_DUMP_1.xlsx', index = None, header=True)

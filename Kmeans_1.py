# -*- coding: utf-8 -*-
"""
Created on Thu May 30 21:34:00 2019

@author: Dell
"""

# K-Means Clustering

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Importing the dataset #PROD_DUMP
data= pd.read_excel('QA_DUMP_1.xlsx')
dataset = pd.read_excel('QA_DUMP_1.xlsx')
labelEncoder = LabelEncoder()
labelEncoder.fit(dataset['PlanPlatform'])
dataset['PlanPlatform_new'] = labelEncoder.transform(dataset['PlanPlatform'])
labelEncoder.fit(dataset['TypeofPlan'])
dataset['TypeofPlan_new'] = labelEncoder.transform(dataset['TypeofPlan'])
labelEncoder.fit(dataset['PlanEffDate'])
dataset['PlanEffDate_new'] = labelEncoder.transform(dataset['PlanEffDate'])
labelEncoder.fit(dataset['PlanTermDate'])
dataset['PlanTermDate_new'] = labelEncoder.transform(dataset['PlanTermDate'])
labelEncoder.fit(dataset['Benefits'])
dataset['Benefits_new'] = labelEncoder.transform(dataset['Benefits'])
labelEncoder.fit(dataset['City'])
dataset['City_new'] = labelEncoder.transform(dataset['City'])
labelEncoder.fit(dataset['State'])
dataset['State_new'] = labelEncoder.transform(dataset['State'])
X = dataset.iloc[:, [10,11,16]].values
# y = dataset.iloc[:, 3].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)

# Visualising the clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')

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

PriorityCount['count'].max()

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
  

export_excel = data.to_excel ('Classified_QA_DUMP.xlsx', index = None, header=True)

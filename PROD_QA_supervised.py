# -*- coding: utf-8 -*-
"""
Created on Fri May 31 09:39:47 2019

@author: Dell
"""
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
data = pd.read_excel('Classified_PROD_DUMP_1.xlsx')
X=data.iloc[:, [10]].values
data1 = data.iloc[:, [1,2,3,4,5,6,7]].values
df1 = pd.DataFrame(data1)

ohe = preprocessing.OneHotEncoder(categories='auto')
Y = ohe.fit_transform(df1).toarray()
# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0) #linear
classifier.fit(Y,X)

data2 = pd.read_excel('Classified_QA_DUMP_1.xlsx')
data4 = pd.read_excel('Classified_QA_DUMP_1.xlsx')
data3 = data2.iloc[:, [1,2,3,4,5,6,7]].values
df2 = pd.DataFrame(data3)
ohe = preprocessing.OneHotEncoder(categories='auto')
Z = ohe.fit_transform(df2).toarray()

y_pred = classifier.predict(Z)
pred=pd.DataFrame(y_pred)

data4['priority']=pred
export_excel = data4.to_excel ('supervised_QA_DUMP_1.xlsx', index = None, header=True)
data4['result'] = np.where(data2['priority'] == data4['priority'], 0, 1)

PriorityCount=data4.groupby('result').result.agg(['count'])
PriorityCountPercent=(data4.groupby('result').size()/data4['result'].count())*100

finalresult = pd.DataFrame(PriorityCountPercent)
val1=PriorityCount.iloc[0][0]
val2=PriorityCount.iloc[1][0]
sizes=[val1,val2]

# Data to plot
labels = 'Matched','Mismatched'
colors = ['gold', 'yellowgreen']
explode = (0.1, 0)  # explode 1st slice

# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=140)

plt.axis('equal')
plt.show()



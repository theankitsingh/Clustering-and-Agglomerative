import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from factor_analyzer import FactorAnalyzer

from sklearn.decomposition import PCA as pca
from sklearn.decomposition import FactorAnalysis as fact
from sklearn import cluster as cls
from sklearn.feature_extraction.image import grid_to_graph
from sklearn import tree
from scipy.cluster import hierarchy as hier 


os.chdir('C:\\Users\\ankit\\python')
df = pd.read_csv('Singh_Ankit_Export.txt', sep = '\t')

red_df = df[['NoFTE','NetPatRev','InOperExp','OutOperExp',
             'OperRev','OperInc','AvlBeds','Compensation','MaxTerm']]

pca_result=pca(n_components=9).fit(red_df)
pca_result.explained_variance_

plt.figure(figsize=(7,5))
plt.plot([1,2,3,4,5,6,7,8,9], pca_result.explained_variance_ratio_, '-o')
plt.ylabel('Proportion of Variance Explained')
plt.xlabel('Principal Component')
plt.xlim(0.75,4.25)
plt.ylim(0,1.05)
plt.xticks([1,2,3,4,5,6,7,8,9])
plt.show()

pca_result.components_

fact_Res = FactorAnalyzer()
fact_Res.analyze(red_df, 9 , rotation = "varimax")
fact_Res.loadings


km=cls.KMeans(n_clusters=4).fit(red_df.loc[:,['Compensation','AvlBeds']])
km.labels_

km2=cls.KMeans(n_clusters=2).fit(red_df.loc[:,['Compensation','AvlBeds']])
km2.labels_

import sklearn.metrics as metcs

df.Teaching.unique()
df.TypeControl.unique()
df.DonorType.unique()

df['Teaching'] = df['Teaching'].astype('object') 
df['TypeControl'] = df['TypeControl'].astype('object') 
df['DonorType'] = df['DonorType'].astype('object')


df.Teaching.replace(['Small/Rural','Teaching'],[1,2], inplace=True)
df.TypeControl.replace(['District','Non Profit','City/County','Investor'],[1,2,3,4], inplace=True)
df.DonorType.replace(['Charity', 'Alumni'],[1,2], inplace=True)

df.dtypes


df['Teaching'] = df['Teaching'].astype('category')
df['TypeControl'] = df['TypeControl'].astype('category')
df['DonorType'] = df['DonorType'].astype('category')

cm = metcs.confusion_matrix(df.Teaching,km2.labels_)
print(cm)       

cm1 = metcs.confusion_matrix(df.TypeControl, km.labels_)
print(cm1)       

cm2 = metcs.confusion_matrix(df.DonorType, km2.labels_)
print(cm2)       

plt.matshow(cm)
plt.title('Confusion Matrix')
plt.xlabel('Actual Value')
plt.ylabel('Predicted Value')
plt.xticks([0,1],['Small/Rural','Teaching'])

plt.matshow(cm1)
plt.title('Confusion Matrix')
plt.xlabel('Actual Value')
plt.ylabel('Predicted Value')
plt.xticks([0,1,2,3],['District','Non Profit','City/County','Investor'])


plt.matshow(cm2)
plt.title('Confusion Matrix')
plt.xlabel('Actual Value')
plt.ylabel('Predicted Value')
plt.xticks([0,1], ['Charity', 'Alumni'])
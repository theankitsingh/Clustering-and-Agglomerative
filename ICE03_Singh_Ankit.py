import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans
from sklearn import cluster as cls
from sklearn import metrics
os.chdir("C:\\Users\\ankit\\python")
df = pd.read_table("car.test.frame.txt",sep ='\t')

km= cls.KMeans(n_clusters=5).fit(df.loc[:,['Price','Mileage']])
km.labels_


km1= cls.KMeans(n_clusters=8).fit(df.loc[:,['Price','Mileage']])
km1.labels_

cm = metrics.confusion_matrix(df.Price,km.labels_)
print(cm)

cm1 = metrics.confusion_matrix(df.Mileage,km1.labels_)
print(cm1)
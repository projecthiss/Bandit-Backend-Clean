
from google.colab import drive
drive.mount('/content/gdrive')

import sys

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import pandas as pd

#import csv
#import seaborn as sns
from scipy.cluster import hierarchy

import os

import matplotlib.pyplot as plt

import numpy as np

from importlib import reload

df = pd.read_excel("/content/gdrive/My Drive/Colab Notebooks/Pipeline_Extended/output_Bert_topic_many_withStopwords_topics_v2.xlsx", sheet_name=0, na_values="NaN")

df.head()

# Remove first row
df = df.iloc[1:]

s = df.loc[2][3]
print(s)

s = s.
keyword =s[1]
print(keyword)

df1= df

df1['key 1'] =[]
df1['key 2'] = []
[]
for index, row in df1.iterrows():
   # print(row['c1'], row['c2'])

keys = [0,1,2,3,4,5,6,7,8,9]
for i in keys:
    keywords = []
    for index, row in df1.iterrows():
        keywords.append(row[keys[i]].split("'")[1])
    column_title = "keyword_" + str(i)
    df[column_title]=keywords

df1['keyword_string'] = df1['keyword_0'] + ' ' + df1['keyword_1'] + ' ' + df1['keyword_2'] + ' ' + df1['keyword_3'] + ' ' + df1['keyword_4'] + ' ' + df1['keyword_5'] + ' ' + df1['keyword_6'] + ' ' + df1['keyword_7'] + ' ' + df1['keyword_8'] + ' ' + df1['keyword_9']

df.head()

CountVec = CountVectorizer(ngram_range=(1, 1))

# transform

Count_top_quality_data = CountVec.fit_transform(df1['keyword_string'])

# create dataframe

cv_top_quality_dataframe = pd.DataFrame(Count_top_quality_data.toarray(),
                                        columns=CountVec.get_feature_names())

print(cv_top_quality_dataframe)

clustering_top_quality = hierarchy.linkage(cv_top_quality_dataframe, method='ward', metric='euclidean',
                                           optimal_ordering=False)

clusters=hierarchy.fcluster(clustering_top_quality, 0)
print(clusters)

df1['cluster']=clusters


# try to experiment wiht other algorithms and metrics

keywords_top_quality = df1['keyword_string'].tolist()
print("Data clustered")

df1.head()

for index, row in df1.iterrows():
    df1['description'] = str(row[0])+"_"+row['keyword_string']
df1.to_excel("clusteredData.xlsx")
keywords_top_quality = df1['description'].tolist()
fig, axes = plt.subplots(1, 2, figsize=(60, 400), dpi=150)



plt.title("Dendograms of Top Quality Topics")

dend = hierarchy.dendrogram(clustering_top_quality, ax=axes[1], above_threshold_color='#bcbddc', orientation='right',
                            labels=keywords_top_quality, leaf_font_size=12)

plt.savefig("test.svg")

df1.to_excel("/content/gdrive/My Drive/Colab Notebooks/Pipeline_Extended/topic_cluster.xlsx")
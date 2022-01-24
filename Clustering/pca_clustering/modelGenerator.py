from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans
from kneed import KneeLocator
from numpy import NaN
import pandas as pd
import numpy as np
import os

#Extracting truth values from a discretized version of the dataset
disc_df = pd.read_csv(
        os.getcwd() + '\\dataset\\UNSW_NB15_testing-set.csv',
        encoding = 'utf-8'
    )
disc_df = disc_df.replace(['Normal' , 'Fuzzers', 'Analysis', 'Backdoors', 'DoS', 'Exploits', 'Generic', 'Reconnaissance', 'Shellcode', 'Worms', ''], ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0'])

attacks = disc_df['attack_cat'].values
attack_list = []

for elem in attacks:
    attack_list.append(elem)

truth_labels = np.array(attack_list)

#Best number of cluster for each processed dataframe
cluster_num = []
cluster_num.append(0)
cluster_num.append(0)

for i in range(2, 41):
    #Opening dataframe
    dataframe = pd.read_csv(
        os.getcwd() + '\\pca_clustering\\pca_dataset\\testing-set_PCA_' + str(i) + '.csv',
        encoding = 'utf-8'
    )

    #Computing the best number of clusters for each dataframe
    #and evaluating each model with elbow method
    sse = []
    for k in range (2, 11):
        kmeans_model = KMeans(n_clusters = k)
        kmeans_model.fit(dataframe)
        sse.append(kmeans_model.inertia_)

    kl = KneeLocator(range(2, 11), sse, curve="convex", direction="decreasing")
    cluster_num.append(str(kl.elbow))
    print('Completed evaluation for model ' + str(i))

ari_res = []
ari_res.append(0)
ari_res.append(0)
for i in range (2, 41):
    #Opening dataframe
    dataframe = pd.read_csv(
        os.getcwd() + '\\pca_clustering\\pca_dataset\\testing-set_PCA_' + str(i) + '.csv',
        encoding = 'utf-8'
    )

    #Training a KMeans model
    kmeans_model = KMeans(n_clusters = int(cluster_num[i]))
    kmeans_model.fit(dataframe)
    pred_labels = kmeans_model.labels_

    #Computing ARI
    ARI = adjusted_rand_score(truth_labels, pred_labels)
    ari_res.append(ARI)
    print('Computed ARI for KMeans model n.' + str(i) + ' with value: ' + str(ARI))

print('Done!')

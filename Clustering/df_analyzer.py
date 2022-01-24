from df_reducer import minimize_by_feature_importance
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import rand_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import completeness_score
from sklearn.metrics import fowlkes_mallows_score
from sklearn.metrics import homogeneity_completeness_v_measure
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import Birch
from sklearn.cluster import MiniBatchKMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from numpy import NaN
import pandas as pd
import numpy as np
import os

#Size of the small version of dataframe
SMALL_DF_SIZE = 5000

# Read file csv
dataFrame = pd.read_csv(
    os.getcwd() + '\\dataset\\UNSW_NB15_testing-set.csv',
    encoding='utf-8'
)

# Save dataset features
FEATURES = dataFrame.columns.values
FEATURES = np.append(FEATURES[0:2], FEATURES[5:-2])
N_FEATURES = int(len(FEATURES))

# Create matrix X
X = dataFrame.iloc[:, np.append([0, 1], np.arange(5, N_FEATURES + 3))].values

# Create vector y
y = dataFrame.iloc[:, N_FEATURES + 1].values

# Standardize value of matrix X
X = StandardScaler().fit_transform(X)

# Split X and y for testing and training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Create and fit the random forest classifier
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

# Test random forest
y_pred = clf.predict(X_test)

# Extract sub dataframe with only first ten feature
minDataFrame = minimize_by_feature_importance(clf, FEATURES,
                                              os.getcwd() + '\\dataset\\UNSW_NB15_testing-set.csv', .9)

#Extracting truth values from the dataset
disc_df = pd.read_csv(
        os.getcwd() + '\\dataset\\UNSW_NB15_testing-set.csv',
        encoding = 'utf-8'
    )
disc_df = disc_df.replace(['Normal' , 'Fuzzers', 'Analysis', 'Backdoors', 'DoS', 'Exploits', 'Generic', 'Reconnaissance', 'Shellcode', 'Worms', ''], ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0'])

small_disc_df = disc_df.sample(SMALL_DF_SIZE)

#Extraction for the standard dataframe
attacks = disc_df['attack_cat'].values
attack_list = []

for elem in attacks:
    attack_list.append(elem)

truth_labels = np.array(attack_list)

#Extraction for the small dataframe
attacks = small_disc_df['attack_cat'].values
attack_list = []

for elem in attacks:
    attack_list.append(elem)

small_truth_labels = np.array(attack_list)

#Opening dataframe
dataframe = pd.read_csv(
    os.getcwd() + '\\dataset\\UNSW_NB15_testing-set_REDUCED.csv',
    encoding = 'utf-8'
)

#Extracting small dataframe
small_dataframe = dataframe.sample(SMALL_DF_SIZE)

#Training a KMeans model
kmeans_model = KMeans(n_clusters = 10)
kmeans_model.fit(dataframe)
pred_labels = kmeans_model.labels_
silhouette_preds = kmeans_model.fit_predict(dataframe)

#Computing RI for KMeans model
RI = rand_score(truth_labels, pred_labels)
print('Computed RI for KMeans model with value: ' + str(RI))

#Computing ARI for KMeans model
ARI = adjusted_rand_score(truth_labels, pred_labels)
print('Computed ARI for KMeans model with value: ' + str(ARI))

#Computing CHS for KMeans model
CHS = calinski_harabasz_score(dataframe, pred_labels)
print('Computed CHS for KMeans model with value: ' + str(CHS))

#Computing DBS for KMeans model
DBS = davies_bouldin_score(dataframe, pred_labels)
print('Computed DBS for KMeans model with value: ' + str(DBS))

#Computing CS for KMeans model
CS = completeness_score(truth_labels, pred_labels)
print('Computed CS for KMeans model with value: ' + str(CS))

#Computing FMI for KMeans model
FMI = fowlkes_mallows_score(truth_labels, pred_labels)
print('Computed FMI for KMeans model with value: ' + str(FMI))

#Computing HCVm for KMeans model
homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(truth_labels, pred_labels)
print('Computed HCVm for KMeans model with following results: homogeneity = {}, completeness = {}, v_measure = {}'.format(homogeneity, completeness, v_measure))

#Computing Silhouette Score for KMeans model
score = silhouette_score(dataframe, silhouette_preds)
print("For n_clusters = {}, silhouette score is {}".format('10', score))

#Training a GaussianMixture model
gmm_model = GaussianMixture(n_components = 10, reg_covar=1e-2)
gmm_preds = gmm_model.fit_predict(dataframe)

#Computing RI for GaussianMixture model
RI = rand_score(truth_labels, gmm_preds)
print('Computed RI for GaussianMixture model with value: ' + str(RI))

#Computing ARI for GaussianMixture model
ARI = adjusted_rand_score(truth_labels, gmm_preds)
print('Computed ARI for GaussianMixture model with value: ' + str(ARI))

#Computing CHS for GaussianMixture model
CHS = calinski_harabasz_score(dataframe, gmm_preds)
print('Computed CHS for GaussianMixture model with value: ' + str(CHS))

#Computing DBS for GaussianMixture model
DBS = davies_bouldin_score(dataframe, gmm_preds)
print('Computed DBS for GaussianMixture model with value: ' + str(DBS))

#Computing CS for GaussianMixture model
CS = completeness_score(truth_labels, gmm_preds)
print('Computed CS for GaussianMixture model with value: ' + str(CS))

#Computing FMI for GaussianMixture model
FMI = fowlkes_mallows_score(truth_labels, gmm_preds)
print('Computed FMI for GaussianMixture model with value: ' + str(FMI))

#Computing HCVm for GaussianMixture model
homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(truth_labels, gmm_preds)
print('Computed HCVm for GaussianMixture model with following results: homogeneity = {}, completeness = {}, v_measure = {}'.format(homogeneity, completeness, v_measure))

#Computing Silhouette Score for GaussianMixture model
score = silhouette_score(dataframe, gmm_preds)
print("For n_clusters = {}, silhouette score is {}".format('10', score))

#Training a MiniBatchKMeans model
MBM = MiniBatchKMeans(n_clusters = 10, batch_size=1024)
MBM.fit(dataframe)
pred_labels = MBM.labels_
silhouette_preds = MBM.fit_predict(dataframe)

#Computing RI for MiniBatchKMeans model
RI = rand_score(truth_labels, pred_labels)
print('Computed RI for MiniBatchKMeans model with value: ' + str(RI))

#Computing ARI for MiniBatchKMeans model
ARI = adjusted_rand_score(truth_labels, pred_labels)
print('Computed ARI for MiniBatchKMeans model with value: ' + str(ARI))

#Computing CHS for MiniBatchKMeans model
CHS = calinski_harabasz_score(dataframe, pred_labels)
print('Computed CHS for MiniBatchKMeans model with value: ' + str(CHS))

#Computing DBS for MiniBatchKMeans model
DBS = davies_bouldin_score(dataframe, pred_labels)
print('Computed DBS for MiniBatchKMeans model with value: ' + str(DBS))

#Computing CS for MiniBatchKMeans model
CS = completeness_score(truth_labels, pred_labels)
print('Computed CS for MiniBatchKMeans model with value: ' + str(CS))

#Computing FMI for MiniBatchKMeans model
FMI = fowlkes_mallows_score(truth_labels, pred_labels)
print('Computed FMI for MiniBatchKMeans model with value: ' + str(FMI))

#Computing HCVm for MiniBatchKMeans model
homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(truth_labels, pred_labels)
print('Computed HCVm for MiniBatchKMeans model with following results: homogeneity = {}, completeness = {}, v_measure = {}'.format(homogeneity, completeness, v_measure))

#Computing Silhouette Score for MiniBatchKMeans model
score = silhouette_score(dataframe, silhouette_preds)
print("For n_clusters = {}, silhouette score is {}".format('10', score))

#Training a Birch model
birchModel = Birch(n_clusters = 10)
birch_preds = birchModel.fit_predict(small_dataframe)

#Computing RI for Birch model
RI = rand_score(small_truth_labels, birch_preds)
print('Computed RI for Birch model with value: ' + str(RI))

#Computing ARI for Birch model
ARI = adjusted_rand_score(small_truth_labels, birch_preds)
print('Computed ARI for Birch model with value: ' + str(ARI))

#Computing CHS for Birch model
CHS = calinski_harabasz_score(small_dataframe, birch_preds)
print('Computed CHS for Birch model with value: ' + str(CHS))

#Computing DBS for Birch model
DBS = davies_bouldin_score(small_dataframe, birch_preds)
print('Computed DBS for Birch model with value: ' + str(DBS))

#Computing CS for Birch model
CS = completeness_score(small_truth_labels, birch_preds)
print('Computed CS for Birch model with value: ' + str(CS))

#Computing FMI for Birch model
FMI = fowlkes_mallows_score(small_truth_labels, birch_preds)
print('Computed FMI for Birch model with value: ' + str(FMI))

#Computing HCVm for Birch model
homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(small_truth_labels, birch_preds)
print('Computed HCVm for Birch model with following results: homogeneity = {}, completeness = {}, v_measure = {}'.format(homogeneity, completeness, v_measure))

#Computing Silhouette Score for Birch model
score = silhouette_score(small_dataframe, birch_preds)
print("For n_clusters = {}, silhouette score is {}".format('10', score))

#Training a AgglomerativeClustering model
aggModel = AgglomerativeClustering(n_clusters=10)
aggModel.fit(small_dataframe)
pred_labels = aggModel.labels_
silhouette_preds = aggModel.fit_predict(small_dataframe)

#Computing RI for AgglomerativeClustering model
RI = rand_score(small_truth_labels, pred_labels)
print('Computed RI for AgglomerativeClustering model with value: ' + str(RI))

#Computing ARI for AgglomerativeClustering model
ARI = adjusted_rand_score(small_truth_labels, pred_labels)
print('Computed ARI for AgglomerativeClustering model with value: ' + str(ARI))

#Computing CHS for AgglomerativeClustering model
CHS = calinski_harabasz_score(small_dataframe, pred_labels)
print('Computed CHS for AgglomerativeClustering model with value: ' + str(CHS))

#Computing DBS for AgglomerativeClustering model
DBS = davies_bouldin_score(small_dataframe, pred_labels)
print('Computed DBS for AgglomerativeClustering model with value: ' + str(DBS))

#Computing CS for AgglomerativeClustering model
CS = completeness_score(small_truth_labels, pred_labels)
print('Computed CS for AgglomerativeClustering model with value: ' + str(CS))

#Computing FMI for AgglomerativeClustering model
FMI = fowlkes_mallows_score(small_truth_labels, pred_labels)
print('Computed FMI for AgglomerativeClustering model with value: ' + str(FMI))

#Computing HCVm for AgglomerativeClustering model
homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(small_truth_labels, pred_labels)
print('Computed HCVm for AgglomerativeClustering model with following results: homogeneity = {}, completeness = {}, v_measure = {}'.format(homogeneity, completeness, v_measure))

#Computing Silhouette Score for AgglomerativeClustering model
score = silhouette_score(small_dataframe, silhouette_preds)
print("For n_clusters = {}, silhouette score is {}".format('10', score))

print('Done!')

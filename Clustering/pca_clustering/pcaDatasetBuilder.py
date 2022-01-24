from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# CSV reading
dataFrame = pd.read_csv(
    os.getcwd() + '\\dataset\\UNSW_NB15_testing-set.csv',
    encoding = 'utf-8'
)

# Saving dataset's features
FEATURES = dataFrame.columns.values
N_FEATURES = int(len(FEATURES)) - 2

# Init X matrix
X = dataFrame.iloc[:, np.append([0, 1], np.arange(5, N_FEATURES))].values

# Init target vector
targetVector = dataFrame.iloc[:, N_FEATURES + 1]

# Scaling datas
X = StandardScaler().fit_transform(X)

# Computing PCA using a range of components
# between 2 and min(n_samples, n_features)=40

for COMPONENTS in range(2, 41):
    pca = PCA(n_components = COMPONENTS)
    principalComponents = pca.fit_transform(X)

    #Plotting PCA variance/n_components graphic
    PCA_VARIANCE = pca.explained_variance_ratio_.cumsum()
    plt.plot(range(1, COMPONENTS + 1),
        PCA_VARIANCE, marker = 'o')
    
    plt.xlim(0, COMPONENTS + 1)
    plt.ylim(0, 1)
    plt.grid()

    plt.xlabel('Components')
    plt.ylabel('Variance (%)')

    plt.savefig('pca_clustering/pca_plots/PCA_' + str(COMPONENTS) + '.png')

    #Creating new dataframe form source using the principal components
    sourceDataFrame = pd.DataFrame(principalComponents,
        columns = range(COMPONENTS))
    outputDataFrame = pd.concat([sourceDataFrame, targetVector], axis = 1)
    outputDataFrame.to_csv('pca_clustering/pca_dataset/testing-set_PCA_' + str(COMPONENTS) + '.csv')

    print('Completed PCA with ' + str(COMPONENTS) + 'components.')

print('Dataset and plots genetarated.')
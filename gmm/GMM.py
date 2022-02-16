
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from scipy import stats
from sklearn.metrics.cluster import adjusted_rand_score, rand_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.mixture import GaussianMixture


def GMMpca(zflowDF, maxcluster, scoretype=None):
    celltypelist = zflowDF.CellType.values
    totalDF = zflowDF.drop(columns=['CellType', 'pSTAT5'])  # Creating matrix that will be used in GMM model
    clusternumb = np.arange(1, maxcluster)  # Amount of clusters
    scores = np.zeros_like(clusternumb, dtype=float)
    zflowDF = zflowDF['CellType'].values  # Obtaining celltypes

    kf = KFold(n_splits=10)  # Cross validation for amount of splits
    for kk in range(len(clusternumb)):
        print(kk)
        GMM = GaussianMixture(n_components=clusternumb[kk], covariance_type='full', tol=1e-6, max_iter=5000)

        best_rand = cross_val_score(GMM, totalDF, celltypelist, cv=kf, scoring=scoretype, n_jobs=-1)
        scores[kk] = np.mean(best_rand)

    return pd.DataFrame({"Cluster": clusternumb, "Score": scores})


def runPCA(dataDF):

    arr = np.arange(1, 4, 1)
    totalvar = np.zeros([len(arr)])
    celltypelist = dataDF.CellType.values
    totalDF = dataDF.drop(columns=['CellType', 'pSTAT5'])

    # Determining variance explained
    for a in range(len(arr)):
        pca = PCA(n_components=arr[a])
        newform = pca.fit_transform(totalDF)
        totalvar[a] = sum(pca.explained_variance_ratio_)

    return arr, totalvar

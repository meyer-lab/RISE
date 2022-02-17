
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, cross_validate
from sklearn.mixture import GaussianMixture


def LLscorer(estimator, X, _):
    return np.mean(estimator.score(X))


def cvGMM(zflowDF, maxcluster):
    celltypelist = zflowDF['CellType'].values # Obtaining celltypes
    totalDF = zflowDF.drop(columns=['CellType', 'pSTAT5'])  # Creating matrix that will be used in GMM model
    clusternumb = np.arange(1, maxcluster)  # Amount of clusters
    LLscores = np.zeros_like(clusternumb, dtype=float)
    randScores = np.zeros_like(clusternumb, dtype=float)

    kf = KFold(n_splits=10)  # Cross validation for amount of splits

    for i in range(len(clusternumb)):
        print("Cluster Number:",clusternumb[i])
        GMM = GaussianMixture(n_components=clusternumb[i], covariance_type='full', tol=1e-6, max_iter=5000)

        scores = cross_validate(GMM, totalDF, celltypelist, cv=kf, scoring={"LL": LLscorer, "rand":"rand_score"}, n_jobs=-1)

        LLscores[i] = np.mean(scores["test_LL"])
        randScores[i] = np.mean(scores["test_rand"])

    return pd.DataFrame({"Cluster": clusternumb, "ll_score": LLscores, "rand_score": randScores})


def runPCA(dataDF):

    arr = np.arange(1, 4, 1)
    totalvar = np.zeros([len(arr)])
    celltypelist = dataDF.CellType.values
    totalDF = dataDF.drop(columns=['CellType', 'pSTAT5'])

    # Determining variance explained
    for i in range(len(arr)):
        pca = PCA(n_components=arr[i])
        newform = pca.fit_transform(totalDF)
        totalvar[i] = sum(pca.explained_variance_ratio_)

    return arr, totalvar

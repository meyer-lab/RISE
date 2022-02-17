
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, cross_validate
from sklearn.mixture import GaussianMixture
from sklearn.mixture._gaussian_mixture import _estimate_gaussian_parameters


def LLscorer(estimator, X, _):
    return np.mean(estimator.score(X))


def cvGMM(zflowDF, maxcluster):
    celltypelist = zflowDF['CellType'].values  # Obtaining celltypes
    totalDF = zflowDF.drop(columns=['CellType', 'pSTAT5'])  # Creating matrix that will be used in GMM model
    clusternumb = np.arange(1, maxcluster)  # Amount of clusters
    LLscores = np.zeros_like(clusternumb, dtype=float)
    randScores = np.zeros_like(clusternumb, dtype=float)

    kf = KFold(n_splits=10)  # Cross validation for amount of splits

    for i in range(len(clusternumb)):
        print("Cluster Number:", clusternumb[i])
        GMM = GaussianMixture(n_components=clusternumb[i], covariance_type='full', tol=1e-6, max_iter=5000)

        scores = cross_validate(GMM, totalDF, celltypelist, cv=kf, scoring={"LL": LLscorer, "rand": "rand_score"}, n_jobs=-1)

        LLscores[i] = np.mean(scores["test_LL"])
        randScores[i] = np.mean(scores["test_rand"])

    return pd.DataFrame({"Cluster": clusternumb, "ll_score": LLscores, "rand_score": randScores})


def probGMM(zflowDF, maxcluster, cellperexp):
    celltypelist = zflowDF['CellType'].values  # Obtaining celltypes
    totalDF = zflowDF.drop(columns=['CellType', 'pSTAT5'])  # Creating matrix that will be used in GMM model
    clusternumb = np.arange(1, maxcluster)  # Amount of clusters
    probDF = []
    for i in range(cellperexp):
        indDF = totalDF.loc[i * 50:50 * (i + 1)]

        for j in range(len(clusternumb)):
            print("Cluster Number:", clusternumb[j], "Experiment Number:", i + 1)
            GMM = GaussianMixture(n_components=clusternumb[i], covariance_type='full', tol=1e-6, max_iter=5000)
            GMM.fit(indDF)
            # prob = GMM.predict_proba(indDF)
            # probDF.append(pd.DataFrame({"Cluster": clusternumb[j], "Probalities": prob.flatten(), "Experiment": i + 1}))

            log_prob_norm, log_resp = GMM._estimate_log_prob_resp(indDF)
            nk, means, covariances = _estimate_gaussian_parameters(indDF.values,
                                     log_resp, reg_covar=1e-6, covariance_type='full')

        if i == 0:
            break    # break here

    return nk, means, covariances


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

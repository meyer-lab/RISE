import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_validate
from sklearn.mixture import GaussianMixture
from sklearn.mixture._gaussian_mixture import _estimate_gaussian_parameters


def LLscorer(estimator, X, _):
    return np.mean(estimator.score(X))


def cvGMM(zflowDF, maxcluster):
    celltypelist = zflowDF["Cell Type"].values  # Obtaining celltypes
    totalDF = zflowDF.drop(
        columns=["Cell Type", "pSTAT5", "Valency", "index", "Time", "Date", "Dose", "Ligand"]
    )  # Creating matrix that will be used in GMM model
    clusternumb = np.arange(1, maxcluster)  # Amount of clusters
    LLscores = np.zeros_like(clusternumb, dtype=float)
    randScores = np.zeros_like(clusternumb, dtype=float)

    kf = KFold(n_splits=10)  # Cross validation for amount of splits

    for i in range(len(clusternumb)):
        print("Cluster Number:", clusternumb[i])
        GMM = GaussianMixture(n_components=clusternumb[i], covariance_type="full", tol=1e-6, max_iter=5000)

        scores = cross_validate(GMM, totalDF, celltypelist, cv=kf, scoring={"LL": LLscorer, "rand": "rand_score"}, n_jobs=-1)

        LLscores[i] = np.mean(scores["test_LL"])
        randScores[i] = np.mean(scores["test_rand"])

    return pd.DataFrame({"Cluster": clusternumb, "ll_score": LLscores, "rand_score": randScores})


def probGMM(zflowDF, n_clusters: int, cellperexp: int):
    """Use the GMM responsibilities matrix to develop means and covariances for each experimental condition.

    NOTE: This method currently assumes there is a constant number of samples per experiment.

    Args:
        zflowDF (pandas.DataFrame): DF w/z-scored epitopes values w/pSTAT5 and celltypes
        n_clusters (int): The number of clusters to run the analysis for.
        cellperexp (int): Amount of cells wanted for GMM for each experiment

    Returns:
        numpy.array: Matrix of data sample numbers across each condition.
        numpy.array: Matrix of means across each condition.
        numpy.array: Tensor of covariance matrices across each condition.
    """
    celltypelist = zflowDF["Cell Type"].values  # Obtaining celltypes
    totalDF = zflowDF.drop(
        columns=["Cell Type", "pSTAT5", "Valency", "index", "Time", "Date", "Dose", "Ligand"]
    )  # Creating matrix that will be used in GMM model
    statDF = zflowDF.drop(columns=["Cell Type", "Valency", "index", "Time", "Date", "Dose", "Ligand"])  # Creating matrix that includes pSTAT5

    # Fit the GMM with the full dataset
    GMM = GaussianMixture(n_components=n_clusters, covariance_type="full", max_iter=5000, verbose=20)
    GMM.fit(totalDF)
    _, log_resp = GMM._estimate_log_prob_resp(totalDF)  # Get the responsibilities

    # Setup storage
    nk = list()
    means = list()
    covariances = list()

    # Loop over separate conditions
    for i in range(0, totalDF.shape[0], cellperexp):
        indDF = statDF.loc[i : i + cellperexp - 1]
        resp_ind = log_resp[i : i + cellperexp, :]
        assert indDF.shape[0] == cellperexp  # Check my indexing
        assert indDF.shape[0] == resp_ind.shape[0]  # Check my indexing

        output = _estimate_gaussian_parameters(indDF.values, resp_ind, reg_covar=1e-6, covariance_type="full")
        nk.append(output[0])
        means.append(output[1])
        covariances.append(output[2])

    nk = np.stack(nk)
    means = np.stack(means)
    covariances = np.stack(covariances)
    return nk, means, covariances

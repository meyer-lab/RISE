import pandas as pd
import numpy as np
import xarray as xa
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.mixture._gaussian_mixture import _estimate_gaussian_parameters
from .tensor import markerslist


def LLscorer(estimator, X, _):
    """ Calculates the scores of the GMM vs. original predicted clusters"""
    return np.mean(estimator.score(X))


def cvGMM(zflowDF, maxcluster: int):
    """ Runs CV on GMM model with score and rand score for multiple clusters"""
    X = zflowDF.drop(
        columns=["Cell Type", "pSTAT5", "index", "Time", "Date", "Dose", "Ligand"]
    )  # Creating matrix that will be used in GMM model

    cv = KFold(10, shuffle=True)
    GMM = GaussianMixture(covariance_type="full", tol=1e-6, max_iter=5000)

    scoring = {"LL": LLscorer, "rand": "rand_score"}
    grid = {'n_components': np.arange(1, maxcluster)}
    grid_search = GridSearchCV(GMM, param_grid=grid, scoring=scoring, cv=cv, refit=False, n_jobs=-1)
    grid_search.fit(X, zflowDF["Cell Type"].values)
    results = grid_search.cv_results_

    return pd.DataFrame({"Cluster": results["param_n_components"], "ll_score": results["mean_test_LL"], "rand_score": results["mean_test_rand"]})


def probGMM(zflowDF, n_clusters: int):
    """Use the GMM responsibilities matrix to develop means and covariances for each experimental condition.

    Args:
        zflowDF (pandas.DataFrame): DF w/z-scored epitopes values w/pSTAT5 and celltypes
        n_clusters (int): The number of clusters to run the analysis for.

    Returns:
        numpy.array: Matrix of data sample numbers across each condition.
        numpy.array: Matrix of means across each condition.
        numpy.array: Tensor of covariance matrices across each condition.
    """
    # Fit the GMM with the full dataset
    GMM = GaussianMixture(n_components=n_clusters, covariance_type="full", max_iter=5000, verbose=20)
    GMM.fit(zflowDF[markerslist])
    _, log_resp = GMM._estimate_log_prob_resp(zflowDF[markerslist])  # Get the responsibilities
    assert log_resp.shape[0] == zflowDF.shape[0]  # Check shapes

    doses = zflowDF["Dose"].unique()
    ligand = zflowDF["Ligand"].unique()
    times = zflowDF["Time"].unique()

    # Setup storage
    nk = xa.DataArray(np.full((n_clusters, len(ligand), len(doses), len(times)), np.nan),
                      coords={"Cluster": np.arange(n_clusters), "Ligand": ligand, "Dose": doses, "Time": times})
    means = xa.DataArray(np.full((n_clusters, len(markerslist), len(ligand), len(doses), len(times)), np.nan),
                         coords={"Cluster": np.arange(n_clusters), "Markers": markerslist, "Ligand": ligand, "Dose": doses, "Time": times})
    covariances = list()

    # Loop over separate conditions
    for name, cond_cells in zflowDF.groupby(["Ligand", "Dose", "Time"]):
        idxx = (zflowDF["Ligand"] == name[0]) & (zflowDF["Dose"] == name[1]) & (zflowDF["Time"] == name[2])
        output = _estimate_gaussian_parameters(cond_cells[markerslist].values, np.exp(log_resp[idxx, :]), 1e-6, "full")

        nk.loc[:, name[0], name[1], name[2]] = output[0]
        means.loc[:, :, name[0], name[1], name[2]] = output[1]
        covariances.append(output[2])

    return nk, means, np.stack(covariances)

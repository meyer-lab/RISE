import numpy as np
import pandas as pd
import seaborn as sns
from .common import subplotLabel, getSetup
from gmm.tensor import minimize_func, gen_points_GMM


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((12, 6), (3, 4))

    # Add subplot labels
    subplotLabel(ax)
    blob_DF = make_synth_pic(magnitude=80)
    plot_synth_pic(blob_DF, t=0, ax=ax[0])
    plot_synth_pic(blob_DF, t=6, ax=ax[1])
    plot_synth_pic(blob_DF, t=12, ax=ax[2])
    plot_synth_pic(blob_DF, t=19, ax=ax[3])

    rank = 6
    n_cluster = 6
    blob_xarray = make_blob_tensor(blob_DF)

    maximizedNK, optCP, optPTfactors, _, _, preNormOptCP = minimize_func(blob_xarray, rank=rank, n_cluster=n_cluster, maxiter=1000)

    for i in np.arange(0, 4):
        print(i)
        points = gen_points_GMM(maximizedNK, preNormOptCP, optPTfactors, i * 6, 0, 0)
        points_DF = pd.DataFrame({"Cluster": points[1], "X": points[0][:, 0], "Y": points[0][:, 1]})
        sns.scatterplot(data=points_DF, x="X", y="Y", hue="Cluster", palette="tab10", ax=ax[i + 8])
        ax[i+8].set(xlim=(-.2, 2.2), ylim=(-.2, 2.2))


    ax[4].bar(np.arange(1, maximizedNK.size + 1), maximizedNK)
    xlabel = "Cluster"
    ylabel = "NK Value"
    ax[4].set(xlabel=xlabel, ylabel=ylabel)

    # CP factors
    cmpCol = [f"Cmp. {i}" for i in np.arange(1, rank + 1)]
    clustArray = np.arange(1, n_cluster + 1)
    coords = {"Cluster": clustArray, "Dimension": ["X", "Y"], "Time": blob_xarray.coords["Time"]}
    maximizedFactors = [pd.DataFrame(optCP.factors[ii], columns=cmpCol, index=coords[key]) for ii, key in enumerate(coords)]

    for i in range(0, len(maximizedFactors)):
        sns.heatmap(data=maximizedFactors[i], ax=ax[i + 5])

    return f


def make_blob_art(mean, cov, size, time, label):
    """Makes a labeled DF for storing blob art"""
    X = np.random.multivariate_normal(mean=mean, cov=cov, size=size) / 10
    return pd.DataFrame({"X": X[:, 0], "Y": X[:, 1], "Time": time, "Label": label})


palette = {"Ground": "khaki",
           "Trunk": "sienna", 
           "Leaf":"limegreen",
           "Sun": "yellow"}


def make_synth_pic(magnitude):
    """Makes blob of points depicting beach scene with sinusoidally moving sun"""
    ts = np.arange(0, 101)
    blob_DF = None

    for t in ts:
        blob_DF = pd.concat([blob_DF, make_blob_art((10, 2), cov=[[20, 0], [0, 0.5]], size=int(1 * magnitude), time=t, label="Ground")])
        blob_DF = pd.concat([blob_DF, make_blob_art((4, 6), cov=[[0.05, 0], [0, 2]], size=int(0.5 * magnitude), time=t, label="Trunk")])
        blob_DF = pd.concat([blob_DF, make_blob_art((16, 6), cov=[[0.05, 0], [0, 2]], size=int(0.5 * magnitude), time=t, label="Trunk")])
        blob_DF = pd.concat([blob_DF, make_blob_art((4, 10), cov=[[1, 0], [0, 1]], size=int(0.5 * magnitude), time=t, label="Leaf")])
        blob_DF = pd.concat([blob_DF, make_blob_art((16, 10), cov=[[1, 0], [0, 1]], size=int(0.5 * magnitude), time=t, label="Leaf")])
        blob_DF = pd.concat([blob_DF, make_blob_art((10, 14 + 8 * np.sin(t * 2 * np.pi / (25))), cov=[[0.5, 0], [0, 0.5]], size=int(1 * magnitude), time=t, label="Sun")])

    return blob_DF


def plot_synth_pic(blob_DF, t, ax):
    """Plots snthetic data at a time point"""
    sns.scatterplot(data=blob_DF.loc[blob_DF["Time"] == t], x="X", y="Y", hue="Label", palette=palette, legend=False, ax=ax)
    ax.set(xlim=(-.2, 2.2), ylim=(-.2, 2.2))


def make_blob_tensor(blob_DF):
    """Makes blob art into 3D tensor with points x coordinate x time as dimensions"""
    times = len(blob_DF.Time.unique())
    points = blob_DF.shape[0] / times
    blob_DF["Points"] = np.tile(np.arange(points, dtype=int), times)
    blob_DF = blob_DF.drop("Label", axis=1)
    blob_xa = blob_DF.set_index(["Points", "Time"]).to_xarray().to_array(dim="Dimension")
    blob_xa = blob_xa.expand_dims(["Throwaway 1", "Throwaway 2"])
    blob_xa = blob_xa.transpose("Dimension", "Points", "Time", "Throwaway 1", "Throwaway 2")
    return blob_xa

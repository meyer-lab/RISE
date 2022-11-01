"""
Creating synthetic data and running tGMM to calculate NK and factors
"""
import numpy as np
import pandas as pd
import seaborn as sns
from .common import subplotLabel, getSetup, add_ellipse
from gmm.tensor import optimal_seed


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((12, 6), (3, 4))

    # Add subplot labels
    subplotLabel(ax)
    blob_DF = make_synth_pic(magnitude=60)

    for i in np.arange(0, 4):
        plot_synth_pic(blob_DF, t=i * 3, ax=ax[i])

    rank = 3
    n_cluster = 6
    blobXA = make_blob_tensor(blob_DF)
    colorpal = sns.color_palette("tab10", n_cluster)

    _, _, fit = optimal_seed(
        10, blobXA, rank=rank, n_cluster=n_cluster
    )
    fac = fit[0]

    points_all, points_y = fac.sample(n_samples=200)

    for i in np.arange(0, 4):
        points_DF = pd.DataFrame(
            {"Cluster": points_y[:, i * 3, 0, 0], "X": points_all[0, :, i * 3, 0, 0], "Y": points_all[1, :, i * 3, 0, 0]}
        )
        sns.scatterplot(
            data=points_DF,
            x="X",
            y="Y",
            hue="Cluster",
            palette="tab10",
            ax=ax[i + 8],
            s=5,
        )
        add_ellipse(
            i * 3,
            0,
            0,
            fac,
            "X",
            "Y",
            n_cluster,
            ax[i + 8],
            colorpal,
            "beach",
        )
        ax[i + 8].set(
            xlim=(-1.2, 1.2),
            ylim=(-1.2, 1.2),
            title="Time: " + str(i * 3) + " - ULTRA",
        )

    ax[4].bar(np.arange(1, fac.nk.size + 1), fac.norm_NK())
    xlabel = "Cluster"
    ylabel = "NK Value"
    ax[4].set(xlabel=xlabel, ylabel=ylabel)

    # CP factors
    facXA = fac.get_factors_xarray(blobXA)
    DimCol = [f"Dimension{i}" for i in np.arange(1, len(facXA) + 1)]

    for i in range(0, 3):
        sns.heatmap(data=facXA[DimCol[i]], xticklabels=facXA[DimCol[i]].coords[facXA[DimCol[i]].dims[1]].values,
                    yticklabels=facXA[DimCol[i]].coords[facXA[DimCol[i]].dims[0]].values, ax=ax[i + 5])

    return f


def make_blob_art(mean, cov, size, time, label):
    """Makes a labeled DF for storing blob art"""
    X = np.random.multivariate_normal(mean=mean, cov=cov, size=size) / 10
    return pd.DataFrame({"X": X[:, 0], "Y": X[:, 1], "Time": time, "Label": label})


palette = {"Ground": "khaki", "Trunk": "sienna", "Leaf": "limegreen", "Sun": "yellow"}


def make_synth_pic(magnitude):
    """Makes blob of points depicting beach scene with sinusoidally moving sun"""
    ts = np.arange(10)
    blob_DF = None

    for t in ts:
        blob_DF = pd.concat(
            [
                blob_DF,
                make_blob_art(
                    (0, -8),
                    cov=[[20, 0], [0, 0.5]],
                    size=int(1 * magnitude),
                    time=t,
                    label="Ground",
                ),
            ]
        )
        blob_DF = pd.concat(
            [
                blob_DF,
                make_blob_art(
                    (-6, -4),
                    cov=[[0.05, 0], [0, 2]],
                    size=int(0.5 * magnitude),
                    time=t,
                    label="Trunk",
                ),
            ]
        )
        blob_DF = pd.concat(
            [
                blob_DF,
                make_blob_art(
                    (6, -4),
                    cov=[[0.05, 0], [0, 2]],
                    size=int(0.5 * magnitude),
                    time=t,
                    label="Trunk",
                ),
            ]
        )
        blob_DF = pd.concat(
            [
                blob_DF,
                make_blob_art(
                    (-6, 0),
                    cov=[[1, 0], [0, 1]],
                    size=int(0.5 * magnitude),
                    time=t,
                    label="Leaf",
                ),
            ]
        )
        blob_DF = pd.concat(
            [
                blob_DF,
                make_blob_art(
                    (6, 0),
                    cov=[[1, 0], [0, 1]],
                    size=int(0.5 * magnitude),
                    time=t,
                    label="Leaf",
                ),
            ]
        )
        blob_DF = pd.concat(
            [
                blob_DF,
                make_blob_art(
                    (0, 4 + 8 * np.sin(t * 2 * np.pi / (25))),
                    cov=[[0.5, 0], [0, 0.5]],
                    size=int(1 * magnitude),
                    time=t,
                    label="Sun",
                ),
            ]
        )

    return blob_DF


def plot_synth_pic(blob_DF, t, ax):
    """Plots snthetic data at a time point"""
    sns.scatterplot(
        data=blob_DF.loc[blob_DF["Time"] == t],
        x="X",
        y="Y",
        hue="Label",
        palette=palette,
        legend=False,
        ax=ax,
        s=5,
    )
    ax.set(
        xlim=(-1.2, 1.2),
        ylim=(-1.2, 1.2),
        title="Time: " + str(t) + " - Synthetic Beach Scene",
    )


def make_blob_tensor(blob_DF):
    """Makes blob art into 3D tensor with points x coordinate x time as dimensions"""
    times = len(blob_DF.Time.unique())
    points = blob_DF.shape[0] / times
    blob_DF["Points"] = np.tile(np.arange(points, dtype=int), times)
    blob_DF = blob_DF.drop("Label", axis=1)
    blob_xa = (
        blob_DF.set_index(["Points", "Time"]).to_xarray().to_array(dim="Dimension")
    )
    blob_xa = blob_xa.expand_dims(["Throwaway 1", "Throwaway 2"])
    blob_xa = blob_xa.transpose(
        "Dimension", "Points", "Time", "Throwaway 1", "Throwaway 2"
    )
    return blob_xa

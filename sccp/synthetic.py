"""
Common functions to plot and create synethetic data for Parafac2
"""
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xa


def synthXA(magnitude, type):
    """Makes blob of points depicting beach scene with sinusoidally moving sun"""
    ts = np.arange(10)
    blob_DF = None
    if type == "beach":
        blob_means = [(5, -8), (0, -5), (8, -5), (0, -2), (8, -2)]
        blob_covar = [
            [[20, 0], [0, 0.5]],
            [[0.05, 0], [0, 2]],
            [[0.05, 0], [0, 2]],
            [[1, 0], [0, 1]],
            [[1, 0], [0, 1]],
        ]
        blob_label = ["Ground", "Trunk1", "Trunk2", "Leaf1", "Leaf2"]

        for t in ts:
            for i in range(6):
                if i != 5:
                    blob_DF = pd.concat(
                        [
                            blob_DF,
                            make_blob_art(
                                blob_means[i],
                                cov=blob_covar[i],
                                size=int(1 * magnitude),
                                time=t,
                                label=blob_label[i],
                            ),
                        ]
                    )
                else:
                    blob_DF = pd.concat(
                        [
                            blob_DF,
                            make_blob_art(
                                (4, 8 * np.sin(t * 2 * np.pi / (25))),
                                cov=[[1, 0], [0, 1]],
                                size=int(1 * magnitude),
                                time=t,
                                label="Sun",
                            ),
                        ]
                    )
    elif type == "movingcovariance":
        blob_means = [(-5, -5), (3, 3), (-5, 5), (2, -5)]
        blob_covar = [[3, 0], [0, 3]]
        blob_planet = ["Planet1", "Planet2", "Planet3", "Planet4", "Planet5"]

        for t in ts:
            for i in range(len(blob_planet)):
                if i == 0 or i == 1:
                    blob_DF = pd.concat(
                        [
                            blob_DF,
                            make_blob_art(
                                blob_means[i],
                                cov=blob_covar,
                                size=int(1 * magnitude),
                                time=t,
                                label=blob_planet[i],
                            ),
                        ]
                    )
                elif i == 2 or i == 3:
                    blob_DF = pd.concat(
                        [
                            blob_DF,
                            make_blob_art(
                                blob_means[i],
                                cov=[[0.5 + 0.75 * t, 0], [0, 0.5 + 0.75 * t]],
                                size=int(1 * magnitude),
                                time=t,
                                label=blob_planet[i],
                            ),
                        ]
                    )
                else:
                    blob_DF = pd.concat(
                        [
                            blob_DF,
                            make_blob_art(
                                (0, 8 * np.sin(t * 2 * np.pi / (25))),
                                cov=blob_covar,
                                size=int(1 * magnitude),
                                time=t,
                                label=blob_planet[i],
                            ),
                        ]
                    )

    elif type == "dividingclusters":
        blob_means = [(-3, 0), (7, 0), (0, 8), (0, -5)]
        blob_covar = [[2, 0], [0, 2]]
        blob_planet = ["Planet1", "Planet2", "Planet3", "Planet4", "Planet5", "Planet6"]

        for t in ts:
            for i in range(len(blob_planet)):
                if i == 0 or i == 1 or i == 2 or i == 3:
                    blob_DF = pd.concat(
                        [
                            blob_DF,
                            make_blob_art(
                                blob_means[i],
                                cov=blob_covar,
                                size=int(1 * magnitude),
                                time=t,
                                label=blob_planet[i],
                            ),
                        ]
                    )
                elif i == 4:
                    blob_DF = pd.concat(
                        [
                            blob_DF,
                            make_blob_art(
                                (0, 2 * np.sin(t * 2 * np.pi / (25))),
                                cov=blob_covar,
                                size=int(1 * magnitude),
                                time=t,
                                label=blob_planet[i],
                            ),
                        ]
                    )
                elif i == 5:
                    blob_DF = pd.concat(
                        [
                            blob_DF,
                            make_blob_art(
                                (0, -2 * np.sin(t * 2 * np.pi / (25))),
                                cov=blob_covar,
                                size=int(1 * magnitude),
                                time=t,
                                label=blob_planet[i],
                            ),
                        ]
                    )

    blobXA, celltypeXA = make_blob_tensor(blob_DF)
    blobXA.name = "data"
    celltypeXA.name = "Cell Type"

    return xa.merge([blobXA, celltypeXA], compat="no_conflicts"), blob_DF


def make_blob_art(mean, cov, size, time, label):
    """Makes a labeled DF for storing blob art"""
    total_synth = np.zeros((11, size))
    for i in range(11):
        X = np.random.multivariate_normal(mean=mean, cov=cov, size=size) / 10
        total_synth[i, :] = X[:, 0].flatten()

    return pd.DataFrame(
        {
            "A": total_synth[0, :],
            "B": total_synth[1, :],
            "C": total_synth[2, :],
            "D": total_synth[3, :],
            "E": total_synth[4, :],
            "F": total_synth[5, :],
            "G": total_synth[6, :],
            "H": total_synth[7, :],
            "I": total_synth[8, :],
            "J": total_synth[9, :],
            "X": total_synth[10, :],
            "Y": X[:, 1],
            "Time": time,
            "Cell Type": label,
        }
    )


def plot_synth_pic(blob_DF, t, palette, type, ax):
    """Plots snthetic data at a time point"""
    sns.scatterplot(
        data=blob_DF.loc[blob_DF["Time"] == t],
        x="X",
        y="Y",
        hue="Cell Type",
        palette=palette,
        ax=ax,
        s=5,
    )
    if type == "beach":
        xlim = (-0.5, 1.5)
        ylim = (-1.2, 1.2)
    elif type == "movingcovariance":
        xlim = (-1.2, 1.2)
        ylim = (-1.2, 1.5)
    elif type == "dividingclusters":
        xlim = (-0.8, 1.2)
        ylim = (-1.0, 1.3)

    ax.set(
        xlim=xlim,
        ylim=ylim,
        title="Time: " + str(t) + " - Synthetic Scene",
    )


def make_blob_tensor(blob_DF):
    """Makes blob art into 3D tensor with points x coordinate x time as dimensions"""
    times = len(blob_DF.Time.unique())
    points = blob_DF.shape[0] / times
    blob_DF["Cell"] = np.tile(np.arange(points, dtype=int), times)
    blob_xa = blob_DF.set_index(["Cell", "Time"]).to_xarray()
    celltypeXA = blob_xa["Cell Type"]
    blob_xa = blob_xa.drop_vars(["Cell Type"])
    blob_xa = blob_xa.to_array(dim="Dimension")

    return blob_xa.transpose(), celltypeXA.transpose()

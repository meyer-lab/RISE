"""
Common functions to plot and create synethetic data for ULTRA
"""
import numpy as np
import pandas as pd
import seaborn as sns
from .common import add_ellipse

def make_synth_pic(magnitude, type):
    """Makes blob of points depicting beach scene with sinusoidally moving sun"""
    ts = np.arange(10)
    blob_DF = None
    if type == "beach":
        blob_means = [(0, -8), (-6, -4), (6, -4), (-6, -0), (6, 0)]
        blob_covar = [[[20, 0], [0, 0.5]], [[0.05, 0], [0, 2]], [[0.05, 0], [0, 2]], 
                  [[1, 0], [0, 1]],[[1, 0], [0, 1]]]
        blob_label = ["Ground", "Trunk", "Trunk", "Leaf", "Leaf"]
    
        for t in ts:
            for i in range(6):
                if i != 5:
                    blob_DF = pd.concat(
                    [blob_DF,make_blob_art(blob_means[i],cov=blob_covar[i],
                    size=int(1 * magnitude),time=t,label=blob_label[i])])
                else: 
                    blob_DF = pd.concat(
                    [blob_DF,make_blob_art((0, 8 * np.sin(t * 2 * np.pi / (25))),
                    cov=[[0.5 + 0.5*t, 0], [0, 0.5 + 0.5*t]],
                    size=int(1 * magnitude),time=t,label="Sun")])
    elif type == "movingcovariance":
        blob_means = [(-5, -5), (5, 5), (-5, 5), (5, -5)]
        blob_covar = [[3, 0], [0, 3]]
        blob_planet = ["Planet1", "Planet2", "Planet3", "Planet4", "Planet5"]
    
        for t in ts:
            for i in range(len(blob_planet)):
                if i == 0 or i == 1:
                    blob_DF = pd.concat(
                    [blob_DF,make_blob_art(blob_means[i],cov=blob_covar,
                    size=int(1 * magnitude),time=t,label=blob_planet[i])])
                elif i == 2 or i == 3:
                    blob_DF = pd.concat(
                    [blob_DF,make_blob_art(blob_means[i],cov=[[0.5 + 0.75*t, 0], [0, 0.5 + 0.75*t]],
                    size=int(1 * magnitude),time=t,label=blob_planet[i])])
                else: 
                    blob_DF = pd.concat(
                    [blob_DF,make_blob_art((0, 8 * np.sin(t * 2 * np.pi / (25))),
                    cov=blob_covar,
                    size=int(1 * magnitude),time=t,label=blob_planet[i])])
                    
    elif type == "dividingclusters":
        blob_means = [(-4, 0), (4, 0), (0, 8), (0,-8)]
        blob_covar = [[2, 0], [0, 2]]
        blob_planet = ["Planet1", "Planet2", "Planet3", "Planet4", "Planet5", "Planet6"]
                    
        for t in ts:  
            for i in range(len(blob_planet)):     
                if i == 0 or i == 1 or i == 2 or i == 3:
                    blob_DF = pd.concat(
                        [blob_DF,make_blob_art(blob_means[i],cov=blob_covar,
                        size=int(1 * magnitude),time=t,label=blob_planet[i])])
                elif i == 4:
                    blob_DF = pd.concat(
                            [blob_DF,make_blob_art((0, 2*np.sin(t * 2 * np.pi / (25))),
                            cov=blob_covar,
                            size=int(1 * magnitude),time=t,label=blob_planet[i])])
                elif i == 5:
                    blob_DF = pd.concat(
                        [blob_DF,make_blob_art((0, -2*np.sin(t * 2 * np.pi / (25))),
                        cov=blob_covar,
                        size=int(1 * magnitude),time=t,label=blob_planet[i])])
            
    return blob_DF


def make_blob_art(mean, cov, size, time, label):
    """Makes a labeled DF for storing blob art"""
    X = np.random.multivariate_normal(mean=mean, cov=cov, size=size) / 10
    return pd.DataFrame({"X": X[:, 0], "Y": X[:, 1], "Time": time, "Label": label})


def plot_synth_pic(blob_DF, t, palette, ax):
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
        title="Time: " + str(t) + " - Synthetic Scene",
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


def scatterRecapitulated(fac, n_cluster, ax):
    """Plots recapitulated data points based on factors"""
    colorpal = sns.color_palette("tab10", n_cluster)
    points_all, points_y = fac.sample(n_samples=200)
    for i in np.arange(0, 3):
        points_DF = pd.DataFrame(
            {
                "Cluster": points_y[:, i * 3, 0, 0].astype(int),
                "X": points_all[0, :, i * 3, 0, 0],
                "Y": points_all[1, :, i * 3, 0, 0],
            }
        )

        sns.scatterplot(
            data=points_DF,
            x="X",
            y="Y",
            hue="Cluster",
            palette="tab10",
            ax=ax[i + 9],
            s=5,
        )

        add_ellipse(i * 3, 0, 0, fac, "X", "Y", n_cluster, ax[i + 9], colorpal, "synthetic")
        ax[i + 9].set(
            xlim=(-1.2, 1.2), ylim=(-1.2, 1.2), title="Time: " + str(i * 3) + " - ULTRA"
        )

    return


def plotFactors_synthetic(facXA, DimCol, n_cluster, ax):
    """Plots factor for dimension, time, and cluster"""
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    yticks = [[f"Clst: {i}" for i in np.arange(1, n_cluster + 1)],
               ["X", "Y"],
               [f"Time: {i}" for i in np.arange(0, 9 + 1)]]
    xticks = facXA[DimCol[0]].coords[facXA[DimCol[0]].dims[1]].values
    
    for i in range(0,3):
        sns.heatmap(
                data=facXA[DimCol[i]],
                xticklabels=xticks, yticklabels=yticks[i],
                ax=ax[i + 4],cmap=cmap,vmin=-1,vmax=1)
            
        ax[i + 4].set_title("Mean Factors")
        ax[i + 4].tick_params(axis="y", rotation=0)

    return


def plotCovFactors_synthetic(fac, blobXA, DimCol, n_cluster, ax):
    """Plots covarinace factors for dimension, time, and cluster"""
    cov_fac = fac.get_covariance_factors(blobXA)
    covSig = cov_fac[DimCol[1]].to_numpy()
    cmap = sns.cubehelix_palette(as_cmap=True)
    
    yticks = [[f"Clst: {i}" for i in np.arange(1, n_cluster + 1)], blobXA.coords[blobXA.dims[0]], 
               [f"Time: {i}" for i in np.arange(0, 9 + 1)]]
    xticks = cov_fac[DimCol[0]].coords[cov_fac[DimCol[0]].dims[1]].values
    covfactors_place = [0, 2]
    
    for i in range(len(covfactors_place)):
        sns.heatmap(data=cov_fac[DimCol[covfactors_place[i]]],
            xticklabels=xticks, yticklabels=yticks[covfactors_place[i]],
            ax=ax[i + 7], cmap=cmap, vmin=0, vmax=1)
        ax[i + 7].set_title("Covariance Factors")
        ax[i + 7].tick_params(axis="y", rotation=0)     
      
    for i in range(fac.rank):
        cov_signal = pd.DataFrame(
            covSig[:, :, i] @ covSig[:, :, i].T,
            columns=yticks[1], index=yticks[1])
        sns.heatmap(data=cov_signal, ax=ax[i + 12],cmap=cmap, vmin=0,vmax=1)
        ax[i + 12].set(title="Covariance Factor - Cmp. " + str(i + 1))

    return


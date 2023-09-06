import seaborn as sns
from matplotlib import gridspec, pyplot as plt
import umap.plot
import numpy as np


def plotDrugUMAP(drugs, decomp, totaldrugs, points, axs):
    """Scatterplot of UMAP visualization weighted by condition"""
    subset = np.random.choice(a=[False, True], size=len(totaldrugs), p=[.75, .25])
    for i, drugz in enumerate(drugs):
        drugList = np.where(np.asarray(totaldrugs == drugz), drugz, "Z Other Drugs")
        umap.plot.points(
            points, labels=drugList, ax=axs[i], color_key_cmap="tab20", subset_points=subset, show_legend=True)
        axs[i].set(
            title=decomp + "-Based Decomposition",
        ylabel="UMAP2",
        xlabel="UMAP1")

    return


def plotGeneUMAP(genes, decomp, points, dataDF, axs):
    """Scatterplot of UMAP visualization weighted by gene"""
    cmap = sns.color_palette("ch:s=-.2,r=.6", as_cmap=True)
    subset = np.random.choice(a=[False, True], size=len(dataDF[genes[0]].values), p=[.75, .25])
    for i, genez in enumerate(genes):
        geneList = dataDF[genez].to_numpy()
        geneList = geneList / np.max(np.abs(geneList))
        psm = plt.pcolormesh([[0, 1], [0, 1]], cmap=cmap)
        plot = umap.plot.points(points, values=geneList, cmap=cmap, subset_points=subset, ax=axs[i])
        colorbar= plt.colorbar(psm, ax=plot)
        axs[i].set(
            title=genez + "-" + decomp + "-Based Decomposition",
            ylabel="UMAP2",
            xlabel="UMAP1")

    return

def plotCmpUMAP(cmp, factors, pf2Points, allP, ax):
    """Scatterplot of UMAP visualization weighted by
    projections for a component and cell state"""
    weightedProjs = allP @ factors[1]
    weightedProjs = weightedProjs[:, cmp-1]
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    weightedProjs = weightedProjs / np.max(np.abs(weightedProjs))
    subset = np.random.choice(a=[False, True], size=np.shape(weightedProjs)[0], p=[.75, .25])
    subset[np.argmax(np.abs(weightedProjs))] = True # Ensure largest value is -1 or 1
    psm = plt.pcolormesh([[-1, 1],[-1, 1]], cmap=cmap)
    plot = umap.plot.points(pf2Points, values=weightedProjs, cmap=cmap, subset_points=subset, ax=ax)
    colorbar= plt.colorbar(psm, ax=plot, label="Cell Specific Weight")
    ax.set(
        ylabel="UMAP2",
        xlabel="UMAP1",
        title="Cmp. " + str(cmp))
    
def plotUMAP_obslabel(labels, pf2Points, ax):
    """Scatterplot of UMAP visualization labeled by cell type or other obs column"""
    umap.plot.points(pf2Points, 
                        labels = labels, 
                        color_key_cmap='Paired', 
                        ax=ax)
    ax.set(
        ylabel="UMAP2",
        xlabel="UMAP1",
        title="Pf2-Based Decomposition: Label " + str(labels.name))
    
    
def plotLabelAllUMAP(conditions, points, ax):
    """Scatterplot of UMAP visualization weighted by condition or cell type"""
    umap.plot.points(
        points, labels=conditions, ax=ax, color_key_cmap="tab20", show_legend=True)
    ax.set(
        title="Pf2-Based Decomposition",
        ylabel="UMAP2",
        xlabel="UMAP1")
    
    
def plotCellTypeUMAP(points, data, ax):
    """Plots UMAP labeled by cell type"""
    subset = np.random.choice(a=[False, True], size=len(data["Cell Type"].values), p=[.75, .25])
    umap.plot.points(points, labels=data["Cell Type"].values, subset_points=subset, ax=ax)
    ax.set(
        ylabel="UMAP2",
        xlabel="UMAP1")

    
def plotCmpPerCellType(weightedprojs, cmp, ax, outliers = True):
    """Boxplot of weighted projections for one component across cell types"""
    cmpName = "Cmp. "+str(cmp)
    sns.boxplot(data=weightedprojs[[cmpName, "Cell Type"]], x=cmpName, y="Cell Type", showfliers = outliers, ax=ax)
    maxvalue = np.max(np.abs(ax.get_xticks()))
    ax.set(xlim=(-maxvalue, maxvalue), xlabel="Cell Specific Weight")
    ax.set_title(cmpName)
    
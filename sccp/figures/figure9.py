"""Factors from scib paper"""
import numpy as np
from .common import (subplotLabel, getSetup, 
    plotFactors, flattenData, 
    plotLabelAllUMAP, plotCellCount)
from ..imports.scib import import_scib_data
from ..parafac2 import parafac2_nd
import umap

def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((15, 15), (3, 2))

    # Add subplot labels
    subplotLabel(ax)

    data, celltypes = import_scib_data(dataname="ImmuneHuman")
    # data, celltypes = import_scib_data(dataname="ImmuneHumanMouse")
    # data, celltypes = import_scib_data(dataname="Stimulation1")
    # data, celltypes = import_scib_data(dataname="Stimulation2")
    # data, celltypes = import_scib_data(dataname="Pancreas")
    
    rank = 2
    _, factors, projs, _ = parafac2_nd(
        data,
        rank=rank,
        random_state=1,
    )
    
    plotFactors(factors, data, ax[0:3], reorder=(0, 2), trim=(2,), saveGenes=False)
    
    dataDF = flattenData(data)
    
     # UMAP dimension reduction
    pf2Points = umap.UMAP(random_state=1).fit(np.concatenate(projs, axis=0))

    plotFactors(factors, data, ax[0:3], trim=(2,))
    
    plotLabelAllUMAP(dataDF["Condition"].values, pf2Points, ax[3])
    plotLabelAllUMAP(celltypes, pf2Points, ax[4])
    
    plotCellCount(dataDF, celltypes, ax[5])
    
    return f


    # exp = [import_scib_data(dataname="ImmuneHuman"), import_scib_data(dataname="ImmuneHumanMouse")] 
    #        import_scib_data(dataname="Stimulation1"), 
    # import_scib_data(dataname="Stimulation2"), import_scib_data(dataname="Pancreas")]
    
    
    # print(celltypes)
    # print(len(celltypes))




    # a
    
    # for i in range(len(exp)):
    #     rank = 1

    #     _, factors, projs, _ = parafac2_nd(
    #         exp[i][0],
    #         rank=rank,
    #         random_state=1,
    #     )
        
    #     dataDF, projDF, _ = flattenData(exp[i][0], factors, projs)
    #     dataDF["Cell Type"] = exp[i][1]
        
    #     celltypeDF = dataDF.groupby(["Cell Type", "Drug"]).size().reset_index(name="Count") 
    #     totalCellCount = dataDF.groupby(["Drug"]).size().values
        
    #     for j, drug in enumerate(np.unique(dataDF["Drug"].values)):
    #         df = celltypeDF.loc[celltypeDF["Drug"] == drug] 
    #         perc = df["Count"].values / np.sum(df["Count"].values)
            
    #         celltypeDF.loc[celltypeDF["Drug"] == drug, "Count"] = perc
            
    #         print(celltypeDF)
            
            

    #     sns.swarmplot(data=celltypeDF, x="Drug", y="Count", hue="Cell Type", ax=ax[i])
        
        
    
        
    # print(len(totalCellCount))
    # print(celltypeDF)
    # celltypeDF["Count"] = celltypeDF["Count"] / np.repeat(totalCellCount, len(dataDF.groupby(["Cell Type"]).size()))
    # print(celltypeDF)
    
    
    
    # def plotDistAllDrug(df, rank, ax):
    # """Plots all Normalized Centroid Distance for all drugs for Pf2 and PCA"""
    # ax.set(title="All Drugs: Rank = " + str(rank))
    
    #  # UMAP dimension reduction
    # pf2Points = umap.UMAP(random_state=1).fit(np.concatenate(projs, axis=0))

    # subset = np.random.choice(a=[False, True], size=len(celltypes), p=[.93, .07])
    # # for i, drugz in enumerate(celltype):
    # #     drugList = np.where(np.asarray(totaldrugs == drugz), drugz, "Other Drugs")
    # umap.plot.points(pf2Points, labels=celltypes, ax=ax[3], color_key_cmap="tab20", show_legend=True, subset_points=subset)
    
    # umap.plot.points(pf2Points, labels=dataDF["Drug"].values, ax=ax[4], color_key_cmap="tab20", show_legend=True, subset_points=subset)
        # axs[i].set(
        #     title=decomp + "-Based Decomposition",
        # ylabel="UMAP2",
        # xlabel="UMAP1")

    # plotFactors(factors, data, ax[0:3], trim=(2,))
    
    return f
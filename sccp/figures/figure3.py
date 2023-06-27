"""
Parafac2 implementation on PBMCs treated wtih PopAlign/Thompson drugs: investigating UMAP
"""
import numpy as np
from .common import (
    subplotLabel,
    getSetup,
    flattenData,
    plotDrugUMAP,
    plotGeneUMAP,
    plotCmpUMAP,
)
from matplotlib import gridspec, pyplot as plt
import umap.plot
import matplotlib
from ..imports.scRNA import ThompsonXA_SCGenes
from ..parafac2 import parafac2_nd
import umap
from sklearn.decomposition import PCA


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((6, 5), (2, 2))

    # Add subplot labels
    subplotLabel(ax)


    # Import of single cells: [Drug, Cell, Gene]
    data = ThompsonXA_SCGenes(offset=1.0)
    rank = 25
    _, factors, projs, _ = parafac2_nd(
        data,
        rank=rank,
        random_state=1,
        verbose=True,
    )

    dataDF, projDF, _ = flattenData(data, factors, projs)

    # UMAP dimension reduction
    pf2Points = umap.UMAP(random_state=1).fit(np.concatenate(projs, axis=0))

    # PCA dimension reduction
    pc = PCA(n_components=rank)
    pcaPoints = pc.fit_transform(data.unfold())
    pcaPoints = umap.UMAP(random_state=1).fit(pcaPoints)

    # genes = ["GNLY", "NKG7"]
    # plotGeneUMAP(genes, "Pf2", pf2Points, dataDF, ax[0:2])
    # plotGeneUMAP(genes, "PCA", pcaPoints, dataDF, ax[2:4])

    # # Find cells associated with drugs
    # drugs = [
    #     "Triamcinolone Acetonide",
    #     "Alprostadil",
    # ]
    # plotDrugUMAP(drugs, "Pf2", dataDF["Drug"].values, pf2Points, ax[4:6])
    # plotDrugUMAP(drugs, "PCA", dataDF["Drug"].values, pcaPoints, ax[6:8])
    
        
    # print(np.random.choice(4, size=(2,3)) @ np.random.choice(5, size=(3,4)))
    # results in 24 matrix
    
    # Proj = cell x (25) 
    # B = 25 x rank
    
    cmp = 25
    cellState = 23
    allP = np.concatenate(projs, axis=0)
    
    weightedProjs = allP[:, cellState-1] * factors[1][cellState-1, cmp-1]


  

    subset = np.random.choice(a=[False, True], size= len(weightedProjs), p=[.95, .05])
    
    psm = plt.pcolormesh([weightedProjs, weightedProjs], cmap=matplotlib.cm.get_cmap('viridis'))
    plot = umap.plot.points(pf2Points, values=weightedProjs, theme='viridis', subset_points= subset, ax=ax[0])
    colorbar= plt.colorbar(psm, ax=plot)

    ax[0].set(
        ylabel="UMAP2",
        xlabel="UMAP1",
        title="Cell State:" + str(cellState)+"-Pf2-Based Decomposition",
    )
    
    
    cmp = 25
    cellState = 25
    allP = np.concatenate(projs, axis=0)
    
    weightedProjs = allP[:, cellState-1] * factors[1][cellState-1, cmp-1]

    subset = np.random.choice(a=[False, True], size= len(weightedProjs), p=[.95, .05])
    
    psm = plt.pcolormesh([weightedProjs, weightedProjs], cmap=matplotlib.cm.get_cmap('viridis'))
    plot = umap.plot.points(pf2Points, values=weightedProjs, theme='viridis', subset_points= subset, ax=ax[1])
    colorbar= plt.colorbar(psm, ax=plot)

    ax[1].set(
        ylabel="UMAP2",
        xlabel="UMAP1",
        title="Cell State:" + str(cellState)+"-Pf2-Based Decomposition")

    
    return f

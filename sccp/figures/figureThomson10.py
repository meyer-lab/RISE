"""
Parafac2 implementation on PBMCs treated wtih PopAlign/Thompson drugs
"""
from .common import (
    subplotLabel,
    getSetup,
    openPf2,
    openUMAP,
    flattenData,
    flattenWeightedProjs,
)
from ..imports.scRNA import ThompsonXA_SCGenes
from ..imports.gating import gateThomsonCells
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from scipy import stats
import seaborn as sns

def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((16, 16), (4, 4))

    # Add subplot labels
    subplotLabel(ax)

    # Import of single cells: [Drug, Cell, Gene]
    data = ThompsonXA_SCGenes()
    dataDF = flattenData(data)

    rank = 30
    dataDF["Cell Type"] = gateThomsonCells()

    _, factors, projs = openPf2(rank, "Thomson")

    weightedProjDF = flattenWeightedProjs(data, factors[1], projs)
    weightedProjDF["Cell Type"] = dataDF["Cell Type"].values
    weightedProjDF.sort_values(by=["Condition", "Cell Type"], inplace=True)
    dataDF.sort_values(by=["Condition", "Cell Type"], inplace=True)
    
    threshold = -.3
    cmp = 20
    Wass_KL_Dist(threshold, cmp, data, dataDF, weightedProjDF, ax[0:2])
    # -.3 for b cells that are more greater

    return f




def Wass_KL_Dist(threshold, cmp, data, dataDF, weightedProjDF, ax, numFactors=10):
    """Finds markers which have average greatest difference from other cells"""
    markerDF = pd.DataFrame([])
    for i, gene in enumerate(data.variable_labels):
        geneAvg = np.mean(dataDF[gene])   
        cmpWeights = weightedProjDF[["Cell Type", "Cmp. "+str(cmp)]]
        if threshold > 0: 
            idxCells = np.argwhere(cmpWeights["Cmp. "+str(cmp)] > threshold)
            idxOpp = np.argwhere(cmpWeights["Cmp. "+str(cmp)] < threshold)
        else:
            idxCells = np.argwhere(cmpWeights["Cmp. "+str(cmp)] < threshold)
            idxOpp = np.argwhere(cmpWeights["Cmp. "+str(cmp)] > threshold)
        
        geneDF = dataDF[["Cell Type", "Condition", gene]]
        
        geneDF[gene] += np.min(dataDF[gene])
        
        targCells = geneDF.iloc[idxCells.flatten(), :] 
        targCells[gene] /= geneAvg
        offCells = geneDF.iloc[idxOpp.flatten(), :] 
        offCells[gene] /= geneAvg
        
        targCells = targCells[gene].values
        offCells = offCells[gene].values
        
        kdeTarg = KernelDensity(kernel='gaussian').fit(np.reshape(targCells, (-1, 1)))
        kdeOffTarg = KernelDensity(kernel='gaussian').fit(np.reshape(offCells, (-1, 1)))
        minVal = np.minimum(np.min(targCells), np.min(offCells)) - 10
        maxVal = np.maximum(np.max(targCells), np.max(offCells)) + 10
        outcomes = np.reshape(np.arange(minVal, maxVal + 1), (-1, 1))
        distTarg = np.exp(kdeTarg.score_samples(outcomes))
        distOffTarg = np.exp(kdeOffTarg.score_samples(outcomes))
        KL_div = stats.entropy(distOffTarg.flatten() + 1e-200, distTarg.flatten() + 1e-200, base=2)
        markerDF = pd.concat([markerDF, pd.DataFrame({"Marker": [gene], "Wasserstein Distance": stats.wasserstein_distance(targCells, offCells), "KL Divergence": KL_div})])
        
    print(markerDF)
        
    for i, distance in enumerate(["Wasserstein Distance", "KL Divergence"]):
        ratioDF = markerDF.sort_values(by=distance)
        sns.barplot(data=ratioDF.tail(numFactors), y="Marker", x=distance, ax=ax[i], color='k')
        ax[i].set(xscale="log")


    ax[0].set(title="Wasserstein Distance")
    ax[1].set(title="KL Divergence")

    return
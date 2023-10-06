"""
Parafac2 implementation on PBMCs treated wtih PopAlign/Thompson drugs
"""
from .common import (
    subplotLabel,
    getSetup,
    openPf2,
    flattenData,
    flattenWeightedProjs,
)
from ..imports.scRNA import ThompsonXA_SCGenes
from ..imports.gating import gateThomsonCells
import pandas as pd
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
    
    genes = ["VPREB3", "CD79A", "FAM111B", "HOPX", "SLC30A3", "MS4A1", "CCNB1", "GADD45A", "SLC40A1", "CDC20", "ITIH3", "CPEB1"]
    cmp = 20
    plotRawGeneWP(cmp, genes, dataDF, weightedProjDF, ax[0:14])


    return f


def plotRawGeneWP(cmp, genes, dataDF, weightedProjDF, axs):
    """Finds markers which have average greatest difference from other cells"""
    markerDF = pd.DataFrame([])
    for i, gene in enumerate(genes):

        df = weightedProjDF[["Cell Type", "Cmp. "+str(cmp)]]
        df[gene] = dataDF[gene].values

        sns.scatterplot(data=df, x="Cmp. "+str(cmp), y=gene, hue="Cell Type", ax=axs[i])
    

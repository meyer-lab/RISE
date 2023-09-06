import seaborn as sns
import numpy as np



def plotCombGO(GO, geneValue, axs):
    """Plots combines score for gene ontology"""
    for i, geneset in enumerate(np.unique(GO["Gene Set"])):
        pvalPlot = sns.barplot(
        data=GO.loc[GO["Gene Set"] == geneset],
        x="Combined Score",
        y="Term",
        ax=axs[i])
        axs[i].set_title(geneValue + "-Genes-" + geneset)
        
def plotPvalGO(GO, geneValue, axs):
    """Plots adjusted p value for gene ontology"""
    for i, geneset in enumerate(np.unique(GO["Gene Set"])):
        pvalPlot = sns.barplot(
        data=GO.loc[GO["Gene Set"] == geneset],
        x="Adjusted P-value",
        y="Term",
        ax=axs[i])
        axs[i].set_title(geneValue + "-Genes-" + geneset)
        pvalPlot.set_xscale("log")

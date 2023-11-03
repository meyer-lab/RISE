import seaborn as sns
import numpy as np
import pandas as pd
from ...geneontology import getGenesfromGO


def plotCombGO(GO, geneValue, axs):
    """Plots combines score for gene ontology"""
    for i, geneset in enumerate(np.unique(GO["Gene Set"])):
        sns.barplot(
            data=GO.loc[GO["Gene Set"] == geneset],
            x="Combined Score",
            y="Term",
            ax=axs[i],
        )
        axs[i].set_title(geneValue + "-Genes-" + geneset)


def plotPvalGO(GO, geneValue, axs):
    """Plots adjusted p value for gene ontology"""
    for i, geneset in enumerate(np.unique(GO["Gene Set"])):
        pvalPlot = sns.barplot(
            data=GO.loc[GO["Gene Set"] == geneset],
            x="Adjusted P-value",
            y="Term",
            ax=axs[i],
        )
        axs[i].set_title(geneValue + "-Genes-" + geneset)
        pvalPlot.set_xscale("log")


def plotGenesFromGO(go_term, C_matrix, component, ax, accession=False):
    """Plot the genes associated with a certain GO term, with bars corresponding to their
    weights in `component`.
    If accesssion == False (default), the function will expect a `go_term` in the format given by
    enrichr analysis with `runGO` (name + accesssion number in one string). If you want to input an
    accession number, send it in as the string go_term and then set `accesssion = TRUE`
    """
    comp_str = "comp_" + str(component)

    # get GO accession number from GO term
    if accession == False:
        go_ac = (
            pd.Series(go_term)
            .str.replace(".*\(", "", regex=True)
            .str.replace("\)", "", regex=True)[0]
        )
    else:
        go_ac = go_term
    list_of_genes_in_go_term = getGenesfromGO(go_ac)
    num_genes_in_go_term = len(list_of_genes_in_go_term)

    alldata = (
        C_matrix.sort_values(by=comp_str)[comp_str]
        .reset_index()
        .rename({"index": "Gene ID"}, axis=1)
    )

    top_value = alldata[comp_str].max()
    bottom_value = alldata[comp_str].min()

    # get only genes that are in that go term
    genes_in_go = alldata[alldata["Gene ID"].isin(list_of_genes_in_go_term)]
    genes_in_go["GO Term"] = (
        go_term
        + "\n"
        + str(num_genes_in_go_term)
        + " Genes in total"
        + "\n"
        + str(100 * len(genes_in_go) / num_genes_in_go_term)
        + "% of genes in GO term shown here"
    )

    sns.barplot(data=genes_in_go, x="Gene ID", y=comp_str, hue="GO Term", ax=ax)
    ax.tick_params(axis="x", rotation=90)
    ax.set_title(
        "Weights of genes from GO term contributing to component " + str(component)
    )
    # make horizontal lines denoting where the top values are for comparison
    ax.axhline(y=top_value, linestyle="--", color="#e76f51")
    ax.axhline(y=bottom_value, linestyle="--", color="#e76f51")

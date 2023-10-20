import numpy as np
import gseapy as gp
import pandas as pd

# Biomart is an alternate option than mygene for getting a list of genes from a GO term
# neither works inspiringly well imho -- sean
# from gseapy import Biomart
# bm = Biomart()
import mygene

mg = mygene.MyGeneInfo()


def geneOntology(cmpNumb: int, dataName, rank, geneAmount, goTerms, geneValue):
    """Plots top Gene Ontology terms for molecular function,
    biological process, cellular component. Uses factors as
    input for function"""

    df = (
        pd.read_csv(f"sccp/data/{dataName}/{dataName}TopBotGenes_Cmp{rank}.csv")
        .rename(columns={"Unnamed: 0": "Gene"})
        .set_index("Gene")
    )
    sort_idx = np.argsort(df.to_numpy(), axis=0)

    # Specifies enrichment sets to run against
    geneSets = [
        "GO_Biological_Process_2021",
        "GO_Cellular_Component_2021",
        "GO_Molecular_Function_2021",
    ]

    genesTop = np.empty((geneAmount), dtype="<U10")
    genesBottom = np.empty((geneAmount), dtype="<U10")

    geneNames = df.index.values[sort_idx[:, cmpNumb - 1]]
    genesTop[:] = np.flip(geneNames[-geneAmount:])
    genesBottom[:] = geneNames[:geneAmount]

    totalCombDF = pd.DataFrame()
    totalPvalDF = pd.DataFrame()
    for i, geneSet in enumerate(geneSets):
        if geneValue == "Overexpressed":
            enrichrGO = runGO(genesTop, geneSets)
        else:
            enrichrGO = runGO(genesBottom, geneSets)

        CombGO = combinedDF(enrichrGO, geneSet, geneValue, goTerms)
        PvalGO = pvalueDF(enrichrGO, geneSet, geneValue, goTerms)

        totalCombDF = pd.concat([totalCombDF, CombGO])
        totalPvalDF = pd.concat([totalPvalDF, PvalGO])

    return totalCombDF, totalPvalDF


def runGO(geneList, geneSets):
    """GSEApy is a Python/Rust implementation of GSEA and wrapper for Enrichr.
    Uses enrichment sets to run against"""
    enrichrGO = gp.enrichr(list(geneList), gene_sets=geneSets, organism="Human").results
    enrichrGO = enrichrGO.set_index("Term", drop=True)

    return enrichrGO


def combinedDF(enrichrGO, geneSet, geneValue, goTerms):
    """Saves combined score for gene terms in DF"""
    # Combined Score
    combined = enrichrGO.loc[enrichrGO["Gene_set"] == geneSet, "Combined Score"]
    combined = combined.sort_values(ascending=False)
    combined = combined.iloc[:goTerms]
    combDF = pd.DataFrame(
        {
            "Term": combined.index.values,
            "Combined Score": combined.values,
            "Expression": geneValue,
            "Gene Set": geneSet,
        }
    )

    return combDF


def pvalueDF(enrichrGO, geneSet, geneValue, goTerms):
    """Saves adjusted p value for gene terms in DF"""
    p_val = enrichrGO.loc[enrichrGO["Gene_set"] == geneSet, "Adjusted P-value"]
    p_val = p_val.sort_values(ascending=True)
    p_val = p_val.iloc[:goTerms]
    pvalDF = pd.DataFrame(
        {
            "Term": p_val.index.values,
            "Adjusted P-value": p_val.values,
            "Expression": geneValue,
            "Gene Set": geneSet,
        }
    )

    return pvalDF


def getGOFromTopGenes(
    C_matrix, component, top_n=30, geneset="GO_Biological_Process_2023"
):
    """Gets a dataframe of GO terms that are enriched in the most positive and most negative (`top_n`) genes
    in a component; using a specified `geneset` and `runGO`
    """
    comp_str = "comp_" + str(component)

    bottom = C_matrix.sort_values(by=comp_str)[comp_str].head(top_n)
    top = C_matrix.sort_values(by=comp_str)[comp_str].tail(top_n)
    top_go = runGO(top.index, geneset)
    bottom_go = runGO(bottom.index, geneset)
    return top_go, bottom_go


def getGenesfromGO(go_accession):
    """Gets a list of the genes associated with a GO term, passed in by accesssion number
    in the format 'GO:########' (str)
    Option to use the Biomart API, which has "limited support" is commented out
    https://gseapy.readthedocs.io/en/latest/gseapy_example.html?highlight=biomart#Biomart-API
    Instead using mygene; which also returns only some of the genes you'd expect it to
    """
    # queries ={'go': [go_accession]}
    # results = bm.query(dataset='hsapiens_gene_ensembl',
    #         attributes=['ensembl_gene_id', 'external_gene_name'],
    #         filters=queries)
    # list_of_genes_in_go_term = results['external_gene_name'].to_numpy()

    queery = mg.query(go_accession, species="human", size=1000)
    genes = queery.get("hits")

    list_of_genes_in_go_term = []
    for dict in genes:
        list_of_genes_in_go_term.append(dict.get("symbol"))

    return list_of_genes_in_go_term

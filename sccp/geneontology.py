import numpy as np
import gseapy as gp
import pandas as pd
import seaborn as sns
from .figures.common import plotCombGO, plotPvalGO

def geneOntology(cmpNumb: int, geneAmount, goTerms, geneValue):
    """Plots top Gene Ontology terms for molecular function, 
    biological process, cellular component. Uses factors as 
    input for function"""
    
    df = pd.read_csv("sccp/data/TopBotGenes_Cmp25.csv").rename(columns={"Unnamed: 0":"Gene"}).set_index("Gene")
    sort_idx = np.argsort(df.to_numpy(), axis=0)
    
    # Specifies enrichment sets to run against
    geneSets = [
        "GO_Biological_Process_2021",
        "GO_Cellular_Component_2021",
        "GO_Molecular_Function_2021"]

    genesTop = np.empty((geneAmount), dtype="<U10")
    genesBottom = np.empty((geneAmount), dtype="<U10")
    
    geneNames = df.index.values[sort_idx[:, cmpNumb-1]]
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
    enrichrGO = gp.enrichr(
            list(geneList),
                gene_sets=geneSets,
                organism="Human"
                ).results
    enrichrGO = enrichrGO.set_index("Term", drop=True)
    
    return enrichrGO
    
def combinedDF(enrichrGO, geneSet, geneValue, goTerms):
    """Saves combined score for gene terms in DF"""
     # Combined Score
    combined = enrichrGO.loc[
            enrichrGO["Gene_set"] == geneSet,
            "Combined Score"
            ]
    combined = combined.sort_values(ascending=False)
    combined = combined.iloc[:goTerms]
    combDF = pd.DataFrame({"Term": combined.index.values, "Combined Score": combined.values, "Expression": geneValue, "Gene Set": geneSet})
    
    return combDF

def pvalueDF(enrichrGO, geneSet, geneValue, goTerms):
    """Saves adjusted p value for gene terms in DF"""
    p_val = enrichrGO.loc[
        enrichrGO["Gene_set"] == geneSet,
        "Adjusted P-value"
            ]
    p_val = p_val.sort_values(ascending=True)
    p_val = p_val.iloc[:goTerms]
    pvalDF = pd.DataFrame({"Term": p_val.index.values, "Adjusted P-value": p_val.values, "Expression": geneValue, "Gene Set": geneSet})
    
    return pvalDF

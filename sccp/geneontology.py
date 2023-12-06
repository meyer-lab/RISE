import pandas as pd
import gseapy as gp
from anndata import AnnData


def geneOntology(X: AnnData, cmpNumb: int) -> pd.DataFrame:
    """Plots top Gene Ontology terms for molecular function,
    biological process, cellular component. Uses factors as
    input for function"""

    # Specifies enrichment sets to run against
    geneSets = [
        "GO_Biological_Process_2021",
        "GO_Cellular_Component_2021",
        "GO_Molecular_Function_2021",
    ]

    geneRank = pd.Series(X.varm["Pf2_C"][:, cmpNumb - 1], index=X.var_names)

    df = gp.prerank(
        geneRank,
        gene_sets=geneSets,
        organism="Human",
        no_plot=True,
        verbose=True,
        threads=30,
    ).res2d

    df = df.loc[df["FWER p-val"] < 0.05]

    return df

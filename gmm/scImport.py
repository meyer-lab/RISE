import numpy as np
import pandas as pd
import csv
from sklearn.decomposition import NMF
from scipy.io import mmread
from scipy.stats import linregress


def import_thompson_drug():
    """Imports cell readings from scRNA-seq data from PBMCs from PopAlign paper"
    -Description of each file-
    drugScreeen : str Path to a sparse matrix
    barcodes : str Path to a .tsv 10X barcodes file
    metafile : str Path to a metadata file. Must contains `cell_barcodes` and `sample_id` columns
    genes : str Path to a .tsv 10X gene file"""

    metafile = pd.read_csv("gmm/data/meta.csv")  # Cell barcodes, sample id of treatment and sample number (33482,3)
    drugScreen = mmread("/opt/andrew/drugscreen.mtx").toarray()  # Sparse matrix of each cell/genes (32738,33482)-(Genes,Cell)
    drugScreen = drugScreen.astype(np.int16)
    barcodes = np.array([row[0] for row in csv.reader(open("gmm/data/barcodes.tsv"), delimiter="\t")])  # Cell barcodes(33482)
    genes = np.array([row[1].upper() for row in csv.reader(open("gmm/data/features.tsv"), delimiter="\t")])  # Gene Names (32738)

    bc_idx = {}
    for i, bc in enumerate(barcodes):  # Attaching each barcode with an index
        bc_idx[bc] = i

    namingList = np.append(genes, ["Drug"])  # Forming column name list
    totalGenes = pd.DataFrame()
    for i, cellID in enumerate(metafile["sample_id"].dropna().unique()):  # Enumerating each experiment/drug
        sample_bcs = metafile[metafile.sample_id == cellID].cell_barcode.values  # Obtaining cell bar code values for a specific experiment
        idx = [bc_idx[bc] for bc in sample_bcs]  # Ensuring barcodes match metafile for an expriment
        geneExpression = drugScreen[:, idx].T  # Obtaining all cells associated with a specific experiment (Cells, Gene)
        cellIdx = np.reshape(np.repeat(cellID, len(sample_bcs)), (-1, 1))  # Connecting drug name with cell
        geneswithbars = np.hstack([geneExpression, cellIdx])  # Combining both matrices
        totalGenes = pd.concat([totalGenes, pd.DataFrame(data=geneswithbars)])  # Setting in a DF
        # Only running a few drugs at time to see if works
        # if cellID == "Etodolac":
        #     break

    totalGenes.columns = namingList  # Naming columns

    return totalGenes, genes


def normalizeGenes(totalGenes, geneNames):
    """Dividing each gene by the total of each gene"""
    sumGenes = totalGenes[geneNames].sum(axis=0).tolist()
    sumGenes = pd.DataFrame(data=np.reshape(sumGenes, (1, -1)), columns=geneNames)

    normG = totalGenes[geneNames].div(sumGenes, axis=1)
    normG = normG[geneNames].replace(np.nan, 0)

    drugs = totalGenes.iloc[:, -1].tolist()
    drugs = np.reshape(drugs, (-1, 1))
    normalizeGenesDF = pd.concat([normG, pd.DataFrame(data=drugs, columns=["Drug"])], axis=1)

    return normalizeGenesDF


def mu_sigma(geneDF, geneNames):
    """Calculates the mu and sigma for every gene and returns means, sigmas, and dataframe filtered for genes expressed in > 0.1% of cells"""
    filtDF = geneDF[geneNames].where(geneDF[geneNames] >= 0, 1, inplace=False)
    filteredGenes = filtDF[geneNames].columns[filtDF.mean(axis=0) > 0.0]
    filtDF = filtDF[geneNames][filteredGenes]

    drugs = geneDF.iloc[:, -1].tolist()
    drugs = np.reshape(drugs, (-1, 1))
    filteredDF = pd.concat([filtDF, pd.DataFrame(data=drugs, columns=["Drug"])], axis=1)

    means = filtDF.mean(axis=0).to_numpy()
    std = filtDF.std(axis=0).to_numpy()
    cv = np.divide(std, means, out=np.zeros_like(std), where=means != 0)

    return filteredDF, np.log10(means), np.log10(cv)


def gene_filter(geneDF, mean, std, offset_value=1.0):
    """Filters genes whos variance are higher than woudl be predicted by a Poisson distribution"""
    slope, intercept, _, _, _ = linregress(mean, std)
    inter = intercept + np.log10(offset_value)

    above_idx = np.where(std > mean * slope + inter)
    finalDF = geneDF.iloc[:, np.append(np.asarray(above_idx).flatten(), geneDF.shape[1] - 1)]

    return finalDF, above_idx


def geneNNMF(X, k=14, verbose=0, maxiteration=2000):
    """Turn gene expression into cells X components"""
    model = NMF(n_components=k, verbose=verbose, max_iter=maxiteration, tol=1e-6)
    X = X.drop("Drug", axis=1)
    W = model.fit_transform(X.to_numpy())

    return model.components_, W

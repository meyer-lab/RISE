import numpy as np
import pandas as pd
import csv
import xarray as xa
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
    drugScreen = drugScreen.astype(np.float64)
    barcodes = np.array([row[0] for row in csv.reader(open("gmm/data/barcodes.tsv"), delimiter="\t")])  # Cell barcodes(33482)
    genes = np.array([row[1].upper() for row in csv.reader(open("gmm/data/features.tsv"), delimiter="\t")])  # Gene Names (32738)

    bc_idx = {}
    for i, bc in enumerate(barcodes):  # Attaching each barcode with an index
        bc_idx[bc] = i

    namingList = np.append(genes, ["Drug"])  # Forming column name list
    totalGenes = pd.DataFrame()
    drugNames = []
    for i, cellID in enumerate(metafile["sample_id"].dropna().unique()):  # Enumerating each experiment/drug
        sample_bcs = metafile[metafile.sample_id == cellID].cell_barcode.values  # Obtaining cell bar code values for a specific experiment
        idx = [bc_idx[bc] for bc in sample_bcs]  # Ensuring barcodes match metafile for an expriment
        geneExpression = drugScreen[:, idx].T  # Obtaining all cells associated with a specific experiment (Cells, Gene)
        cellIdx = np.repeat(cellID, len(sample_bcs)) # Connecting drug name with cell
        drugNames = np.append(drugNames, cellIdx)
        totalGenes = pd.concat([totalGenes, pd.DataFrame(data=geneExpression)])  # Setting in a DF
        # Only running a few drugs at time to see if works
        # if cellID == "Etodolac":
        #     break

    totalGenes.columns = genes # Attaching gene name to each column
    totalGenes["Drug"] = drugNames # Attaching drug name to each cell
    
    return totalGenes.reset_index(drop=True), genes


def normalizeGenes(totalGenes, geneNames):
    """Dividing each gene by the total of each gene"""
    drugNames = totalGenes["Drug"].values
    totalGenes = totalGenes.drop("Drug", axis=1)
    sumGenes = totalGenes.sum(axis=0).values

    sumGenes = pd.DataFrame(data=np.reshape(sumGenes, (1, -1)), columns=geneNames)

    normG = totalGenes.div(sumGenes, axis=1)
    normG = normG.replace(np.nan, 0)
    
    normG["Drug"] = drugNames

    return normG 


def mu_sigma(geneDF):
    """Calculates the mu and sigma for every gene and returns means, sigmas, and dataframe filtered for genes expressed in > 0.1% of cells"""
    drugNames = geneDF["Drug"].values
    filtDF = geneDF.drop("Drug", axis=1)
    
    inplaceDF = filtDF.where(filtDF >= 0, 1, inplace=False)

    filteredGenes = filtDF[filtDF.columns[inplaceDF.mean(axis=0) > 0]]
    means = filteredGenes.mean(axis=0).to_numpy()
    std = filteredGenes.std(axis=0).to_numpy()
    cv = np.divide(std, means, out=np.zeros_like(std), where=means != 0)
    filteredGenes["Drug"] = drugNames
    
    return filteredGenes, np.log10(means+1e-10), np.log10(cv+1e-10)


def gene_filter(geneDF, mean, std, offset_value=1.0):
    """Filters genes whos variance are higher than woudl be predicted by a Poisson distribution"""
    slope, intercept, _, _, _ = linregress(mean, std)
    inter = intercept + np.log10(offset_value)

    above_idx = np.where(std > mean * slope + inter)
    finalDF = geneDF.iloc[:, np.append(np.asarray(above_idx).flatten(), geneDF.shape[1] - 1)]
    
    drugNames = geneDF["Drug"].values
    filtDF = geneDF.drop("Drug", axis=1)
    
    maxGenes = pd.DataFrame(data=np.reshape(filtDF.max(axis=0).values, (1, -1)), columns=filtDF.columns.values)
    normMaxG = filtDF.div(maxGenes, axis=1)
    normMaxG = normMaxG.replace(np.nan, 0)
    
    normMaxG["Drug"] = drugNames
    

    return normMaxG, above_idx


def geneNNMF(X, k=14, verbose=0, maxiteration=2000):
    """Turn gene expression into cells X components"""
    model = NMF(n_components=k, verbose=verbose, max_iter=maxiteration, tol=1e-6)
    X = X.drop("Drug", axis=1)
    W = model.fit_transform(X.to_numpy())

    return model.components_, W


def gene_import(offset):
    """Imports gene data from PopAlign and perfroms gene filtering process"""
    genesDF, geneNames = import_thompson_drug()
    genesN = normalizeGenes(genesDF, geneNames)
    filteredGeneDF, logmean, logstd = mu_sigma(genesN)
    finalDF, filtered_index = gene_filter(filteredGeneDF, logmean, logstd, offset_value=offset)

    return finalDF


def ThompsonDrugXA(numCells: int, rank: int, maxit: int):
    """Converts DF to Xarray given number of cells, factor number, and max iter: Factor, CellNumb, Drug, Empty, Empty"""
    finalDF = pd.read_csv("/opt/andrew/FilteredDrugs_Offset1.3.csv")
    finalDF.drop(columns=["Unnamed: 0"], axis=1, inplace=True)
    finalDF = finalDF.groupby(by="Drug").sample(n=numCells).reset_index(drop=True)

    _, geneFactors = geneNNMF(finalDF, k=rank, verbose=0, maxiteration=maxit)
    cmpCol = [f"Fac. {i}" for i in np.arange(1, rank + 1)]

    PopAlignDF = pd.DataFrame(data=geneFactors, columns=cmpCol)
    PopAlignDF["Drug"] = finalDF["Drug"].values
    PopAlignDF["Cell"] = np.tile(np.arange(1, numCells + 1), int(PopAlignDF.shape[0] / numCells))

    PopAlignXA = PopAlignDF.set_index(["Cell", "Drug"]).to_xarray()
    PopAlignXA = PopAlignXA[cmpCol].to_array(dim="Factor")

    npPopAlign = np.reshape(PopAlignXA.to_numpy(), (PopAlignXA.shape[0], PopAlignXA.shape[1], -1, 1, 1))
    PopAlignXA = xa.DataArray(
        npPopAlign,
        dims=("Factor", "Cell", "Drug", "Throwaway 1", "Throwaway 2"),
        coords={"Factor": cmpCol,
            "Cell": np.arange(1, numCells + 1),
            "Drug": finalDF["Drug"].unique(),
            "Throwaway 1": ["Throwaway"],
            "Throwaway 2": ["Throwaway"],},)

    return PopAlignXA

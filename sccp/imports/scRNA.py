import os
from os.path import join
import numpy as np
import pandas as pd
import csv
import xarray as xa
from scipy.io import mmread
from scipy.stats import linregress


path_here = os.path.dirname(os.path.dirname(__file__))


def import_thompson_drug():
    """Imports cell readings from scRNA-seq data from PBMCs from PopAlign paper"
    -Description of each file-
    drugScreeen : str Path to a sparse matrix
    barcodes : str Path to a .tsv 10X barcodes file
    metafile : str Path to a metadata file. Must contains `cell_barcodes` and `sample_id` columns
    genes : str Path to a .tsv 10X gene file"""

    metafile = pd.read_csv("sccp/data/meta.csv")  # Cell barcodes, sample id of treatment and sample number (33482,3)
    drugScreen = mmread("/opt/andrew/drugscreen.mtx").toarray()  # Sparse matrix of each cell/genes (32738,33482)-(Genes,Cell)
    drugScreen = drugScreen.astype(np.float64)
    barcodes = np.array([row[0] for row in csv.reader(open("sccp/data/barcodes.tsv"), delimiter="\t")])  # Cell barcodes(33482)
    genes = np.array([row[1].upper() for row in csv.reader(open("sccp/data/features.tsv"), delimiter="\t")])  # Gene Names (32738)

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
        cellIdx = np.repeat(cellID, len(sample_bcs))  # Connecting drug name with cell
        drugNames = np.append(drugNames, cellIdx)
        totalGenes = pd.concat([totalGenes, pd.DataFrame(data=geneExpression)])  # Setting in a DF

    totalGenes.columns = genes  # Attaching gene name to each column
    totalGenes["Drug"] = drugNames  # Attaching drug name to each cell
    totalGenes = totalGenes.reset_index(drop=True)
    totalGenes = totalGenes.loc[:, ~totalGenes.columns.duplicated()].copy()

    return totalGenes, genes


def mu_sigma_normalize(geneDF, scalingfactor=1000):
    """Calculates the mu and sigma for every gene and returns means, sigmas, and dataframe filtered for genes expressed in > 0.1% of cells"""
    drugNames = geneDF["Drug"].values
    filtDF = geneDF.drop("Drug", axis=1)

    assert np.isnan(filtDF.to_numpy()).all() == False
    assert np.isfinite(filtDF.to_numpy()).all() == True

    inplaceDF = filtDF.where(filtDF <= 0, 1, inplace=False)
    filteredGenes = filtDF[filtDF.columns[inplaceDF.mean(axis=0) > 0.001]]
    sumGenes = filteredGenes.sum(axis=0)

    assert np.isnan(filteredGenes.to_numpy()).all() == False
    assert np.isfinite(filteredGenes.to_numpy()).all() == True
    assert np.isnan(sumGenes).all() == False
    assert np.isfinite(sumGenes).all() == True

    indices_nonzero = []
    for i in range(len(sumGenes.values)):
        if sumGenes.values[i] != 0:
            indices_nonzero = np.append(indices_nonzero, int(i))

    nonZeroGenes = filteredGenes.iloc[:, indices_nonzero]
    genes = nonZeroGenes.columns.values

    sumGenes = sumGenes.iloc[indices_nonzero].to_numpy()
    assert sumGenes.all() != 0

    normG = np.divide(nonZeroGenes.to_numpy(), sumGenes)

    assert np.isnan(normG).all() == False
    assert np.isfinite(normG).all() == True

    logG = np.log10((scalingfactor * (normG)) + 1)

    means = np.mean(logG, axis=0)
    std = np.std(logG, axis=0)

    cv = np.divide(std, means, out=np.zeros_like(std), where=means != 0)

    normDF = pd.DataFrame(data=logG, columns=genes)  # Setting in a DF
    normDF["Drug"] = drugNames  # Attaching drug name to each cell
    normDF = normDF.reset_index(drop=True)

    return normDF, np.log10(means + 1e-10), np.log10(cv + 1e-10)


def gene_filter(geneDF, mean, std, offset_value=1.0):
    """Filters genes whos variance are higher than woudl be predicted by a Poisson distribution"""
    slope, intercept, _, _, _ = linregress(mean, std)
    inter = intercept + np.log10(offset_value)

    above_idx = np.where(std > mean * slope + inter)
    finalDF = geneDF.iloc[:, np.append(np.asarray(above_idx).flatten(), geneDF.shape[1] - 1)]

    return finalDF, above_idx


def gene_import(offset=1.0, filter=False):
    """Imports gene data from PopAlign and perfroms gene filtering process"""
    genesDF, _ = import_thompson_drug()
    filteredGeneDF, logmean, logstd = mu_sigma_normalize(genesDF, scalingfactor=1000)
    if filter == True:
        filteredGeneDF, _ = gene_filter(filteredGeneDF, logmean, logstd, offset_value=offset)
    return filteredGeneDF


def ThompsonXA_SCGenes():
    """Turns filtered and normalized cells into an Xarray."""
    df = pd.read_csv("/opt/andrew/FilteredLogDrugs_Offset_1.1.csv", sep=",")
    df.drop(columns=["Unnamed: 0"], axis=1, inplace=True)
    df = df.sort_values(by=["Drug"])

    # Assign cells a count per-experiment so we can reindex
    cellCount = df.groupby(by=["Drug"]).size().values
    df["Cell"] = np.concatenate([np.arange(int(cnt)) for cnt in cellCount])

    xarr = df.set_index(["Cell", "Drug"]).to_xarray()
    xarr = xarr.to_array(dim="Gene")

    ### I *believe* that padding with zeros does not affect PARAFAC2 results.
    ### We should check this though.
    xarr.values = np.nan_to_num(xarr.values)

    return xarr

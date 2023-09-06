from pandas.plotting import parallel_coordinates as pc
from matplotlib import gridspec, pyplot as plt


def plotMetricSCIB(metricsDF, sheetName, axs):
    """Plots all metrics values across SCIB and Pf2 for one dataset"""
    for i, sheets in enumerate(sheetName):
        datasetDF = metricsDF.loc[metricsDF["Dataset"] == sheets]
        datasetDF = datasetDF.drop(columns="Dataset").reset_index(drop=True)
        datasetDF = datasetDF.pivot_table(index="Metric", columns="Method", values="Value").reset_index()
        pc(datasetDF, "Metric", colormap=plt.get_cmap("Set1"), ax=axs[i])
        axs[i].tick_params(axis="x", rotation=45)
        axs[i].set(title=sheets)
        
def plotMetricNormSCIB(metricsDF, sheetName, axs):
    """Plots overall metric values across SCIB and Pf2 for one dataset"""
    for i, sheets in enumerate(sheetName):
        datasetDF = metricsDF.loc[metricsDF["Dataset"] == sheets]
        datasetDF = datasetDF.drop(columns="Dataset").reset_index(drop=True)
        datasetDF = datasetDF.pivot_table(index="Metric", columns="Method", values="Value").reset_index()
        pc(datasetDF, "Metric", colormap=plt.get_cmap("Set1"), ax=axs[i])
        axs[i].tick_params(axis="x", rotation=45)
        axs[i].set(title=sheets)
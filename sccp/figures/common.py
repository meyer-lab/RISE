"""
This file contains functions that are used in multiple figures.
"""
from string import ascii_letters
import sys
import time
import seaborn as sns
import pandas as pd
import matplotlib
from matplotlib import gridspec, pyplot as plt
import umap.plot
import numpy as np
import scipy.cluster.hierarchy as sch
from ..parafac2 import Pf2X
from ..crossVal import CrossVal
from ..decomposition import R2X
import os
from os.path import join
from pandas.plotting import parallel_coordinates as pc
from parafac2 import parafac2_nd
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.metrics import roc_auc_score, RocCurveDisplay


path_here = os.path.dirname(os.path.dirname(__file__))


matplotlib.use("AGG")

matplotlib.rcParams["legend.labelspacing"] = 0.2
matplotlib.rcParams["legend.fontsize"] = 8
matplotlib.rcParams["xtick.major.pad"] = 1.0
matplotlib.rcParams["ytick.major.pad"] = 1.0
matplotlib.rcParams["xtick.minor.pad"] = 0.9
matplotlib.rcParams["ytick.minor.pad"] = 0.9
matplotlib.rcParams["legend.handletextpad"] = 0.5
matplotlib.rcParams["legend.handlelength"] = 0.5
matplotlib.rcParams["legend.framealpha"] = 0.5
matplotlib.rcParams["legend.markerscale"] = 0.7
matplotlib.rcParams["legend.borderpad"] = 0.35
matplotlib.rcParams["svg.fonttype"] = "none"


def getSetup(figsize, gridd, multz=None, empts=None, constrained_layout=True):
    """Establish figure set-up with subplots."""
    sns.set(
        style="whitegrid",
        font_scale=0.7,
        color_codes=True,
        palette="colorblind",
        rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6},
    )

    # create empty list if empts isn't specified
    if empts is None:
        empts = []

    if multz is None:
        multz = {}

    # Setup plotting space and grid
    f = plt.figure(figsize=figsize, constrained_layout=constrained_layout)
    gs1 = gridspec.GridSpec(*gridd, figure=f)

    # Get list of axis objects
    x = 0
    ax = []
    while x < gridd[0] * gridd[1]:
        if x not in empts and x not in multz.keys():  # If this is just a normal subplot
            ax.append(f.add_subplot(gs1[x]))
        elif x in multz.keys():  # If this is a subplot that spans grid elements
            ax.append(f.add_subplot(gs1[x : x + multz[x] + 1]))
            x += multz[x]
        x += 1

    return (ax, f)


def subplotLabel(axs):
    """Place subplot labels on figure."""
    for ii, ax in enumerate(axs):
        ax.text(
            -0.2,
            1.2,
            ascii_letters[ii],
            transform=ax.transAxes,
            fontweight="bold",
            va="top",
        )


def genFigure():
    """Main figure generation function."""
    fdir = "./output/"
    start = time.time()
    nameOut = "figure" + sys.argv[1]

    exec("from sccp.figures." + nameOut + " import makeFigure", globals())
    ff = makeFigure()
    ff.savefig(fdir + nameOut + ".svg", dpi=300, bbox_inches="tight", pad_inches=0)
    ff.savefig(fdir + nameOut + ".png", dpi=300, bbox_inches="tight", pad_inches=0)

    print(f"Figure {sys.argv[1]} is done after {time.time() - start} seconds.\n")


def plotFactors(factors, data: Pf2X, axs, reorder=tuple(), trim=tuple(), saveGenes=False, cond_group_labels= None):
    """Plots parafac2 factors."""
    pd.set_option('display.max_rows', None)
    rank = factors[0].shape[1]
    xticks = [f"Cmp. {i}" for i in np.arange(1, rank + 1)]
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    for i in range(3):
        # The single cell mode has a square factors matrix
        if i == 0:
            yt = data.condition_labels
            title = "Components by Condition"
        elif i == 1:
            yt = [f"Cell State {i}" for i in np.arange(1, rank + 1)]
            title = "Components by Cell State"
        else:
            yt = data.variable_labels
            title = "Components by Gene"

        X = factors[i]

        if i in trim:
            max_weight = np.max(np.abs(X), axis=1)
            kept_idxs = max_weight > 0.08
            X = X[kept_idxs]
            yt = yt[kept_idxs]
            if i == 0 and not (cond_group_labels is None):
                cond_group_labels = cond_group_labels[ind]

        if i in reorder:
            X, ind = reorder_table(X)
            yt = yt[ind]
            if i == 0 and not (cond_group_labels is None):
                cond_group_labels = cond_group_labels[ind]

        sns.heatmap(
                data=X,
                xticklabels=xticks,
                yticklabels=yt,
                ax=axs[i],
                center=0,
                cmap=cmap,
            )

        if i == 0 and not (cond_group_labels is None):
            # add little boxes to denote SLE/healthy rows
            axs[i].tick_params(axis='y', which='major', pad=20, length=0) # extra padding to leave room for the row colors
            for iii, status in enumerate(cond_group_labels):
                if status == 'SLE':
                    axs[i].add_patch(plt.Rectangle(xy=(-0.05, iii), width=0.05, height=1, color='cyan', lw=0,
                                transform=axs[i].get_yaxis_transform(), clip_on=False))
                elif status == 'Healthy': 
                    axs[i].add_patch(plt.Rectangle(xy=(-0.05, iii), width=0.05, height=1, color='magenta', lw=0,
                                transform=axs[i].get_yaxis_transform(), clip_on=False))

        axs[i].set_title(title)
        axs[i].tick_params(axis="y", rotation=0)
        
        if saveGenes == True:
            if i == 2 and len(yt) > 50:
                df = pd.DataFrame(data=X, index=yt, columns=[f"Cmp. {i}" for i in np.arange(1, rank + 1)])
                df.to_csv(join(path_here, "data/TopBotGenes_Cmp"+str(rank)+".csv"))
                
                geneAmount=50
                genesTop = np.empty((geneAmount, X.shape[1]), dtype="<U10")
                genesBottom = np.empty((geneAmount, X.shape[1]), dtype="<U10")
                sort_idx = np.argsort(X, axis=0)

                for j in range(rank):
                    sortGenes = yt[sort_idx[:, j]]
                    genesTop[:, j] = np.flip(sortGenes[-geneAmount:])  
                    genesBottom[:, j] = sortGenes[:geneAmount]

                dfTop = pd.DataFrame(data=genesTop, columns=[f"Cmp. {i}" for i in np.arange(1, rank + 1)])
                dfBot = pd.DataFrame(data=genesBottom, columns=[f"Cmp. {i}" for i in np.arange(1, rank + 1)])

                dfTop.to_csv(join(path_here, "data/TopGenes_Cmp"+str(rank)+".csv"))
                dfBot.to_csv(join(path_here, "data/BotGenes_Cmp"+str(rank)+".csv"))
   

def reorder_table(projs):
    """Reorder a table's rows using heirarchical clustering"""
    assert projs.ndim == 2
    Z = sch.linkage(projs, method="centroid", optimal_ordering=True)
    index = sch.leaves_list(Z)
    return projs[index, :], index


def flattenData(data):
    """Flattens tensor into dataframe"""
    cellCount = []
    for i in range(len(data.X_list)):
        cellCount = np.append(cellCount, data.X_list[i].shape[0])

    condNames = []

    for i in range(len(data.X_list)):
        condNames = np.append(
            condNames, np.repeat(data.condition_labels[i], cellCount[i])
        )
    flatData = np.concatenate(data.X_list, axis=0)
    dataDF = pd.DataFrame(data=flatData, columns=data.variable_labels)
    dataDF["Condition"] = condNames

    return dataDF

def flattenProjs(data, projs):
    """Flattens tensor into dataframe"""
    cellCount = []
    for i in range(len(data.X_list)):
        cellCount = np.append(cellCount, data.X_list[i].shape[0])

    condNames = []

    for i in range(len(data.X_list)):
        condNames = np.append(
            condNames, np.repeat(data.condition_labels[i], cellCount[i])
        )

    flatProjs= np.concatenate(projs, axis=0)
    cmpNames = [f"Cmp. {i}" for i in np.arange(1, flatProjs.shape[1] + 1)]
    dataDF = pd.DataFrame(data=flatProjs, columns=cmpNames)
    dataDF["Condition"] = condNames

    return dataDF


def plotGeneUMAP(genes, decomp, points, dataDF, axs):
    """Scatterplot of UMAP visualization weighted by gene"""
    subset = np.random.choice(a=[False, True], size=np.shape(dataDF)[0], p=[.95, .05])
    cmap = sns.color_palette("ch:s=-.2,r=.6", as_cmap=True)
    for i, genez in enumerate(genes):
        geneList = dataDF[genez].to_numpy()
        geneList = geneList / np.max(np.abs(geneList))
        psm = plt.pcolormesh([[0, 1], [0, 1]], cmap=cmap)
        plot = umap.plot.points(points, values=geneList, cmap=cmap, subset_points= subset, ax=axs[i])
        colorbar= plt.colorbar(psm, ax=plot)
        axs[i].set(
            title=genez + "-" + decomp + "-Based Decomposition",
            ylabel="UMAP2",
            xlabel="UMAP1")

    return


def plotDrugUMAP(drugs, decomp, totaldrugs, points, axs):
    """Scatterplot of UMAP visualization weighted by condition"""
    subset = np.random.choice(a=[False, True], size=len(totaldrugs), p=[.95, .05])
    for i, drugz in enumerate(drugs):
        drugList = np.where(np.asarray(totaldrugs == drugz), drugz, "Other Drugs")
        umap.plot.points(
            points, labels=drugList, ax=axs[i], color_key_cmap="tab20", show_legend=True, subset_points=subset)
        axs[i].set(
            title=decomp + "-Based Decomposition",
        ylabel="UMAP2",
        xlabel="UMAP1")

    return


def plotCmpUMAP(cellState, cmp, factors, pf2Points, allP, ax):
    """Scatterplot of UMAP visualization weighted by
    projections for a component and cell state"""
    weightedProjs = allP[:, cellState-1] * factors[1][cellState-1, cmp-1]
    subset = np.random.choice(a=[False, True], size= len(weightedProjs), p=[.95, .05])
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    weightedProjs = weightedProjs / np.max(np.abs(weightedProjs))
    psm = plt.pcolormesh([[-1, 1],[-1, 1]], cmap=cmap)
    plot = umap.plot.points(pf2Points, values=weightedProjs, cmap=cmap, subset_points= subset, ax=ax)
    colorbar= plt.colorbar(psm, ax=plot)
    ax.set(
        ylabel="UMAP2",
        xlabel="UMAP1",
        title="Cell State:" + str(cellState)+"- Component:" + str(cmp))


def plotBatchUMAP(decomp_DF, ax):
    """Scatterplot of UMAP visualization weighted by condition"""
    sns.scatterplot(data=decomp_DF, x="UMAP 1", y="UMAP 2", hue="Batch", s=1, palette="muted", ax=ax)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels)

    return


def plotCellUMAP(decomp_DF, ax):
    """Scatterplot of UMAP visualization weighted by condition"""
    sns.scatterplot(data=decomp_DF, x="UMAP 1", y="UMAP 2", hue="Cell Type", s=1, palette="muted", legend=False, ax=ax)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels)

    return


def plotR2X(data, rank, ax):
    """Creates R2X plot for parafac2 tensor decomposition"""
    r2xError = R2X(data, rank)

    rank_vec = np.arange(1, rank + 1)
    labelNames = ["Fit: Pf2", "Fit: PCA"]
    colorDecomp = ["r", "b"]
    markerShape = ["|", "_"]

    for i in range(2):
        ax.scatter(
            rank_vec,
            r2xError[i],
            label=labelNames[i],
            marker=markerShape[i],
            c=colorDecomp[i],
            s=30.0,
        )

    ax.set(
        ylabel="Variance Explained",
        xlabel="Number of Components",
        xticks=np.linspace(0, rank, num=8, dtype=int),
        yticks=np.linspace(
            0, np.max(np.append(r2xError[0], r2xError[1])) + 0.01, num=5
        ),
    )

    ax.legend()


def plotCV(data, rank, trainPerc, ax):
    """Creates variance explained plot for parafac2 tensor decomposition CV"""
    cvError = CrossVal(data, rank, trainPerc=trainPerc)

    rank_vec = np.arange(1, rank + 1)
    labelNames = ["CV: Pf2", "CV: PCA"]
    colorDecomp = ["r", "b"]
    markerShape = ["o", "o"]

    for i in range(2):
        ax.scatter(
            rank_vec,
            cvError[i],
            label=labelNames[i],
            marker=markerShape[i],
            c=colorDecomp[i],
            s=30.0,
        )

    ax.set(
        ylabel="Variance Explained",
        xlabel="Number of Components",
        xticks=np.linspace(0, rank, num=8, dtype=int),
        yticks=np.linspace(0, np.max(np.append(cvError[0], cvError[1])) + 0.01, num=5),
    )

    ax.legend()

def plotDistDrug(df, conds, ax):
    """Plots normalized centroid distance across PCA and Pf2 for different ranks"""
    for i, cond in enumerate(conds):
        plotDF = df.loc[df["Condition"] == cond]
        sns.lineplot(data=plotDF, x="Rank", y="Normalized Centroid Distance", hue="Method", ax=ax[i])
        ax[i].set(title=cond)
        
def plotDistGene(df, genes, ax):
    """Plots normalized centroid distance across PCA and Pf2 for different ranks"""
    for i, gene in enumerate(genes):
        plotDF = df.loc[df["Marker"] == gene]
        sns.lineplot(data=plotDF, x="Rank", y="Normalized Centroid Distance", hue="Method", ax=ax[i])
        ax[i].set(title=gene)
        
def plotDistAllDrug(df, rank, ax):
    """Plots all Normalized Centroid Distance for all drugs for Pf2 and PCA"""
    sns.swarmplot(data=df, x="Method", y="Normalized Centroid Distance", hue="Method", ax=ax)
    ax.set(title="All Conditions: Rank = " + str(rank))
    
def plotDistAllGene(df, rank, ax):
    """Plots all Normalized Centroid Distance for all genes for Pf2 and PCA"""
    sns.swarmplot(data=df, x="Method", y="Normalized Centroid Distance", hue="Method", ax=ax)
    ax.set(title="All Canonical Genes: Rank = " + str(rank))

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


def plotLabelAllUMAP(conditions, points, ax):
    """Scatterplot of UMAP visualization weighted by condition or cell type"""
    subset = np.random.choice(a=[False, True], size=len(conditions), p=[.93, .07])
    umap.plot.points(
        points, labels=conditions, ax=ax, color_key_cmap="tab20", show_legend=True, subset_points=subset)
    ax.set(
        title="Pf2-Based Decomposition",
        ylabel="UMAP2",
        xlabel="UMAP1")


def plotCellType(dataDF, celltypes, ax):
    """Plots a swarmplot for cell type distribution for each condition """
    dataDF["Cell Type"] = celltypes
    celltypeDF = dataDF.groupby(["Cell Type", "Condition"]).size().reset_index(name="Count") 

    for j, cond in enumerate(np.unique(dataDF["Condition"].values)):
        df = celltypeDF.loc[celltypeDF["Condition"] == cond] 
        perc = df["Count"].values / np.sum(df["Count"].values)
        celltypeDF.loc[celltypeDF["Condition"] == cond, "Count"] = perc
            
    sns.swarmplot(data=celltypeDF, x="Condition", y="Count", hue="Cell Type", ax=ax)
    
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
    
def plotCellCount(dataDF, ax):
    """Plot number of cells per experiment for a dataframe"""
    cellcountDF = dataDF.groupby(["Condition"]).size().reset_index(name="Cell Count") 
    sns.barplot(data=cellcountDF, x="Condition", y="Cell Count", ax=ax)
    ax.tick_params(axis="x", rotation=90)

def plotWeight(weight, ax):
    """Plots weights from Pf2 model"""
    df = pd.DataFrame(data=np.transpose([weight]), columns=["Value"])
    df["Value"] = df["Value"]/np.max(df["Value"])
    df["Component"] = [f"Cmp. {i}" for i in np.arange(1, len(weight) + 1)]
    sns.barplot(data=df, x="Component", y="Value", ax=ax)
    ax.tick_params(axis="x", rotation=90)


def plotUMAP_ct(labels, pf2Points, ax):
    """Scatterplot of UMAP visualization labeled by cell type"""
    plot = umap.plot.points(pf2Points, 
                            labels = labels, 
                            theme='viridis', 
                            ax=ax)
    ax.set(
        ylabel="UMAP2",
        xlabel="UMAP1",
        title="Pf2-Based Decomposition: Label Cell Types")
    
def plotCompViolins(projection_B, cell_types, component: int, ax):
    all_cell_projs = pd.DataFrame(projection_B)
    comp_n = pd.concat([all_cell_projs.iloc[:, (component - 1)], cell_types], axis = 1)
    comp_n.columns.values[0] = "contribution"

    sns.violinplot(data = comp_n,
                   x = "cg_cov",
                   y = 'contribution',
                   hue = 'cg_cov',
                   dodge = False,
                   ax = ax)
    
    ax.set_title('Cell Type Contrib to Component ' + str(component))
    ax.tick_params(axis="x", rotation=90)
    ax.get_legend().remove()


def savePf2(weight, factors, projs, dataName: str):
    """Saves weight factors and projections for one dataset for a component"""
    rank = len(weight)
    np.save(join(path_here, "data/"+dataName+"/"+dataName+"_WeightCmp"+str(rank)+".npy"), weight)
    for i in range(3):
        np.save(join(path_here, "data/"+dataName+"/"+dataName+"_Factor"+str(i)+"Cmp"+str(rank)+ ".npy"), factors[i])
    np.save(join(path_here, "data/"+dataName+"/"+dataName+"_ProjCmp"+str(rank)+".npy"), np.concatenate(projs, axis=0))

    
def openPf2(rank: int, dataName: str, optProjs = False):
    """Opens weight factors and projections for one dataset for a component as numpy arrays"""
    weight = np.load(join(path_here, "data/"+dataName+"/"+dataName+"_WeightCmp"+str(rank)+".npy"), allow_pickle=True)
    factors = [np.load(join(path_here, "data/"+dataName+"/"+dataName+"_Factor0Cmp"+str(rank)+ ".npy"), allow_pickle=True),
               np.load(join(path_here, "data/"+dataName+"/"+dataName+"_Factor1Cmp"+str(rank)+ ".npy"), allow_pickle=True),
               np.load(join(path_here, "data/"+dataName+"/"+dataName+"_Factor2Cmp"+str(rank)+ ".npy"), allow_pickle=True)]
        
    if optProjs is False:
        projs = np.load(join(path_here, "data/"+dataName+"/"+dataName+"_ProjCmp"+str(rank)+".npy"), allow_pickle=True)
    else:
        projs = np.load(join(path_here, "/opt/andrew/"+dataName+"/"+dataName+"_ProjCmp"+str(rank)+".npy"), allow_pickle=True)
        
    return weight, factors, projs

def testPf2Ranks(pfx2_data, condition_labels, ranks_to_test,
                 penalty_type = 'l1', solver = 'saga', error_metric = 'accuracy',
                 penalties_to_test = 10):
    
    results = []
    for rank in ranks_to_test:

        # perform pf2 on the given rank
        print('########################################################################\n',
              '########################################################################',
              '\n\nPARAFAC2 FITTING: RANK ', str(rank))
        _, factors, _, _ = parafac2_nd(pfx2_data, 
                                rank = rank, 
                                random_state = 1, 
                                verbose=True)
        
        A_matrix = factors[0]
        
        # train a logisitic regression model on that rank, using cross validation

        log_reg = LogisticRegressionCV(random_state=0, 
                                       max_iter = 5000, 
                                       penalty = penalty_type, 
                                       solver = solver,
                                       Cs = penalties_to_test,
                                       scoring = error_metric)
        
        log_fit = log_reg.fit(A_matrix, condition_labels.to_numpy())

        acc_scores = pd.DataFrame(pd.DataFrame(log_fit.scores_.get('SLE')).mean()).rename(columns = {0: error_metric})
        c_vals = pd.DataFrame(log_fit.Cs_).rename(columns = {0: "penalty"})

        acc_w_c = acc_scores.merge(c_vals, left_index = True, right_index = True)

        # grab fit results as a pandas dataframe, indicate which rank these are from
        initial_results = pd.DataFrame(acc_w_c)
        initial_results['rank'] = rank

        # save best results into results list
        results.append(initial_results)

    # concatenate all the results into one frame for viewing:

    return pd.concat(results, ignore_index = True)

def plotPf2RankTest(rank_test_results, ax, error_metric = "accuracy", palette = 'Set2'):
    sns.lineplot(data = rank_test_results, 
                 x = 'rank', y = error_metric, 
                 hue = 'penalty',
                 palette= 'Set2',
                 ax = ax)
    sns.scatterplot(data = rank_test_results,
                    x = 'rank', y = error_metric,
                    hue = 'penalty',
                    palette= palette,
                    legend=False,
                    ax = ax)
    ax.set_title(error_metric + ' by Hyperparameter input')

def plotCmpRegContributions(A_matrix, target, penalty_amt, rank, ax):
    log_reg = LogisticRegression(random_state=0, max_iter = 5000, penalty = 'l1', solver = 'saga', C = penalty_amt)

    log_fit = log_reg.fit(A_matrix, target)

    coefs = pd.DataFrame(log_fit.densify().coef_,
                         columns = [f"comp_{i}" for i in np.arange(1, rank + 1)]).melt(var_name = "Component",
                                                                                       value_name = "Weight")
    
    sns.barplot(data = coefs, x = "Component", y = "Weight", color = '#1a759f', ax = ax)
    ax.tick_params(axis="x", rotation=90)
    ax.set_title('Weight of Each component in Logsitic Regression')

def plotPf2ROC(A_matrix, conditions, condition_batch_labels, rank, ax, penalties_to_test = 10):
    
    A_matrix = pd.DataFrame(A_matrix, 
                            index = conditions,
                            columns = [f"comp_{i}" for i in np.arange(1, rank + 1)])
    comps_w_sle_status = A_matrix.merge(condition_batch_labels, left_index=True, right_index=True)
    cohort_4 = comps_w_sle_status[comps_w_sle_status["Processing_Cohort"] == str(4.0)]
    cohorts_123 = comps_w_sle_status[comps_w_sle_status["Processing_Cohort"] != str(4.0)]
    last_comp = "comp_" + str(rank)
    cmp_train = cohort_4.loc[:, "comp_1":last_comp].to_numpy()
    y_train = cohort_4.loc[:, "SLE_status"].to_numpy()
    cmp_test = cohorts_123.loc[:, "comp_1":last_comp].to_numpy()
    y_test = cohorts_123.loc[:, "SLE_status"].to_numpy()
    # train a logisitic regression model using cross validation
    log_reg = LogisticRegressionCV(random_state=0, max_iter = 10000, penalty = 'l1', solver = 'saga',
                                   scoring = "roc_auc",
                                    Cs = penalties_to_test)
    log_fit = log_reg.fit(cmp_train, y_train)

    # get decision function for ROC AUC
    sle_decisions = log_fit.decision_function(cmp_test)
    # validate the ROC AUC of the model
    roc_auc = roc_auc_score(y_test, sle_decisions)
    print("The best ROC AUC is: ", roc_auc)
    RocCurveDisplay.from_predictions(y_test, sle_decisions, 
                                     pos_label = "SLE",
                                     plot_chance_level = True,
                                     ax = ax)
    
    ax.set_title('OOS ROC for Cases/Controls: ' + str(rank) + ' Comp LASSO')

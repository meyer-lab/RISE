"""
S3b: Logistic Regression (and maybe SVM) on Pf2 Factor matrix A output
article: https://www.science.org/doi/10.1126/science.abf1970
data: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE174188
"""

# GOAL: run logisitc regression to see which components are best able to predict disease status

# load functions/modules ----
from .common import subplotLabel, getSetup, openPf2
from .commonFuncs.plotLupus import plotROCAcrossGroups
from ..imports.scRNA import load_lupus_data
import warnings
import numpy as np
import seaborn as sns
import pandas as pd
import itertools
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn import preprocessing

warnings.filterwarnings("ignore")

def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((6, 6), (1, 1))  # fig size  # grid size

    # Add subplot labels
    subplotLabel(ax)

    rank = 40

    _, factors, _ = openPf2(rank=rank, dataName="lupus", optProjs=True)

    _, obs = load_lupus_data()

    status = obs[["sample_ID", "SLE_status", "Processing_Cohort"]].drop_duplicates()

    group_labs = status.set_index("sample_ID")[["SLE_status", "Processing_Cohort"]]

    A_matrix = factors[0]

    Lupus_comp_scan_plot(ax[0], A_matrix, status)

    return f


def Lupus_comp_scan_plot(ax, patient_facs, status_DF):
    """Plot factor weights for donor BC prediction"""
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=20)
    lrmodel = LogisticRegressionCV(penalty="l1", solver="saga", max_iter=20000, tol=1e-6, cv=cv)
    mode_facs = patient_facs

    Lupus_y = preprocessing.label_binarize(status_DF.SLE_status, classes=["Healthy", "SLE"]).flatten()
    all_comps = np.arange(0, mode_facs.shape[1])
    Acc_DF = pd.DataFrame()
    all_comps = np.arange(0, 2)

    for comps in itertools.product(all_comps, all_comps):
        print(comps)
        if comps[0] > comps[1]:
            acc = Acc_DF.loc[(Acc_DF["Component 1"] == "Comp. " + str(comps[1] + 1)) & (Acc_DF["Component 2"]== "Comp. " + str(comps[0] + 1))].Accuracy.values[0]
            Acc_DF = pd.concat([Acc_DF,pd.DataFrame({"Component 1": "Comp. " + str(comps[0] + 1),"Component 2": "Comp. " + str(comps[1] + 1), "Accuracy": [acc]})])
        else:
            if comps[0] == comps[1]:
                compFacs = mode_facs[:, comps[0]][:, np.newaxis]
            else:
                compFacs = mode_facs[:, [comps[0], comps[1]]]

            LR_CoH = lrmodel.fit(compFacs, Lupus_y)
            acc = LR_CoH.score(compFacs, Lupus_y)
            Acc_DF = pd.concat(
                [
                    Acc_DF,
                    pd.DataFrame(
                        {
                            "Component 1": "Comp. " + str(comps[0] + 1),
                            "Component 2": "Comp. " + str(comps[1] + 1),
                            "Accuracy": [acc],
                        }
                    ),
                ]
            )
            print(Acc_DF)

    Acc_DF = Acc_DF.pivot_table(
        index="Component 1", columns="Component 2", values="Accuracy", sort=False
    )
    sns.heatmap(
        data=Acc_DF,
        vmin=0.5,
        vmax=1,
        xticklabels=True,
        yticklabels=True,
        cbar_kws={"label": "Accuracy 10-fold CV"},
        ax=ax,
    )
    ax.set(xlabel="First Component", ylabel="Second Component")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
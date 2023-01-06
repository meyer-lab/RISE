import numpy as np
import seaborn as sns
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from os.path import join
from .tensor import optimal_seed
from .CoHimport import patients

path_here = os.path.dirname(os.path.dirname(__file__))


def cp_normalize(tFac):
    """Normalize the factors using the inf norm."""
    for i, factor in enumerate(tFac.factors):
        scales = np.linalg.norm(factor, ord=np.inf, axis=0)
        tFac.weights *= scales
        if i == 0 and hasattr(tFac, "mFactor"):
            mScales = np.linalg.norm(tFac.mFactor, ord=np.inf, axis=0)
            tFac.mWeights = scales * mScales
            tFac.mFactor /= mScales

        tFac.factors[i] /= scales

    return tFac


def CoH_LogReg_plot(ax, tFac, CoH_Array, numComps):
    """Plot factor weights for donor BC prediction"""
    coord = CoH_Array.dims.index("Patient")
    mode_facs = tFac[1][coord]
    status_DF = pd.read_csv(
        join(path_here, "sccp/data/CoH_Patient_Status.csv"), index_col=0
    )
    Donor_CoH_y = preprocessing.label_binarize(
        status_DF["Status"], classes=["Healthy", "BC"]
    ).flatten()
    LR_CoH = LogisticRegression(random_state=0).fit(mode_facs, Donor_CoH_y)
    CoH_comp_weights = pd.DataFrame(
        {"Component": np.arange(1, numComps + 1), "Coefficient": LR_CoH.coef_[0]}
    )
    sns.barplot(data=CoH_comp_weights, x="Component", y="Coefficient", color="k", ax=ax)
    print("Accuracy:", LR_CoH.score(mode_facs, Donor_CoH_y))


def BC_status_plot(cohXA, rank, n_cluster, seed, ax):
    """Plot 5 fold CV by # components"""
    accDF = pd.DataFrame([])
    status_DF = pd.read_csv(
        join(path_here, "sccp/data/CoH_Patient_Status.csv"), index_col=0
    )
    Donor_CoH_y = preprocessing.label_binarize(
        status_DF["Status"], classes=["Healthy", "BC"]
    ).flatten()
    crossval = StratifiedKFold(n_splits=10)
    for i in range(2, rank + 1):
        print("Rank:", i)
        _, _, fit = optimal_seed(seed, cohXA, rank=i, n_cluster=n_cluster)
        fac = fit[0]
        cp_normalize(fac)
        mode_labels = cohXA["Patient"]
        coord = cohXA.dims.index("Patient")
        mode_facs = fac[1][coord]
        tFacDF = pd.DataFrame()
        mode_facs = np.squeeze(mode_facs)

        for j in range(0, i):
            tFacDF = pd.concat(
                [
                    tFacDF,
                    pd.DataFrame(
                        {
                            "Component_Val": mode_facs[:, j],
                            "Component": (j + 1),
                            "Patient": mode_labels,
                        }
                    ),
                ]
            )

        tFacDF = pd.pivot(
            tFacDF, index="Component", columns="Patient", values="Component_Val"
        )
        tFacDF = tFacDF[patients]
        TFAC_X = tFacDF.transpose().values
        LR = LogisticRegression()
        scoresTFAC = cross_val_score(LR, TFAC_X, Donor_CoH_y, cv=crossval)
        accDF = pd.concat(
            [
                accDF,
                pd.DataFrame(
                    {"Components": [i], "Accuracy (10-fold CV)": np.mean(scoresTFAC)}
                ),
            ]
        )
        print("Ave. Accuracy:", np.mean(scoresTFAC))
    accDF = accDF.reset_index(drop=True)
    sns.lineplot(data=accDF, x="Components", y="Accuracy (10-fold CV)", ax=ax)
    ax.set(xticks=np.arange(2, rank + 1))

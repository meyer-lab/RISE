# LOGISTIC REGRESSION HELPER FUNCTIONS
# for more information about possible inputs/specifications, see the sci-kit learn documentation:
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from .factorization import pf2


def testPf2Ranks(
    pfx2_data,
    condition_labels_all,
    ranks_to_test: list[int],
    error_metric="accuracy",
    cv_group=None,
):
    """Tests various numbers of components for Pf2 by optimizing some error metric in logisitic regression (predicting SLE status)
    pfx2_data: data in Pf2X format
    condition_labels_all: condition labels for both the thing you are predicting (like SLE Status) and your grouping variable (if applicable)
    ranks_to_test: Pf2 ranks to try
    error_metric: error metric to pass to the logistic regression `scoring` parameter (https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics)
    cv_group: (str) name of column in `condition_labels_all` that should be grouped by in cross validation
    """

    results = []
    for rank in ranks_to_test:
        # perform pf2 on the given rank
        print(f"\n\nPARAFAC2 FITTING: RANK {rank}")

        X = pf2(pfx2_data, rank=rank, doEmbedding=False)

        A_matrix = X.uns["Pf2_A"]
        condition_labels = condition_labels_all["SLE_status"]

        # train a logisitic regression model on that rank, using cross validation
        # if we want cross validation groups made across a certain feature (like batch or patient); make them
        cvs = None

        if cv_group is not None:
            sgkf = StratifiedGroupKFold(n_splits=4)
            # get labels for the group that you want to do cross validation by
            group_cond_labels = condition_labels_all[cv_group]
            cvs = sgkf.split(
                A_matrix, condition_labels.to_numpy(), group_cond_labels.to_numpy()
            )

        log_reg = LogisticRegressionCV(
            random_state=0,
            max_iter=5000,
            cv=cvs,  # type: ignore
            penalty="l1",
            solver="saga",
            scoring=error_metric,
            n_jobs=5,
        )

        log_fit = log_reg.fit(A_matrix, condition_labels.to_numpy())

        # grab fit results as a pandas dataframe, indicate which rank these are from
        initial_results = pd.DataFrame(
            {"penalty": log_fit.Cs_, error_metric: log_fit.scores_["SLE"].mean(axis=0)}
        )
        initial_results["rank"] = rank

        # save best results into results list
        results.append(initial_results)

    # concatenate all the results into one frame for viewing:
    return pd.concat(results, ignore_index=True)


def getPf2ROC(
    A_matrix: np.ndarray, condition_batch_labels: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Train a logistic regression model using CV on some cohorts, test on another
    A_matrix: first factor matrix (Pf2 output)
    condition_batch_labels: unique list of observation categories, indexed by sample ID
    """
    cohort_four = (condition_batch_labels["Processing_Cohort"] == "4.0").to_numpy(
        dtype=bool
    )
    y = (condition_batch_labels["SLE_status"] == "SLE").to_numpy(dtype=bool)

    # train + fit a logisitic regression model using cross validation ON ONLY THE TRAINING (GROUP 4) DATA
    log_reg = LogisticRegressionCV(
        random_state=0,
        max_iter=10000,
        penalty="l1",
        solver="saga",
        scoring="roc_auc",
    )
    log_fit = log_reg.fit(A_matrix[cohort_four], y[cohort_four])

    # get decision function for ROC AUC
    sle_decisions = log_fit.decision_function(A_matrix[~cohort_four])
    y_test = y[~cohort_four]

    # validate the ROC AUC of the model
    roc_auc = roc_auc_score(y_test, sle_decisions)
    print("The best ROC AUC is: ", roc_auc)

    return y_test, sle_decisions

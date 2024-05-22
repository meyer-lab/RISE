# LOGISTIC REGRESSION HELPER FUNCTIONS
# for more information about possible inputs/specifications, see the sci-kit learn documentation:
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from .factorization import pf2, correct_conditions


def predaccuracy_ranks_lupus(
    pfx2_data,
    condition_labels_all,
    ranks_to_test,
    error_metric="roc_auc",
):
    """Tests various numbers of components for Pf2 by optimizing some error metric in logisitic regression (predicting SLE status)
    pfx2_data: data in Pf2X format
    condition_labels_all: condition labels for both the thing you are predicting (like SLE Status) and your grouping variable (if applicable)
    ranks_to_test: Pf2 ranks to try
    error_metric: error metric to pass to the logistic regression `scoring` parameter (https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics)
    cv_group: (str) name of column in `condition_labels_all` that should be grouped by in cross validation
    """

    results = []
    pfx2_data = pfx2_data.to_memory()
    for rank in ranks_to_test:
        print(f"\n\n Component:{rank}")

        pf2_output = pf2(pfx2_data, rank=int(rank), doEmbedding=False)

        pf2_output.uns["Pf2_A"] = correct_conditions(pf2_output)

        A_matrix = pf2_output.uns["Pf2_A"]

        cohort_four = (condition_labels_all["Processing_Cohort"] == "4.0").to_numpy(
            dtype=bool
        )
        y = (condition_labels_all["SLE_status"] == "SLE").to_numpy(dtype=bool)

        log_reg = logistic_regression(scoring=error_metric)
        log_fit = log_reg.fit(A_matrix[cohort_four], y[cohort_four])

        if error_metric == "roc_auc":
            sle_decisions = log_fit.decision_function(A_matrix[~cohort_four])
            y_true = y[~cohort_four]
            score = roc_auc_score(y_true, sle_decisions)
            initial_results = pd.DataFrame({error_metric: [score]})

        if error_metric == "accuracy":
            score = log_fit.score(A_matrix[~cohort_four], y[~cohort_four])
            initial_results = pd.DataFrame({error_metric: [score]})

        initial_results["Component"] = rank

        results.append(initial_results)

    return pd.concat(results, ignore_index=True)


def roc_lupus_fourtbatch(
    X,
    condition_batch_labels: pd.DataFrame,
    error_metric="roc_auc",
) -> tuple[np.ndarray, np.ndarray]:
    """Train a logistic regression model using CV on some cohorts, test on another
    A_matrix: first factor matrix (Pf2 output)
    condition_batch_labels: unique list of observation categories, indexed by sample ID
    """

    cond_factors = np.array(X.uns["Pf2_A"])
    X.uns["Pf2_A"] = correct_conditions(X)

    cohort_four = (condition_batch_labels["Processing_Cohort"] == "4.0").to_numpy(
        dtype=bool
    )
    y = (condition_batch_labels["SLE_status"] == "SLE").to_numpy(dtype=bool)

    log_reg = logistic_regression(scoring=error_metric)
    log_fit = log_reg.fit(cond_factors[cohort_four], y[cohort_four])

    sle_decisions = log_fit.decision_function(cond_factors[~cohort_four])
    y_test = y[~cohort_four]

    roc_auc = roc_auc_score(y_test, sle_decisions)
    print("ROC AUC: ", roc_auc)

    return y_test, sle_decisions


def logistic_regression(scoring):
    """Standardizing LogReg for all functions"""
    lrcv = LogisticRegressionCV(
        random_state=0,
        max_iter=10000,
        penalty="l1",
        solver="saga",
        scoring=scoring,
    )

    return lrcv

import numpy as np
import pandas as pd
import anndata
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from .factorization import pf2, correct_conditions


def predaccuracy_ranks_lupus(
    pfx2_data: anndata.AnnData,
    condition_labels_all: pd.DataFrame,
    ranks_to_test: np.ndarray,
    error_metric: str = "roc_auc",
):
    """Tests various numbers of components for Pf2 by optimizing metric for predicting SLE status
    pfx2_data: annData file
    condition_labels_all: Labels for patient samples
    ranks_to_test: Pf2 ranks
    error_metric: Metric used for LR
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
    X: anndata.AnnData,
    condition_batch_labels: pd.DataFrame,
    error_metric: str = "roc_auc",
) -> tuple[np.ndarray, np.ndarray]:
    """Train a logistic regression model using CV on a cohort and testing on others
    X: annData file
    condition_batch_labels: Labels for patient samples
    ranks_to_test: Pf2 ranks
    error_metric: Metric used for LR
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

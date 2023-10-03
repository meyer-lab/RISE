# LOGISTIC REGRESSION HELPER FUNCTIONS
# for more information about possible inputs/specifications, see the sci-kit learn documentation:
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html

import pandas as pd
import numpy as np
from parafac2 import parafac2_nd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score


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
        _, factors, _, _ = parafac2_nd(
            pfx2_data, rank=rank, random_state=1
        )

        A_matrix = factors[0]
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


def getCompContribs(A_matrix, target, penalty_amt=50):
    """Fit logistic regression model, return coefficients of that model"""
    rank = A_matrix.shape[1]
    log_reg = LogisticRegression(
        random_state=0, max_iter=5000, penalty="l1", solver="saga", C=penalty_amt
    )

    log_fit = log_reg.fit(A_matrix, target)

    coefs = pd.DataFrame(
        log_fit.densify().coef_, columns=[f"Cmp. {i}" for i in np.arange(1, rank + 1)]
    ).melt(var_name="Component", value_name="Weight")
    return coefs


def getPf2ROC(A_matrix, condition_batch_labels, rank, penalties_to_test=10):
    """Train a logistic regression model using CV on some cohorts, test on another
    A_matrix: first factor matrix (Pf2 output)
    condition_batch_labels: unique list of observation categories, indexed by sample ID
    rank: rank of Pf2 model being used
    penalties_to_test: Penalties to be passed to `Cs` parameter of sklearn.linear_model.LogisticRegressionCV
    """
    # get list of conditions, patients
    conditions = condition_batch_labels.index
    patients = condition_batch_labels["patient"].tolist()
    # make a combined dataframe from which to draw samples after splitting into train/test data
    A_matrix = pd.DataFrame(
        A_matrix,
        index=conditions,
        columns=[f"comp_{i}" for i in np.arange(1, rank + 1)],
    )
    comps_w_sle_status = A_matrix.merge(
        condition_batch_labels, left_index=True, right_index=True
    )

    # need two lists of patients: ones with batch 4 samples, and those that never had samples processed in batch 4
    patients_in_batch_4 = condition_batch_labels[
        condition_batch_labels["Processing_Cohort"] == str(4.0)
    ]["patient"].tolist()
    other_patients = []
    for pat in set(patients):
        if pat not in patients_in_batch_4:
            other_patients.append(pat)

    # using set to make sure we don't count duplicates in that list
    assert len(set(patients)) == len(patients_in_batch_4) + len(other_patients)
    # now we have two lists of patients, one which are patients with samples in cohort 4, one which is everyone else

    group_4 = comps_w_sle_status[
        comps_w_sle_status["patient"].isin(patients_in_batch_4)
    ]
    group_4 = group_4[group_4["Processing_Cohort"] == str(4.0)]
    assert group_4["Processing_Cohort"].nunique() == 1
    group_123 = comps_w_sle_status[comps_w_sle_status["patient"].isin(other_patients)]
    assert group_123["Processing_Cohort"].nunique() == 3
    # NOW we have the two groups that we need-- one with only cohort 4 (n=96) and one with other samples from non-4 patients (n=228)
    # need to separate these out into predictor sets (components) and target variable (SLE Status)
    # also can rename the 4 group as training and the 123 group as testing

    last_comp = "comp_" + str(rank)
    cmp_train = group_4.loc[:, "comp_1":last_comp].to_numpy()
    y_train = group_4.loc[:, "SLE_status"].to_numpy()
    cmp_test = group_123.loc[:, "comp_1":last_comp].to_numpy()
    y_test = group_123.loc[:, "SLE_status"].to_numpy()

    # train + fit a logisitic regression model using cross validation ON ONLY THE TRAINING (GROUP 4) DATA
    log_reg = LogisticRegressionCV(
        random_state=0,
        max_iter=10000,
        penalty="l1",
        solver="saga",
        scoring="roc_auc",
        #Cs=penalties_to_test,
    )
    log_fit = log_reg.fit(cmp_train, y_train)

    # get decision function for ROC AUC
    sle_decisions = log_fit.decision_function(cmp_test)
    # validate the ROC AUC of the model
    roc_auc = roc_auc_score(y_test, sle_decisions)
    print("The best ROC AUC is: ", roc_auc)

    return y_test, sle_decisions

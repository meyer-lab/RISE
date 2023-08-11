import pandas as pd
import numpy as np
from parafac2 import parafac2_nd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score


def getCompContribs(A_matrix, target, penalty_amt = 50):
    """Fit logistic regression model, return coefficients of that model"""
    rank = A_matrix.shape[1]
    log_reg = LogisticRegression(random_state=0, max_iter = 5000, penalty = 'l1', solver = 'saga', C = penalty_amt)

    log_fit = log_reg.fit(A_matrix, target)

    coefs = pd.DataFrame(log_fit.densify().coef_,
                         columns = [f"comp_{i}" for i in np.arange(1, rank + 1)]).melt(var_name = "Component",
                                                                                       value_name = "Weight")
    return coefs

def getPf2ROC(A_matrix, condition_batch_labels, rank, penalties_to_test = 10):
    """Train a logistic regression model using CV on some cohorts, test on another"""
    # get list of conditions, patients
    conditions = condition_batch_labels.index
    patients = condition_batch_labels['patient'].tolist()
    # make a combined dataframe from which to draw samples after splitting into train/test data
    A_matrix = pd.DataFrame(A_matrix, 
                            index = conditions,
                            columns = [f"comp_{i}" for i in np.arange(1, rank + 1)])
    comps_w_sle_status = A_matrix.merge(condition_batch_labels, left_index=True, right_index=True)

    # need two lists of patients: one that have 
    patients_in_batch_4 = condition_batch_labels[condition_batch_labels['Processing_Cohort'] == str(4.0)]['patient'].tolist()
    other_patients = []
    for pat in set(patients):
        if pat not in patients_in_batch_4:
            other_patients.append(pat)

    # using set to make sure we don't count duplicates in that list
    assert len(set(patients)) == len(patients_in_batch_4) + len(other_patients)
    # now we have two lists of patients, one which are patients with samples in cohort 4, one which is everyone else

    group_4 = comps_w_sle_status[comps_w_sle_status['patient'].isin(patients_in_batch_4)]
    group_4 = group_4[group_4['Processing_Cohort'] == str(4.0)]
    assert group_4['Processing_Cohort'].nunique() == 1
    group_123 = comps_w_sle_status[comps_w_sle_status['patient'].isin(other_patients)]
    assert group_123['Processing_Cohort'].nunique() == 3
    # NOW we have the two groups that we need-- one with only cohort 4 (n=96) and one with other samples from non-4 patients (n=228)
    # need to separate these out into predictor sets (components) and target variable (SLE Status)
    # also can rename the 4 group as training and the 123 group as testing

    last_comp = "comp_" + str(rank)
    cmp_train = group_4.loc[:, "comp_1":last_comp].to_numpy()
    y_train = group_4.loc[:, "SLE_status"].to_numpy()
    cmp_test = group_123.loc[:, "comp_1":last_comp].to_numpy()
    y_test = group_123.loc[:, "SLE_status"].to_numpy()

    # train + fit a logisitic regression model using cross validation ON ONLY THE TRAINING (GROUP 4) DATA
    log_reg = LogisticRegressionCV(random_state=0, max_iter = 10000, penalty = 'l1', solver = 'saga',
                                   scoring = "roc_auc",
                                    Cs = penalties_to_test)
    log_fit = log_reg.fit(cmp_train, y_train)

    # get decision function for ROC AUC
    sle_decisions = log_fit.decision_function(cmp_test)
    # validate the ROC AUC of the model 
    roc_auc = roc_auc_score(y_test, sle_decisions)
    print("The best ROC AUC is: ", roc_auc)

    return y_test, sle_decisions

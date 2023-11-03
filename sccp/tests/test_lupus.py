"""
Test prediction metrics for Lupus
"""
from ..logisticReg import getPf2ROC
from sklearn.metrics import roc_auc_score
from ..figures.common import openPf2


def test_Lupus_AUCROC():
    """Test ROC AUC for Lupus"""
    rank = 40
    X = openPf2(rank, "Lupus")

    condStatus = X.obs[
        ["Condition", "SLE_status", "Processing_Cohort", "patient"]
    ].drop_duplicates()
    condStatus = condStatus.set_index("Condition")

    y_test, sle_decisions = getPf2ROC(X.uns["Pf2_A"], condStatus)

    assert roc_auc_score(y_test, sle_decisions) > 0.925

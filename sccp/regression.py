import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score


def BC_prediction(factors: np.ndarray, patients_dim: list, rng=None):
    """Perform logistic regression.
    
    patients_dim should be just a list of the dims along the patient axis.
    """
    rng = np.random.default_rng(rng)
    assert len(patients_dim) == factors.shape[0]
    y = np.array([1 for s in xs if "BC" in s else 0], dtype=int)

    crossval = RepeatedStratifiedKFold(n_splits=10, random_state=rng)
    LR = LogisticRegression(random_state=rng)

    score = cross_val_score(LR, TFAC_X, Donor_CoH_y, cv=crossval)
    LR.fit(factors, y)

    return LR.coef_, score


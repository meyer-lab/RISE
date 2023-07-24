from typing import Sequence
import numpy as np


class Pf2X(Sequence):
    def __init__(self, X_list: list, condition_labels, variable_labels):
        assert isinstance(X_list, list)
        self.X_list = X_list
        self.condition_labels = np.array(condition_labels, dtype=object)
        self.variable_labels = np.array(variable_labels, dtype=object)
        assert len(X_list) == len(condition_labels)
        for X in X_list:
            assert X.shape[1] == len(variable_labels)

    def unfold(self):
        return np.concatenate(self.X_list, axis=0)

    def __getitem__(self, item):
        if item >= len(self.X_list):
            raise IndexError("Pf2X index out of range")
        return self.X_list[item]

    def __len__(self):
        return len(self.X_list)

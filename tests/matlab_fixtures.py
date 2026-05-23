import re

import numpy as np


def cell_array(values):
    inner = np.empty((1, len(values)), dtype=object)
    for index, value in enumerate(values):
        inner[0, index] = value

    outer = np.empty((1,), dtype=object)
    outer[0] = inner
    return outer


def mat_data_from_trials(labels, trials, time):
    return {
        "trial": cell_array([np.asarray(trial, dtype=float) for trial in trials]),
        "time": cell_array([np.asarray(time, dtype=float) for _ in trials]),
        "trialinfo": np.array([[np.asarray(labels, dtype=int)]], dtype=object),
    }


def loadmat_side_effect(data_by_participant):
    def loadmat(path):
        match = re.search(r"Part(\d+)Data\.mat$", str(path))
        if not match:
            raise AssertionError(f"Unexpected MAT path: {path}")
        participant = int(match.group(1))
        return {"data": np.array([data_by_participant[participant]], dtype=object)}

    return loadmat

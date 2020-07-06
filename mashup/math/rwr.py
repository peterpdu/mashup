#!/usr/bin/env python3.6

import numpy as np
# from scipy.linalg import lstsq


def rwr(a, restart_prob):
    n = a.shape[0]
    a = a - np.diag(np.diag(a))
    a = a + np.diag(a.sum() == 0)
    p = a / a.sum()
    # q, resid, rank, s = np.linalg.lstsq((np.identity(n) - (1 - restart_prob) * p), (restart_prob * np.identity(n)), rcond=None)
    q = np.linalg.solve((np.identity(n) - (1 - restart_prob) * p), (restart_prob * np.identity(n)))
    return q

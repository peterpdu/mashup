#!/usr/bin/env python3.6

import sys, os
import numpy as np
import pandas as pd
from scipy.spatial import distance


if __name__ == '__main__':
    # metric = 'cosine'
    # metric = 'correlation'
    # metric = 'dotprod'
    # metric = 'euclidean'
    metric = sys.argv[2]
    assert metric in ['cosine', 'correlation', 'dotprod', 'euclidean']

    f = sys.argv[1]
    outpath = os.path.splitext(f)[0] + f'_{metric}_corr.npy'

    # assumes data is features x genes
    if f.endswith('.npy'):
        a = np.load(f)
    else:
        a = pd.read_table(f, index_col=0).values
    if metric == 'dotprod':
        p = distance.squareform(distance.pdist(a.T, metric='cosine'))
        p = p * np.dot(a.T, a)
    else:
        p = distance.squareform(distance.pdist(a.T, metric=metric))
    np.fill_diagonal(p, 0)
    np.save(outpath, p)

#!/usr/bin/env python3.6

import numpy as np
import pandas as pd

from ..utils.matrix_convert import edge_to_adj


def load_network(f, genelist):
    with open(f, 'r') as handle:
        line = handle.readline()
        s = len(line.split('\t'))
        handle.close()
    # edge list
    if s == 3:
        print('Edge list detected')
        df = pd.read_table(f, header=None)
        a = edge_to_adj(df)
        a = a.reindex(genelist).T.reindex(genelist)
        a = a.replace(np.NaN, 0)
    # adjacency matrix
    elif s > 3:
        print('Adjacency matrix detected')
        a = pd.read_table(f, index_col=0)
        a = a.loc[genelist, genelist]
    else:
        raise ValueError('Incorrect input format detected')
    if not np.all(a.values == a.values.T):
        # symmetrize
        a = a + a.T
    a = a + np.diag(a.sum(axis=1) == 0)
    return a

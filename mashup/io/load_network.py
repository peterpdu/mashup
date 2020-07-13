#!/usr/bin/env python3.6

import numpy as np
import pandas as pd

from ..utils.matrix_convert import edge_to_adj

# TODO: directed networks
def load_network(f, genelist):
    # text file
    if f.endswith('.txt'):
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
    # numpy adj matrix
    elif f.endswith('.npy'):
        a = np.load(f)
        if a.shape != (len(genelist), len(genelist)):
            raise ValueError(f'npy matrix dimensions ({a.shape}) do not match gene list input shape ({len(genelist)})')
        a = pd.DataFrame(a, index=genelist, columns=genelist)
    else:
        raise ValueError('input file must be .txt or .npy')
    # TODO: don't need to symmetrize if directed
    if not np.all(a.values == a.values.T):
        # symmetrize
        a = a + a.T
    # set all zero vector diagonals to one
    a = a + np.diag(a.sum(axis=1) == 0)
    return a

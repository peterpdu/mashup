#!/usr/bin/env python3.6

import numpy as np
import pandas as pd
import networkx as nx


def load_network(f, genelist):
    with open(f, 'r') as handle:
        line = handle.readline()
        s = len(line.split('\t'))
        handle.close()
    # edge list
    if s == 3:
        print('Edge list detected')
        df = pd.read_table(f, header=None)
        G = nx.convert_matrix.from_pandas_edgelist(df, source=0, target=1, edge_attr=2)
        a = nx.convert_matrix.to_pandas_adjacency(G, weight=2)
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

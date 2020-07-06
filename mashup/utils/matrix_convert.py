#!/usr/bin/env python3.6

import numpy as np
import pandas as pd
import networkx as nx


def edge_to_adj(e):
    c = e.columns
    G = nx.convert_matrix.from_pandas_edgelist(e, source=c[0], target=c[1], edge_attr=c[2])
    return nx.convert_matrix.to_pandas_adjacency(G, weight=c[2])


def adj_to_edge(a, undirected=True):
    ac = a.copy()
    if undirected:
        ac.values[np.tril_indices(ac.shape[0], -1)] = np.NaN
    e = ac.stack().reset_index()
    e.columns = ['source', 'target', 'weight']
    return e

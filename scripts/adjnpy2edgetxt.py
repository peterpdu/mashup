#!/usr/bin/env python3.6

import sys, os
import numpy as np
import pandas as pd


def adj_to_edge(a, undirected=True):
    ac = a.copy().astype(float)
    if undirected:
        ac.values[np.tril_indices(ac.shape[0], -1)] = np.NaN
    e = ac.stack().reset_index()
    e.columns = ['source', 'target', 'weight']
    return e


# adjnpy2edgetxt.npy input.npy genes.txt
if __name__ == '__main__':
    a = np.load(sys.argv[1])
    genes = pd.read_table(sys.argv[2], header=None)[0].tolist()
    a = pd.DataFrame(a, index=genes, columns=genes)

    e = adj_to_edge(a)

    outpath = os.path.splitext(sys.argv[1])[0] + '_edge.txt'
    e.to_csv(outpath, sep='\t', header=None, index=None)

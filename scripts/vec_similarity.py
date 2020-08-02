#!/usr/bin/env python3.6

import sys, os
import argparse
import numpy as np
import pandas as pd
from scipy.spatial import distance

def getParser():
    parser = argparse.ArgumentParser()
    req = parser.add_argument_group('required')
    opt = parser.add_argument_group('optional')
    req.add_argument('--input',
                     help='m x n matrix (npy or txt); will calculate similarity between columns (n)',
                     type=str,
                     required=True)
    req.add_argument('--metric',
                     help='comparison metric; default=euclidean',
                     type=str,
                     choices=['cosine', 'correlation', 'dotprod', 'euclidean'],
                     default='euclidean')

    opt.add_argument('--sigma',
                     help='gaussian kernel sigma for converting distance to similarity; ignored for correlation; default=0.1',
                     type=float,
                     default=0.1)
    return parser


def gaussian(d, sigma):
    return np.power(np.e, -np.power(d, 2) / (2 * sigma ** 2))


if __name__ == '__main__':
    args = getParser().parse_args()
    f = args.input
    metric = args.metric
    sigma = args.sigma

    outpath = os.path.splitext(f)[0] + f'_{metric}.npy'

    # assumes data is features x genes; will compare genes
    if f.endswith('.npy'):
        a = np.load(f)
    else:
        a = pd.read_table(f, index_col=0).values

    if metric == 'correlation':
        p = np.corrcoef(a.T)
    elif metric == 'dotprod':
        p = distance.squareform(distance.pdist(a.T, metric='cosine'))
        p = p * np.dot(a.T, a)
        p = gaussian(p, sigma)
    else:
        p = distance.squareform(distance.pdist(a.T, metric=metric))
        p = gaussian(p, sigma)
    np.fill_diagonal(p, 0)
    np.save(outpath, p)

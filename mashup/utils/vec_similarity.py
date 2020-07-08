#!/usr/bin/env python3.6

import sys, os

import numpy as np
import pandas as pd


def correlate(a, method='pearson'):
    pairwise = a.corr(method=method)
    return pairwise


if __name__ == '__main__':
    f = sys.argv[1]
    outpath = os.path.splitext(f)[0] + '_corr.npy'
    a = np.load(f)
    df = pd.DataFrame(a)
    p = correlate(df)
    np.save(p.values, outpath)

#!/usr/bin/env python3.6

import os, sys

import numpy as np
import pandas as pd


if __name__ == '__main__':
    # first file is "base" for comparison
    f1 = sys.argv[1]
    f2 = sys.argv[2]
    genes = pd.read_table(sys.argv[3], header=None)[0].tolist()

    a1 = np.load(f1)
    a2 = np.load(f2)

    if len(genes) != a1.shape != a2.shape:
        raise ValueError('gene list does not match input matrix shape(s)')

    d = a2 - a1

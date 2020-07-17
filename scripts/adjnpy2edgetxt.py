#!/usr/bin/env python3.6

import sys, os
import numpy as np
import pandas as pd

from mashup.utils.matrix_convert import adj_to_edge


# adjnpy2edgetxt.npy input.npy genes.txt
if __name__ == '__main__':
    a = np.load(sys.argv[1])
    genes = pd.read_table(sys.argv[2], header=None)[0].tolist()
    a = pd.DataFrame(a, index=genes, columns=genes)

    e = adj_to_edge(a)

    outpath = os.path.splitext(sys.argv[1] + '_edge.txt')
    e.to_csv(outpath, sep='\t', header=None, index=None)

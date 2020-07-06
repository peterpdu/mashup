#!/usr/bin/env python3.6

import numpy as np
from scipy.sparse.linalg import eigsh

from .io.load_network import load_network
from .math.rwr import rwr
from .math.vector_embedding import vector_embedding


def mashup(infiles, genelist, ndim, do_svd):
    ngene = len(genelist)
    if do_svd:
        rr_sum = np.zeros(ngene)
        for f in infiles:
            print(f'Loading {f}')
            a = load_network(f, genelist)
            print('Running diffusion')
            q = rwr(a, 0.5)
            # smoothing
            r = np.log(q + 1/ngene)
            rr_sum = rr_sum + np.matmul(r, r.T)
            print()
        print('ALl networks loaded. Learning vectors via SVD...')
        w, v = eigsh(rr_sum, ndim, which='LM')
        d = np.diag(w)
        x = np.matmul(np.diag(np.sqrt(np.sqrt(np.diag(d)))), v.T)
    else:
        q_concat = []
        for f in infiles:
            print(f'Loading {f}')
            a = load_network(f, genelist)
            print('Running diffusion')
            q = rwr(a, 0.5)

            q_concat.append(q.copy())
        q_concat = np.concatenate(q_concat, axis=1)
        q_concat = q_concat / len(infiles)
        print('ALl networks loaded. Learning vectors via iterative optimization...')
        x = vector_embedding(q_concat, ndim, 1000)

    print('Mashup features obtained')
    return x

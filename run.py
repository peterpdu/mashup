#!/usr/bin/env python3.6

import os, argparse

import numpy as np
import pandas as pd

from mashup.mashup import mashup
from mashup.io.load_annot import load_go
from mashup.eval.cross_validation import cross_validation


def getParser():
    parser = argparse.ArgumentParser()
    req = parser.add_argument_group('required')
    opt = parser.add_argument_group('optional')
    req.add_argument('--networks',
                     help='3 column edge lists or square adjacency matrix',
                     type=str,
                     nargs='+',
                     required=True)
    req.add_argument('--genes',
                     help='gene list',
                     type=str,
                     required=True)
    opt.add_argument('--nfold',
                     help='cross validation trials; default=5',
                     type=int,
                     default=5)
    opt.add_argument('--no_svd',
                     action='store_true',
                     help='do not perform SVD approximation')
    opt.add_argument('--ndim',
                     help='number of dimensions; default=800',
                     type=int,
                     default=800)
    return parser


if __name__ == '__main__':
    args = getParser().parse_args()
    infiles = args.networks
    nfold = args.nfold
    do_svd = not args.no_svd
    ndim = args.ndim
    outdir = os.path.dirname(args.genes)
    genelist = pd.read_table(args.genes, header=None)[0].tolist()
    ngenes = len(genelist)

    # perform mashup
    print('[Mashup]')
    x = mashup(infiles, genelist, ndim, do_svd)

    outpath = os.path.join(outdir, 'mashup.npy')
    print(f'Saving outputs to {outpath}')
    np.save(outpath, x)

    exit(0)
    # load benchmark data
    print('[Loading annotations]')
    annot = load_go()
    print(f'Number of functional labels: {annot.shape}')

    # function prediction validation
    print('[Function prediction]')
    acc, f1, auprc = cross_validation(x, annot, nfold)

    # output performance stats
    print('[Performance]')
    print(f'Accuracy:\t{np.mean(acc)} (stdev = {np.std(acc)})')
    print(f'F1:\t{np.mean(f1)} (stdev = {np.std(f1)}')
    print(f'AUPRC:\t{np.mean(auprc)} (stdev = {np.std(auprc)}')

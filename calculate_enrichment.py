#!/usr/bin/env python3.6

import os, argparse

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

from mashup.eval.enrichment import hart_enrichment, pan_enrichment, wainberg_enrichment


mpl.rc('pdf', fonttype=42)
mpl.rc('font', family='sans-serif')
mpl.rc('font', serif='Helvetica')
mpl.rcParams['pdf.use14corefonts'] = True
mpl.rcParams['axes.unicode_minus'] = False


def getParser():
    parser = argparse.ArgumentParser()
    req = parser.add_argument_group('required')
    opt = parser.add_argument_group('optional')
    req.add_argument('--inputs',
                     help='square weighted adjacency npy(s)',
                     type=str,
                     required=True,
                     nargs='+')
    req.add_argument('--genes',
                     help='gene list for npy',
                     type=str,
                     required=True)
    req.add_argument('--database',
                     help='edge list or gene x class annotation database',
                     type=str,
                     required=True)

    opt.add_argument('--prefix',
                     help='output prefix file label; default=method flag',
                     type=str,
                     default=None)
    opt.add_argument('--method',
                     help='enrichment method; default=pan2018',
                     choices=('hart2017', 'pan2018', 'wainberg'),
                     default='pan2018')
    return parser


if __name__ == '__main__':
    args = getParser().parse_args()
    infiles = args.inputs
    outdir = os.path.dirname(infiles[0])
    method = args.method
    prefix = args.prefix
    if prefix is None:
        prefix = method

    # load genes
    genes = pd.read_table(args.genes, header=None)[0].tolist()

    # load annotations
    with open(args.database, 'r') as handle:
        line = handle.readline()
        s = len(line.split('\t'))
        handle.close()
    if s == 3:
        annot = pd.read_table(args.database, header=None)
    elif s > 3:
        annot = pd.read_table(args.database, index_col=0)
    else:
        raise ValueError('Incorrect database format detected')

    print('Calculating enrichments')
    results = pd.DataFrame()
    for f in infiles:
        print(f)
        name = os.path.basename(os.path.splitext(f)[0])
        a = np.load(f)
        df = pd.DataFrame(a, index=genes, columns=genes)

        if method == 'hart2017':
            enrichment = hart_enrichment(df, annot)
        elif method == 'pan2018':
            enrichment = pan_enrichment(df, annot)
        else:   # wainberg
            enrichment = wainberg_enrichment(df, annot)
        results[name] = enrichment
        print()

    # save results
    outpath = os.path.join(outdir, prefix + '_enrichments.txt')
    print('Writing results to {}'.format(outpath))
    results.to_csv(outpath, sep='\t')

    # save figure
    for c in results.columns:
        v = results[c]
        plt.plot(range(len(v)), v, label=c)
    plt.legend()
    plt.ylabel('{} enrichment'.format(method))
    plt.title(os.path.basename(os.path.splitext(args.database)[0]))

    outpath = os.path.join(outdir, prefix + '_enrichments.pdf')
    print('Saving figure to {}'.format(outpath))
    plt.savefig(outpath)

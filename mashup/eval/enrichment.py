#!/usr/bin/env python3.6

import numpy as np
import pandas as pd
import networkx as nx

from tqdm import tqdm

from ..utils.matrix_convert import adj_to_edge


def hart_enrichment(x, annot, bin_size=1000, num_bins=100):
    """
    [Hart et al., 2017]
    fraction of true positive interactions : fraction of false positive interactions (log2)
    in bins of 1000 ranked pairs
    :param x: adjacency matrix
    :param annot: edge list or gene x class matrix
    :param bin_size: ranked gene pairs over bins
    :param num_bins: number of bins to calculate
    :return: enrichment values
    """
    print('Converting observations to edge list')
    e = adj_to_edge(x)
    e = e.sort_values('weight', ascending=False)

    print('Building reference graph')
    if len(annot.columns) > 3:
        # gene x class matrix
        coocc = np.matmul(annot.values.T, annot.values)
        coocc = pd.DataFrame(coocc, index=annot.columns, columns=annot.columns)
        coocc.index.name = None
        annot = adj_to_edge(coocc)
    annot.columns = ['source', 'target', 'weight']
    G = nx.convert_matrix.from_pandas_edgelist(annot, edge_attr='weight')

    if len(e) < (num_bins * bin_size):
        num_bins = int(len(e) / bin_size)
    e = e.head(int(num_bins * bin_size))
    enrichment = []

    print('Calculating enrichment')
    for split in tqdm(np.split(e, num_bins), total=num_bins):
        split['annot'] = [G.has_edge(g1, g2) for g1, g2 in zip(split['source'], split['target'])]
        tp = split['annot'].sum()
        fp = (bin_size - split['annot'].sum())
        enrichment.append(np.log2(tp / fp))

    return enrichment


def pan_enrichment(x, annot, bin_size=1000, num_bins=100):
    """
    [Pan et al., 2018]
    fraction of true positive interactions : total number of interactions
    in bins of 1000 ranked pairs
    :param x: adjacency matrix
    :param annot: edge list or gene x class matrix
    :param bin_size: ranked gene pairs over windows
    :param num_bins: number of windows to calculate over
    :return: enrichment values
    """
    print('Converting observations to edge list')
    e = adj_to_edge(x)
    e = e.sort_values('weight', ascending=False)

    print('Building reference graph')
    if len(annot.columns) > 3:
        # gene x class matrix
        coocc = np.matmul(annot.values.T, annot.values)
        coocc = pd.DataFrame(coocc, index=annot.columns, columns=annot.columns)
        coocc.index.name = None
        annot = adj_to_edge(coocc)
    annot.columns = ['source', 'target', 'weight']
    G = nx.convert_matrix.from_pandas_edgelist(annot, edge_attr='weight')

    total_interaction_fraction = G.number_of_edges() / G.number_of_nodes() ** 2

    if len(e) < bin_size * num_bins:
        num_bins = int(len(e) / bin_size)
    e = e.head(int(num_bins * bin_size))
    enrichment = []

    print('Calculating enrichment')
    for split in tqdm(np.split(e, num_bins), total=num_bins):
        split['annot'] = [G.has_edge(g1, g2) for g1, g2 in zip(split['source'], split['target'])]
        tpr = split['annot'].sum() / bin_size
        enrichment.append(np.log2(tpr / total_interaction_fraction))

    return enrichment


def wainberg_enrichment(x, annot, n=10):
    """
    [Wainberg, Kamber, Balsubramani et al., preprint]
    fraction of true positive interactions : total number of interactions
    considering the top N partners
    :param x: adjacency matrix
    :param annot: edge list or gene x class matrix
    :param n: calculated with up to n partners
    :return: enrichment values
    """

    def top_k(arr, k):
        idx = np.argpartition(arr, -1 * k)[-1 * k:]
        return idx[np.argsort(arr[idx])]

    print('Building reference graph')
    if len(annot.columns) > 3:
        # gene x class matrix
        coocc = np.matmul(annot.values.T, annot.values)
        coocc = pd.DataFrame(coocc, index=annot.columns, columns=annot.columns)
        coocc.index.name = None
        annot = adj_to_edge(coocc)
    annot.columns = ['source', 'target', 'weight']
    G = nx.convert_matrix.from_pandas_edgelist(annot, edge_attr='weight')

    # num_edges / num_nodes ^ 2
    total_interaction_fraction = G.number_of_edges() / G.number_of_nodes() ** 2
    n_genes = x.shape[0]
    enrichment = []

    print('Calculating enrichment')
    top_idx = np.apply_along_axis(lambda _: top_k(_, n), 0, x.values)
    for i in tqdm(range(n), total=n):
        tp = 0
        # get top n partners
        for top_n in zip(x.columns, *[x.columns[top_idx[_]] for _ in range(i+1)]):
            # genes are all neighbors to root
            try:
                tp += all([_ in list(G.neighbors(top_n[0])) for _ in top_n[1:]])
            except nx.NetworkXError:
                pass
        enrichment.append(tp / n_genes / total_interaction_fraction)

    return enrichment

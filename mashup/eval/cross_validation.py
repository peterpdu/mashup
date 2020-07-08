#!/usr/bin/env python3.6

import numpy as np
import pandas as pd

from sklearn import svm
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score

from tqdm import tqdm

seed = 23

params = {
    'C': np.logspace(-2, 2, 5),
    'gamma': np.logspace(-3, 1, 5)
}

scoring_funcs = {
    'AUC': roc_auc_score,
    'F1': f1_score,
    'precision': precision_score,
    'recall': recall_score,
    'accuracy': accuracy_score
}


def cross_validation(x, annot, nfold=5, scoring_metric='F1', robust=False, member_min=10):
    """
    N fold cross validation with RBFk SVC for each class
    :param x: features x genes dataframe
    :param annot: class x genes dataframe (binary)
    :param nfold: number of cross validation folds
    :param scoring_metric: metric used to select best hyperparameters
    :param robust: use median across classes instead of mean
    :return: cross validation results table, holdout testing results table
    """
    if scoring_metric not in scoring_funcs.keys():
        raise ValueError('scoring_metric must be in {}'.format(list(scoring_funcs.keys())))

    # set genes to be equal across input and annotations
    annot = annot.T.reindex(x.columns).T
    annot = annot.replace(np.NaN, 0)

    annot = annot.loc[annot.sum(axis=1) >= member_min]
    nclasses = len(annot)
    # split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(x.T.values, annot.T.values, test_size=0.2, random_state=seed)

    # cross validation
    print('Performing {}-fold cross validation'.format(nfold))
    kf = KFold(n_splits=nfold)
    cv_results = pd.DataFrame()
    for c in params['C']:
        for g in params['gamma']:
            print('C: {}\tgamma: {}'.format(c, g))
            cv_metrics = {k: [] for k in scoring_funcs.keys()}
            for train_idx, test_idx in tqdm(kf.split(X_train), total=nfold):
                X_cv_train, X_cv_test = X_train[train_idx], X_train[test_idx]
                y_cv_train, y_cv_test = y_train[train_idx], y_train[test_idx]

                class_metrics = {k: [] for k in scoring_funcs.keys()}
                for s in range(nclasses):
                    y = y_cv_train[:, s]
                    svc = svm.SVC(C=c, gamma=g, kernel='rbf')

                    svc.fit(X_cv_train, y)
                    preds = svc.predict(X_cv_test)

                    for m in scoring_funcs.keys():
                        try:
                            score = scoring_funcs[m](y_cv_test[:, s], preds)
                        except ValueError:
                            score = 0
                        class_metrics[m].append(score)

                for m in cv_metrics.keys():
                    if robust:
                        cv_metrics[m].append(np.median(class_metrics[m]))
                    else:
                        cv_metrics[m].append(np.mean(class_metrics[m]))
            cv_res = pd.DataFrame(cv_metrics)
            cv_res['param_C'] = c
            cv_res['param_gamma'] = g
            cv_results = pd.concat([cv_results, cv_res])

    # pick best hyperparams
    print('Best hyperparameters')
    best_c, best_g = cv_results.sort_values(scoring_metric, ascending=False).iloc[0][['param_C', 'param_gamma']].values
    print('C: {}'.format(best_c))
    print('gamma: {}'.format(best_g))

    # holdout testing
    print('Performing holdout testing')
    test_results = {k: [] for k in scoring_funcs.keys()}
    for s in tqdm(range(nclasses), total=nclasses):
        y = y_train[:, s]
        svc = svm.SVC(C=best_c, gamma=best_g, kernel='rbf')
        svc.fit(X_train, y)
        preds = svc.predict(X_test)

        for m in scoring_funcs.keys():
            score = scoring_funcs[m](y_test[:, s], preds)
            test_results[m].append(score)
    test_results = pd.DataFrame(test_results)
    test_results.index = annot.index
    return cv_results, test_results

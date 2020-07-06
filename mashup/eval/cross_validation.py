#!/usr/bin/env python3.6

import numpy as np

from sklearn import svm
from sklearn.model_selection import GridSearchCV, train_test_split, KFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score

seed = 23

params = {
    'C': np.logspace(-2, 2, 5),
    'gamma': np.logspace(-3, 1, 5)
}

scoring = {
    'AUC': 'roc_auc',
    'F1': 'f1',
    'precision': 'precision',
    'recall': 'recall',
    'accuracy': 'accuracy'
}

scoring_funcs = {
    'AUC': roc_auc_score,
    'F1': f1_score,
    'precision': precision_score,
    'recall': recall_score,
    'accuracy': accuracy_score
}


def cross_validation(x, annot, nfold):
    # split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(x.T, annot.T, test_size=0.2, random_state=seed)

    # cross validation
    kf = KFold(n_splits=nfold)
    for c in params['C']:
        for g in params['gamma']:
            for train_idx, test_idx in kf.split(X_train):
                X_cv_train, X_cv_test = X_train[train_idx], X_train[test_idx]
                y_cv_train, y_cv_test = y_train[train_idx], y_train[test_idx]
                for s in y_cv_train.columns:
                    svc = svm.SVC(C=c, gamma=g, kernel='rbf')
                    svc.fit(X_cv_train[s], y_cv_train[s])
                    preds = svc.predict(X_cv_test[s])

                    for m in scoring_funcs.keys():
                        score = scoring_funcs[m](y_cv_test, preds)


    svc = svm.SVC()
    clf = GridSearchCV(svc, params,
                       scoring=scoring, cv=nfold, refit=False,
                       return_train_score=False, n_jobs=-1)
    clf.fit(X_train, y_train)
    clf.cv_results_.best_params
    return acc, f1, auprc

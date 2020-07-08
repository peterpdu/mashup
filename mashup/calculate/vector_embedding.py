#!/usr/bin/env python 3.6

import numpy as np
from scipy.optimize import minimize, fmin_l_bfgs_b


def vector_embedding(q, ndim, maxiter):
    def optim_fn(wx):
        wx = wx.reshape(ndim, nnode + ncontext)

        p = p_fn(wx[:, :ncontext], wx[:, ncontext:])

        fval = obj_fn(p)

        wgrad = np.matmul(wx[:, ncontext:], (p - q))
        xgrad = np.matmul(wx[:, :ncontext], (p - q).T)
        grad = np.concatenate([wgrad, xgrad], axis=1)
        return fval, grad.reshape(-1)

    def p_fn(w, x):
        p = np.exp(np.matmul(x.T, w))
        return p / p.sum()

    def obj_fn(p):
        v = np.zeros((ncontext, 1))
        for j in range(ncontext):
            v[j] = kldiv(q[:, j], p[:, j])
        return np.sum(v)

    def kldiv(p, q):
        idx = p > 0
        return np.sum(p[idx] * np.log(p[idx] / q[idx]))

    nnode, ncontext = q.shape
    # nparam = (nnode + ncontext) * ndim

    # options for minimize
    # opts = {
    #     'ftol': 1E4 * np.finfo(float).eps,
    #     'gtol': 0,
    #     'maxcor': 5,
    #     'iprint': 50,
    #     'maxiter': maxiter
    # }

    # options for fmin_l_bfgs_b
    opts = {
        'factr': 1E4,
        'pgtol': 0,
        'm': 5,
        'iprint': 50,
        'maxiter': maxiter
    }

    while True:
        # init vectors
        print('Initializing vectors...')
        wx = np.random.uniform(-0.05, 0.05, (ndim, nnode + ncontext))

        # res = minimize(optim_fn, wx, method='L-BFGS-B', jac=True, bounds=None, options=opts)
        xopt, fval, info = fmin_l_bfgs_b(optim_fn, wx, fprime=None, **opts)
        if info['nit'] > 10:
            break
        print(f'Premature termination (took {info["nit"]} iter to converge); trying again')

    wx = xopt.reshape(ndim, nnode + ncontext)
    print('Done')

    w = wx[:, :ncontext]
    x = wx[:, ncontext:]
    p = p_fn(w, x)
    fval = obj_fn(p)

    return w, x, p, fval



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auxiliary functions.
"""
# @author: drmccloy
# Created on Tue Aug  9 10:58:22 2016
# License: BSD (3-clause)

import numpy as np


def pca(cov, max_components=None, thresh=0):
    """Perform PCA decomposition from a covariance matrix

    Parameters
    ----------
    cov : array-like
        Covariance matrix
    max_components : int | None
        Maximum number of components to retain after decomposition. ``None``
        (the default) keeps all suprathreshold components (see ``thresh``).
    thresh : float | None
        Threshold (relative to the largest component) above which components
        will be kept. The default keeps all non-zero values; to keep all
        values, specify ``thresh=None`` and ``max_components=None``.

    Returns
    -------
    eigval : array
        1-dimensional array of eigenvalues.
    eigvec : array
        2-dimensional array of eigenvectors.
    """
    if thresh is not None and (thresh > 1 or thresh < 0):
        raise ValueError('Threshold must be between 0 and 1 (or None).')
    eigval, eigvec = np.linalg.eigh(cov)
    eigval = np.abs(eigval)
    sort_ix = np.argsort(eigval)[::-1]
    eigvec = eigvec[:, sort_ix]
    eigval = eigval[sort_ix]
    if max_components is not None:
        eigval = eigval[:max_components]
        eigvec = eigvec[:, :max_components]
    if thresh is not None:
        suprathresh = np.where(eigval / eigval.max() > thresh)[0]
        eigval = eigval[suprathresh]
        eigvec = eigvec[:, suprathresh]
    return eigval, eigvec


def dss(data, pca_max_components=None, pca_thresh=0, bias_max_components=None,
        bias_thresh=0, norm=False, return_data=False, return_power=False):
    from mne import BaseEpochs
    if isinstance(data, BaseEpochs):
        data = data.get_data()
    # norm each channel's time series
    if norm:
        channel_norms = np.linalg.norm(data, ord=2, axis=-1)
        data = data / channel_norms[:, :, np.newaxis]
    # PCA across channels
    data_cov = np.einsum('hij,hkj->ik', data, data)
    data_eigval, data_eigvec = pca(data_cov, max_components=pca_max_components,
                                   thresh=pca_thresh)
    # diagonal data-PCA whitening matrix:
    W = np.diag(np.sqrt(1 / data_eigval))
    """
    # make sure whitening works
    white_data = W @ data_eigvec.T @ data
    white_data_cov = np.einsum('hij,hkj->ik', white_data, white_data)
    assert np.allclose(white_data_cov, np.eye(white_data_cov.shape[0]))
    """
    # compute bias rotation matrix
    evoked = data.mean(axis=0)
    bias_cov = np.einsum('hj,ij->hi', evoked, evoked)
    biased_white_eigvec = W @ data_eigvec.T @ bias_cov @ data_eigvec @ W
    bias_eigval, bias_eigvec = pca(biased_white_eigvec,
                                   max_components=bias_max_components,
                                   thresh=bias_thresh)
    # compute DSS operator
    dss_mat = data_eigvec @ W @ bias_eigvec
    dss_normalizer = 1 / np.sqrt(np.diag(dss_mat.T @ data_cov @ dss_mat))
    dss_operator = dss_mat @ np.diag(dss_normalizer)
    results = [dss_operator]
    # apply DSS to data
    if return_data:
        data_dss = np.einsum('hij,ik->hkj', data, dss_operator)
        results.append(data_dss)
    if return_power:
        unbiased_power = _power(data_cov, dss_operator)
        biased_power = _power(bias_cov, dss_operator)
        results.extend([unbiased_power, biased_power])
    return tuple(results)


def _power(cov, dss):
    return np.sqrt(((cov @ dss) ** 2).sum(axis=0))


def time_domain_pca(epochs, max_components=None):
    """ NORMAL WAY
    time_cov = np.sum([trial.T @ trial for trial in epochs], axis=0)
    eigval, eigvec = pca(time_cov, max_components=max_components, thresh=1e-6)
    W = np.sqrt(1 / eigval)  # whitening diagonal
    epochs = np.array([trial @ eigvec * W[np.newaxis, :] for trial in epochs])
    """
    time_cov = np.einsum('hij,hik->jk', epochs, epochs)
    eigval, eigvec = pca(time_cov, max_components=max_components, thresh=1e-6)
    W = np.diag(np.sqrt(1 / eigval))  # diagonal time-PCA whitening matrix
    epochs = np.einsum('hij,jk,kk->hik', epochs, eigvec, W)
    return epochs


def print_elapsed(start_time, end=' sec.\n'):
    from time import time
    print(np.round(time() - start_time, 1), end=end)


def _eer(probs, thresholds, pos, neg):
    # predictions
    guess_pos = probs[np.newaxis, :] >= thresholds[:, np.newaxis]
    guess_neg = np.logical_not(guess_pos)
    # false pos / false neg
    false_pos = np.logical_and(guess_pos, neg)
    false_neg = np.logical_and(guess_neg, pos)
    # false pos/neg rates for each threshold step (ignore div-by-zero warnings)
    err_state = np.seterr(divide='ignore')
    false_pos_rate = false_pos.sum(axis=1) / neg.sum()
    false_neg_rate = false_neg.sum(axis=1) / pos.sum()
    np.seterr(**err_state)
    # get rid of infs and zeros
    false_pos_rate[false_pos_rate == np.inf] = 1e9
    false_neg_rate[false_neg_rate == np.inf] = 1e9
    false_pos_rate[false_pos_rate == -np.inf] = -1e9
    false_neg_rate[false_neg_rate == -np.inf] = -1e9
    false_pos_rate[false_pos_rate == 0.] = 1e-9
    false_neg_rate[false_neg_rate == 0.] = 1e-9
    # FPR / FNR ratio
    ratios = false_pos_rate / false_neg_rate
    reverser = -1 if np.any(np.diff(ratios) < 0) else 1
    # find crossover point
    ix = np.searchsorted(ratios[::reverser], v=1.)
    closest_threshold_index = len(ratios) - ix if reverser < 0 else ix
    # check for convergence
    converged = np.isclose(ratios[closest_threshold_index], 1.)
    # return EER estimate
    eer = np.max([false_pos_rate[closest_threshold_index],
                  false_neg_rate[closest_threshold_index]])
    return closest_threshold_index, converged, eer


def _EER_score_threshold(estimator, X, y):
    # higher return values are better than lower return values
    probs = estimator.predict_proba(X)[:, 1]
    thresholds = np.linspace(0, 1, 101)
    converged = False
    threshold = -1
    # ground truth
    pos = np.array(y, dtype=bool)
    neg = np.logical_not(pos)
    while not converged:
        ix, converged, eer = _eer(probs, thresholds, pos, neg)
        old_threshold = threshold
        threshold = thresholds[ix]
        converged = converged or np.isclose(threshold, old_threshold)
        low = (ix - 1) if ix > 0 else ix
        high = low + 1
        thresholds = np.linspace(thresholds[low], thresholds[high], 101)
    return (1 - eer), threshold


def EER_score(estimator, X, y):
    """EER scoring function
    estimator:
        sklearn estimator
    X: np.ndarray
        training data
    y: np.ndarray
        training labels
    """
    return _EER_score_threshold(estimator, X, y)[0]


def EER_threshold(clf, X, y, return_eer=False):
    """Get EER threshold
    clf: sklearn.GridSearchCV
        the fitted crossval object
    X: np.ndarray
        training data
    y: np.ndarray
        training labels
    """
    estimator = clf.estimator
    estimator.C = clf.best_params_['C']
    estimator.fit(X, y)
    score, threshold = _EER_score_threshold(estimator, X, y)
    result = (threshold, (1 - score)) if return_eer else threshold
    return result


def map_ipa_to_feature(ipa, feature, features_file):
    # load features
    import pandas as pd
    feats = pd.read_csv(features_file, sep='\t', comment='#', header=0,
                        index_col=0)
    feat_mapper = feats[feature].to_dict()
    return feat_mapper.get(ipa, np.nan)

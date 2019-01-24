#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
Script 'make-confmat.py'
===============================================================================

This script converts feature-level error rates (from a bank of binary
phonological feature classifiers) into a matrix of phone confusion
probabilities.
"""
# @author: drmccloy
# Created on Tue Jun 06 13:13:09 2017
# License: BSD (3-clause)

import os
import sys
import yaml
import mne
import numpy as np
import pandas as pd
from aux_functions import map_ipa_to_feature

subj_code = sys.argv[1]      # IJ, IQ, etc
subj_num = int(sys.argv[2])  # integer
analysis = sys.argv[3]       # ovr, pairwise, featural
trunc_dur = int(sys.argv[4])

sparse_value = 0.5  # or np.nan; results don't differ much

# load params
paramdir = 'params'
with open(os.path.join(paramdir, 'params.yaml'), 'r') as f:
    params = yaml.load(f)
    n_comp = params['n_dss_comp']

# load list of english consonants
phone_file = os.path.join(paramdir, 'canonical-english-consonant-order.yaml')
with open(phone_file, 'r') as f:
    eng_phones = yaml.load(f)

# load features
features_file = os.path.join(paramdir, 'feature-tables', 'all-features.tsv')
ground_truth = pd.read_csv(features_file, sep='\t', index_col=0, comment='#')
ground_truth = ground_truth.loc[eng_phones]
features = ground_truth.columns.tolist()

# file I/O
folder = f'logistic-{trunc_dur}' if analysis == 'featural' else analysis

# load EERs
eerfile = os.path.join('processed-data', folder, 'eers.csv')
eers = pd.read_csv(eerfile, index_col=0)
this_eer = eers[subj_code]

# we'll use this index a few times...
eng_index = pd.Index(eng_phones, name='ipa_in')

# make 3d array of EERs. Each feature plane of shape (ipa_in,
# ipa_out) has a uniform value corresponding to the EER for that
# feature.
eer_df = pd.DataFrame({p: this_eer for p in eng_index}).T
eer_df.index.name, eer_df.columns.name = 'ipa_out', 'features'
eer_3d = pd.Panel({p: eer_df for p in eng_index}, items=eng_index)

# make 3d arrays of feature values where true feature values are
# repeated along orthogonal planes (i.e., feats_in.loc['p'] looks
# like feats_out.loc[:, 'p'].T)
feats_out = pd.Panel({p: ground_truth for p in eng_index}, items=eng_index)
feats_in = feats_out.copy().swapaxes(0, 1)

# intersect feats_in with feats_out -> boolean feature_match array
feat_match = np.logical_not(np.logical_xor(feats_in, feats_out))

# where features mismatch, insert the EER for that feature.
# where they match, insert 1. - EER.
prob_3d = eer_3d.copy()
match_ix = np.where(feat_match)
prob_3d.values[match_ix] = 1. - prob_3d.values[match_ix]

# handle feature values that are "sparse" in this feature system
sparse_mask = np.where(np.isnan(feats_out))
prob_3d.values[sparse_mask] = sparse_value

# collapse across features to compute joint probabilities
axis = [x.name for x in prob_3d.axes].index('features')
''' this one-liner can be numerically unstable, use the three-liner below
joint_prob = prob_3d.prod(axis=axis, skipna=True).swapaxes(0, 1)
'''
log_prob_3d = (-1. * prob_3d.apply(np.log))
joint_log_prob = (-1. * log_prob_3d.sum(axis=axis)).swapaxes(0, 1)
joint_prob = joint_log_prob.apply(np.exp)

# TODO: save joint_prob
outfile = os.path.join('processed-data', folder, 'confusion-matrices', 
                       subj_code, 'eer-confmat-eng-cvalign-dss5-{featsys}.csv')
joint_prob.to_csv(outfile)


raise RuntimeError

if analysis == 'logistic':
    for feature in features:
        this_file = (f'classifier-probabilities-eng-cvalign-redux-dss{n_comp}-'
                     f'{feature}.csv')
        this_df = pd.read_csv(os.path.join(indir, this_file))

elif analysis == 'ovr':
    for phone in eng_phones:
        this_file = (f'classifier-probabilities-eng-cvalign-redux-dss{n_comp}-'
                     f'ovr_{phone}.csv')
        this_df = pd.read_csv(os.path.join(indir, this_file))

elif analysis == 'pairwise':
    for contrast in contrasts:
        this_file = (f'classifier-probabilities-eng-cvalign-redux-dss{n_comp}-'
                     f'pairwise_{contrast}.csv')
        this_df = pd.read_csv(os.path.join(indir, this_file))

else:
    raise RuntimeError(f'invalid argument "{analysis}"; must be one of "ovr", '
                       '"pairwise", or "featural".')



#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
Script 'time-domain-redux.py'
===============================================================================

This script runs PCA on the time domain of epoched data, to reduce colinearity
of "features" (AKA time points) prior to classification, and then unrolls the
EEG channels (or DSS components, if using DSS) before saving.
"""
# @author: drmccloy
# Created on Tue Aug  8 13:33:19 PDT 2017
# License: BSD (3-clause)

import os
import sys
import yaml
import mne
import numpy as np
from time import time
from aux_functions import time_domain_pca, print_elapsed

subj_code = sys.argv[1]
subj_num = int(sys.argv[2])
trunc_durs = list(map(int, sys.argv[3:]))

dss = True

# basic file I/O
paramdir = 'params'
eegdir = 'eeg-data-clean'
styledir = 'styles'

# load params
with open(os.path.join(paramdir, 'params.yaml'), 'r') as f:
    params = yaml.load(f)
    n_comp = params['n_dss_comp']

# loop over truncation durations
for trunc_dur in trunc_durs:
    prefix = 'dss' if dss else 'epochs'
    indir = os.path.join(eegdir, f'{prefix}-{trunc_dur}')
    outdir = os.path.join(eegdir, f'td-redux-{prefix}-{trunc_dur}')
    basename = f'{subj_num:03}-{subj_code}-cvalign-'
    # load data
    print(f'loading data for subject {subj_code}, {trunc_dur}', end=': ')
    _st = time()
    if dss:
        data = np.load(os.path.join(indir, basename + 'dss-data.npy'))
        data = data[:, :n_comp, :]
    else:
        epochs = mne.read_epochs(os.path.join(indir, basename + 'epo.fif.gz'),
                                 verbose=False)
        data = epochs.get_data()
    print_elapsed(_st)

    # reduce time-domain dimensionality
    print('running PCA on time domain', end=': ')
    _st = time()
    data = time_domain_pca(data, max_components=None)
    print_elapsed(_st)

    # unroll data / concatenate channels
    data = data.reshape(data.shape[0], -1)  # (n_epochs, n_chans * n_times)

    # save
    file_ext = f'dss{n_comp}-data.npy' if dss else 'epoch-data.npy'
    out_fname = basename + 'redux-' + file_ext
    np.save(os.path.join(outdir, out_fname), data, allow_pickle=False)

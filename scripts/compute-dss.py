#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
Script 'compute-dss.py'
===============================================================================

This script loads EEG data in mne.Epochs format and applies denoising source
separation (DSS).
"""
# @author: drmccloy
# Created on Tue Nov 15 12:32:11 2016
# License: BSD (3-clause)

import os
import sys
import yaml
import mne
import numpy as np
from aux_functions import dss

subj_code = sys.argv[1]
subj_num = int(sys.argv[2])
trunc_durs = list(map(int, sys.argv[3:]))

# basic file I/O
eegdir = 'eeg-data-clean'

for trunc_dur in trunc_durs:
    indir = os.path.join(eegdir, f'epochs-{trunc_dur}')
    outdir = os.path.join(eegdir, f'dss-{trunc_dur}')
    basename = f'{subj_num:03}-{subj_code}-cvalign-'

    # read epochs
    epochs = mne.read_epochs(os.path.join(indir, basename + 'epo.fif.gz'))
    epochs.apply_proj()
    data = epochs.get_data()

    # compute DSS matrix
    dss_mat, dss_data = dss(data, pca_thresh=0.01, return_data=True)

    # save
    fpath = os.path.join(outdir, basename)
    np.save(fpath + 'dss-mat.npy', dss_mat, allow_pickle=False)
    np.save(fpath + 'dss-data.npy', dss_data, allow_pickle=False)

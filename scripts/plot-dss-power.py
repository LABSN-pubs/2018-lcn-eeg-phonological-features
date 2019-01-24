#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
Script 'validate-dss.py'
===============================================================================

This script loads EEG data in mne.Epochs format, and DSS'ed versions of the
same data, for interactive inspection and plotting.
"""
# @author: drmccloy
# Created on Mon Apr 10 13:57:27 2017
# License: BSD (3-clause)

import os
import yaml
import mne
import numpy as np
import matplotlib.pyplot as plt
from aux_functions import dss

# basic file I/O
paramdir = 'params'
eegdir = 'eeg-data-clean'
outdir = 'figures'
styledir = 'styles'

# load params
with open(os.path.join(paramdir, 'subjects.yaml'), 'r') as f:
    subjects = yaml.load(f)

with open(os.path.join(paramdir, 'params.yaml'), 'r') as f:
    params = yaml.load(f)
    n_comp = params['n_dss_comp']

# iterate over subjects
fig, axs = plt.subplots(3, 4, figsize=(12, 9), sharex=True, sharey=True)
fig.suptitle('Relative power of DSS components')
fig.text(0.5, 0.04, 'DSS component number', ha='center')
fig.text(0.04, 0.5, 'relative power', va='center', rotation='vertical')
for ix, (subj_code, subj_num) in enumerate(subjects.items()):
    basename = f'{subj_num:03}-{subj_code}-cvalign-'
    ax = axs.ravel()[ix]
    # read epochs
    epochs = mne.read_epochs(os.path.join(eegdir, 'epochs-0',
                                          basename + 'epo.fif.gz'))
    data = epochs.get_data()
    # load DSS matrix
    dss_mat = np.load(os.path.join(eegdir, 'dss-0', basename + 'dss-mat.npy'))
    # compute power
    evoked = data.mean(axis=0)
    bias_cov = np.einsum('hj,ij->hi', evoked, evoked)
    biased_power = np.sqrt(((bias_cov @ dss_mat) ** 2).sum(axis=0))
    # plot powers
    dss_pow = biased_power / biased_power.max()
    dss_line = ax.plot(dss_pow, label='DSS component power')
    ax.axvline(n_comp-1, color='k', linewidth=0.5, linestyle='dashed')
    ax.set_title(subj_code)
# finish plots
ax.annotate('smallest\nretained\ncomponent', xy=(n_comp-1, dss_pow[n_comp-1]),
            xytext=(15, 15), textcoords='offset points', ha='left',
            va='bottom', arrowprops=dict(arrowstyle='->'))
fig.savefig(os.path.join(outdir, 'dss-component-power.pdf'))

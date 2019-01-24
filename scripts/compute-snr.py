#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
Script 'compute-snr.py'
===============================================================================

This script measures the SNR of some epoched EEG recordings, by comparing
evoked power to power during the baseline period of each trial.
"""
# @author: drmccloy
# Created on Wed Jul 19 16:31:35 PDT 2017
# License: BSD (3-clause)

import os
import sys
import yaml
import mne
import numpy as np
import pandas as pd

subj_code = sys.argv[1]
subj_num = int(sys.argv[2])

# basic file I/O. Don't use CV-aligned epochs (need aligned baseline)
eegdir = 'eeg-data-clean'
indir = os.path.join(eegdir, 'epochs')

# read Epochs
basename = f'{subj_num:03}-{subj_code}-'
fname = os.path.join(indir, basename + 'epo.fif.gz')
epochs = mne.read_epochs(fname, verbose=False)

# compute SNR
baseline = epochs.copy().crop(*epochs.baseline)
stim_evoked = epochs.copy().crop(tmin=epochs.baseline[1])
evoked_power = stim_evoked.pick_types(eeg=True).get_data().var()
baseline_power = baseline.pick_types(eeg=True).get_data().var()
snr = 10 * np.log10(evoked_power / baseline_power)
# write SNR to summary file
with open(os.path.join(eegdir, 'snr-summary.csv'), 'a') as outfile:
    outfile.write(f'{subj_code},{snr}\n')

# while we've got the epochs loaded: which retained epochs are English talkers?
eps = [{v: k for k, v in epochs.event_id.items()}[e]
       for e in epochs.events[:, 2]]
eng_eps = [e for e in eps if e.startswith('eng')]
# write number of English-talker retained epochs to summary file
with open(os.path.join(eegdir, 'eng-epoch-summary.csv'), 'a') as outfile:
    outfile.write(f'{subj_code},{len(eng_eps)}\n')

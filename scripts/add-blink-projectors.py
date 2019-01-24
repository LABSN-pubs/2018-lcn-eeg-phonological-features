#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
Script 'add-eeg-projectors.py'
===============================================================================

This script reads EEG data from mne.io.Raw format, runs blink detection, and
re-saves in mne.io.Raw format with an SSP projector added that removes blink
artifacts. Also filters the data, and sets the reference channel and then drops
it after subtraction.
"""
# @author: drmccloy
# Created on Tue Jul 11 15:07:09 2017
# License: BSD (3-clause)

import os
import sys
import yaml
import mne
from mne.preprocessing import find_eog_events

subj_code = sys.argv[1]
subj_num = int(sys.argv[2])

# basic file I/O
root = '.'
eegdir = os.path.join(root, 'eeg-data-clean')
evdir = os.path.join(eegdir, 'events')
bldir = os.path.join(eegdir, 'blinks')
indir = os.path.join(eegdir, 'raws-annotated')
outdir = os.path.join(eegdir, 'raws-with-projectors')
paramdir = os.path.join(root, 'params')

# load params
with open(os.path.join(paramdir, 'params.yaml'), 'r') as f:
    params = yaml.load(f)
    n_jobs = params['n_jobs']
    ref_chan = params['ref_channel']
    blink_chan = params['blink_channel']

blink_chan = blink_chan.get(subj_code, blink_chan['default'])

# read Raws and events
basename = f'{subj_num:03}-{subj_code}-'
events = mne.read_events(os.path.join(evdir, basename + 'eve.txt'), mask=None)
raw_fname = os.path.join(indir, basename + 'raw.fif.gz')
raw = mne.io.read_raw_fif(raw_fname, preload=True)
sfreq = raw.info['sfreq']

# set EEG reference
mne.io.set_eeg_reference(raw, ref_chan, copy=False)
raw.drop_channels(ref_chan)

# filter
picks = mne.pick_types(raw.info, meg=False, eeg=True)
raw.filter(l_freq=0.1, h_freq=40., picks=picks, n_jobs=n_jobs)

# add blink projector to raw
blink_events = find_eog_events(raw, ch_name=blink_chan,
                               reject_by_annotation=True)
blink_epochs = mne.Epochs(raw, blink_events, event_id=998, tmin=-0.5,
                          tmax=0.5, proj=False, reject=None, flat=None,
                          baseline=None, picks=picks, preload=True,
                          reject_by_annotation=True)
ssp_blink_proj = mne.compute_proj_epochs(blink_epochs, n_grad=0, n_mag=0,
                                         n_eeg=4, n_jobs=n_jobs,
                                         desc_prefix=None, verbose=None)
raw = raw.add_proj(ssp_blink_proj)

# save raw with projectors added; save blinks for later plotting / QA
raw.save(os.path.join(outdir, basename + 'raw.fif.gz'), overwrite=True)
mne.write_events(os.path.join(bldir, basename + 'blink-eve.txt'), blink_events)
blink_epochs.save(os.path.join(bldir, basename + 'blink-epo.fif.gz'))

with open(os.path.join(bldir, 'blink-summary.csv'), 'a') as outfile:
    outfile.write(f'{subj_code},{len(blink_epochs)}\n')

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
Script 'annotate-eeg-interactive.py'
===============================================================================

This script opens and plots MNE raw files for (re)annotation. It also runs
blink detection before and after annotation, as a way of checking whether the
level of annotation is adequate to allow the blink detection algorithm to
perform reasonably well.
"""
# @author: drmccloy
# Created on Fri Apr  7 09:43:03 2017
# License: BSD (3-clause)

import os
import sys
import yaml
import mne
from mne.preprocessing import find_eog_events

subj_code = sys.argv[1]
subj_num = int(sys.argv[2])
rerun = bool(int(sys.argv[3]))
save = bool(int(sys.argv[4]))

# basic file I/O
root = '.'
indir = os.path.join(root, 'eeg-data-clean')
outdir = os.path.join(indir, 'raws-annotated')
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
_dir = outdir if rerun else os.path.join(indir, 'raws')
fname = os.path.join(_dir, basename + 'raw.fif.gz')
raw = mne.io.read_raw_fif(fname, preload=True)
# make a copy when setting reference channel
raw_ref, _ = mne.io.set_eeg_reference(raw, ref_chan, copy=True)
# filter
picks = mne.pick_types(raw_ref.info, meg=False, eeg=True)
raw_ref.filter(l_freq=0.1, h_freq=40., picks=picks, n_jobs=n_jobs)
# see how well blink algorithm works before annotation
blink_args = dict(ch_name=blink_chan, reject_by_annotation=True)
blink_events = find_eog_events(raw_ref, **blink_args)
# create blink projector, so we can toggle it on and off during annotation
blink_epochs = mne.Epochs(raw_ref, blink_events, event_id=998, tmin=-0.5,
                          tmax=0.5, proj=False, reject=None, flat=None,
                          baseline=None, picks=picks, preload=True,
                          reject_by_annotation=True)
ssp_blink_proj = mne.compute_proj_epochs(blink_epochs, n_grad=0, n_mag=0,
                                         n_eeg=5, n_jobs=n_jobs,
                                         desc_prefix=None, verbose=None)
raw_ref = raw_ref.add_proj(ssp_blink_proj)
# interactive annotation: mark bad channels & transient noise during blocks
raw_ref.plot(n_channels=33, duration=30, events=blink_events, block=True,
             scalings=dict(eeg=50e-6))
# compare blink algorithm performance after annotation
new_blink_events = find_eog_events(raw_ref, **blink_args)
print('#############################')
print(f'old: {blink_events.shape[0]}')
print(f'new: {new_blink_events.shape[0]}')
print('#############################')
if save:
    # copy the annotation data to the (unfiltered, unreferenced) Raw object
    raw.annotations = raw_ref.annotations
    # save annotated Raw object. If re-running, change overwrite to True
    raw.save(os.path.join(outdir, basename + 'raw.fif.gz'), overwrite=rerun)

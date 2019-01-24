#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
Script 'make-epochs.py'
===============================================================================

This script processes EEG data from mne.io.Raw format into epochs. Baseline
correction is done on the 100ms preceding each stimulus onset; after baseline
correction the epochs are temporally shifted to place time-0 at the
consonant-vowel boundary of each stimulus syllable. Finally, epochs are
(optionally) truncated to begin after the CV transition, to limit the
available consonant information to later stages of processing (i.e., no
information from cortical representations of acoustic/spectrotemporal
properties, instead just cortical representations of abstract category).
"""
# @author: drmccloy
# Created on Tue Nov 15 12:32:11 2016
# License: BSD (3-clause)

import os
import sys
import yaml
import mne
import numpy as np
import pandas as pd

subj_code = sys.argv[1]
subj_num = int(sys.argv[2])
trunc_durs = list(map(int, sys.argv[3:]))

# basic file I/O
eegdir = 'eeg-data-clean'
indir = os.path.join(eegdir, 'raws-with-projectors')
paramdir = 'params'
# NB: outdir defined at end (different for each value of trunc_durs)

# load params
with open(os.path.join(paramdir, 'params.yaml'), 'r') as f:
    params = yaml.load(f)
    brain_resp_dur = params['brain_resp_dur']
    blink_chan = params['blink_channel']
    bad_chan = params['bad_channels']
    ref_chan = params['ref_channel']
    reject = params['reject']
    n_jobs = params['n_jobs']
# load more params
stim_params = np.load(os.path.join(paramdir, 'stim-params.npz'))
stim_fs = stim_params['fs'][()]
isi_range = stim_params['isi_range']
wav_names = stim_params['wav_names']
wav_nsamp = stim_params['wav_nsamps']
wav_idx = np.arange(wav_nsamp.shape[0])
wav_params = pd.DataFrame(dict(wav_path=wav_names, wav_nsamp=wav_nsamp, 
                               wav_idx=wav_idx))

# set blink channel
blink_chan = blink_chan.get(subj_code, blink_chan['default'])

# LOAD DURATION DATA...
df = pd.read_csv(os.path.join(paramdir, 'cv-boundary-times.tsv'), sep='\t')
df['key'] = df['talker'] + '/' + df['consonant'] + '.wav'
# make sure all keys have a CV transition time
assert set(wav_params['wav_path']) - set(df['key']) == set([])
# merge in wav params (eventID, nsamp)
df = df.merge(wav_params, how='right', left_on='key', right_on='wav_path')
# compute word and vowel durations
df['w_dur'] = df['wav_nsamp'] / stim_fs
df['v_dur'] = df['w_dur'] - df['cv_transition_time']
df.rename(columns={'cv_transition_time': 'c_dur', 'wav_idx': 'event_id',
                   'wav_nsamp': 'nsamp'}, inplace=True)
df = df[['event_id', 'key', 'nsamp', 'c_dur', 'v_dur', 'w_dur']]

# set epoch temporal parameters. Include as much pre-stim time as possible,
# so later we can shift to align on C-V transition.
prev_trial_offset = brain_resp_dur - isi_range.min()
baseline = (prev_trial_offset, 0)
tmin_onset = min([prev_trial_offset, df['c_dur'].min() - df['c_dur'].max()])
tmax_onset = df['w_dur'].max() + brain_resp_dur
tmin_cv = 0 - df['c_dur'].max()
tmax_cv = df['v_dur'].max() + brain_resp_dur

# read Raws
basename = f'{subj_num:03}-{subj_code}-'
raw_fname = os.path.join(indir, basename + 'raw.fif.gz')
raw = mne.io.read_raw_fif(raw_fname, preload=True)
# raw.set_eeg_reference([])  # EEG ref. chan. set & dropped in prev. script
raw.info['bads'] = bad_chan[subj_code]

# make event dicts
events = mne.read_events(os.path.join(eegdir, 'events', basename + 'eve.txt'),
                         mask=None)
ev = events[:, -1]

# generate epochs aligned on stimulus onset (baselining needs to be done on
# onset-aligned epochs, even if we want to end up with CV-aligned epochs).
# we ignore annotations here, and instead reject epochs based on channel
# amplitude thresholds (set in the param file)
picks = mne.pick_types(raw.info, meg=False, eeg=True)
ev_dict = df.loc[np.in1d(df['event_id'], ev), ['key', 'event_id']]
ev_dict = ev_dict.set_index('key').to_dict()['event_id']
epochs_baseline = mne.Epochs(raw, events, ev_dict, tmin_onset, tmax_onset,
                             baseline, picks, reject=reject, preload=True,
                             reject_by_annotation=False, proj=True)
drops = np.where(epochs_baseline.drop_log)[0]

# generate events file aligned on consonant-vowel transition time.
# for each stim, shift event later by the duration of the consonant
orig_secs = np.squeeze([raw.times[samp] for samp in events[:, 0]])
df2 = df.set_index('event_id', inplace=False)
events[:, 0] = raw.time_as_index(orig_secs + df2.loc[ev, 'c_dur'])
ev_fname = os.path.join(eegdir, 'events', basename + 'cvalign-eve.txt')
mne.write_events(ev_fname, events)

# generate epochs aligned on CV-transition
epochs = mne.Epochs(raw, events, ev_dict, tmin_cv, tmax_cv,
                    baseline=None, picks=picks, reject=None,
                    preload=True, reject_by_annotation=False)
del raw  # conserve memory
# dropped epochs are determined by the baselined epochs object
epochs.drop(drops)
# zero out all data in the unbaselined, CV-aligned epochs object
# (it will get replaced with time-shifted baselined data)
epochs._data[:, :, :] = 0.
# select only the retained trials/epochs
df3 = df2.loc[ev, ['c_dur', 'v_dur', 'w_dur']].reset_index(drop=True)
df3 = df3.loc[epochs.selection].reset_index(drop=True)
# then for each trial, insert time-shifted baselined data
for ix, c_dur, v_dur, w_dur in df3.itertuples(index=True, name=None):
    # compute start/end samples for onset- and cv-aligned data
    st_onsetalign = np.searchsorted(epochs_baseline.times, 0.)
    nd_onsetalign = np.searchsorted(epochs_baseline.times,
                                    w_dur + brain_resp_dur)
    st_cvalign = np.searchsorted(epochs.times, 0. - c_dur)
    nd_cvalign = np.searchsorted(epochs.times, v_dur + brain_resp_dur)
    # handle duration difference of ±1 samp. (anything larger will error out)
    nsamp_onsetalign = nd_onsetalign - st_onsetalign
    nsamp_cvalign = nd_cvalign - st_cvalign
    if np.abs(nsamp_cvalign - nsamp_onsetalign) == 1:
        nd_cvalign = st_cvalign + nsamp_onsetalign
    # insert baselined data, shifted to have C-V alignment
    epochs._data[ix, :, st_cvalign:nd_cvalign] = \
        epochs_baseline._data[ix, :, st_onsetalign:nd_onsetalign]

# downsample onset-aligned epochs and save (need for SNR computations)
epochs_baseline = epochs_baseline.resample(100, npad=0, n_jobs='cuda')
epochs_baseline.save(os.path.join(eegdir, 'epochs', basename + 'epo.fif.gz'))
del epochs_baseline  # conserve memory

# optionally truncate after CV transition
for trunc_dur in trunc_durs:
    outdir = os.path.join(eegdir, f'epochs-{trunc_dur}')
    this_epochs = epochs.copy()
    if trunc_dur:
        tmin = float(trunc_dur) / 1000.  # milliseconds to seconds
        this_epochs = this_epochs.crop(tmin=tmin)
    # downsample
    this_epochs = this_epochs.resample(100, npad=0, n_jobs='cuda')
    # save epochs
    this_epochs.save(os.path.join(outdir, basename + 'cvalign-epo.fif.gz'))
    del this_epochs  # conserve memory

# output summary info
with open(os.path.join(eegdir, 'epoch-summary.csv'), 'a') as outfile:
    outfile.write(f'{subj_code},{len(epochs)}\n')

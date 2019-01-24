#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
Script 'make_block_length_stimuli.py'
===============================================================================

This script makes audio stimuli from folders of component WAV files.
"""
# @author: drmccloy
# Created on Mon Nov 30 13:41:39 2015
# License: BSD (3-clause)

import os
import yaml
import numpy as np
import pandas as pd
from glob import glob
from functools import partial
from itertools import combinations
from expyfun.io import read_wav, write_wav
from expyfun.stimuli import resample, get_tdt_rates
from pyglet.media import load as pygload

# flags
do_resample = True
write_wavs = True

# file paths
paramdir = 'params'
videodir = os.path.join('stimuli', 'videos')
stimroot = os.path.join('stimuli', 'stimuli-rms')
outdir = os.path.join('stimuli', 'stimuli-final')
testdirs = ('hin-m', 'hun-f', 'swh-m', 'nld-f')
engtestdirs = ('eng-m1', 'eng-f1')
traindirs = ('eng-m2', 'eng-f2')

# load params
with open(os.path.join(paramdir, 'params.yaml'), 'r') as f:
    params = yaml.load(f)
    isi_range = np.array(params['isi_range'])
# load IPA-ASCII dict
with open(os.path.join(paramdir, 'ipa-ascii.yaml'), 'r') as f:
    ipa_ascii = yaml.load(f)


def asciify(filepath, strip_digit=False):
    # convert unicode IPA filenames to ASCII (needed to reproduce sorting
    # behavior that occurred when stimuli were first made). Also recreate the
    # condition where only some stimuli had trailing digits in the filename.
    path, fname = os.path.split(filepath)
    syllable, rest_of_fname = fname.split('-')
    rest_of_fname = rest_of_fname[1:] if strip_digit else '-' + rest_of_fname
    ascii_fname = ipa_ascii[syllable] + rest_of_fname
    return os.path.join(path, ascii_fname)


# config
rand = np.random.RandomState(seed=0)
n_subj = 12
n_token_reps = 20
total_end_pad = 5.  # total silent secs at start/end of video (1/2 at each end)
fs_out = get_tdt_rates()['25k']
foreign_talker_combos = list(combinations(testdirs, 2))

# init containers
trainfiles = list()
testfiles = list()

# sort audio into training and test
for stimdir in traindirs + testdirs + engtestdirs:
    stimpath = os.path.join(stimroot, stimdir)
    wavfiles = sorted(glob(os.path.join(stimpath, '*.wav')))
    # impose ASCII-based sort order (that's what was originally used to make
    # the stimulus order; decision to retain non-ASCII filenames came later).
    # Also must reproduce the fact that, at stimulus generation time, non-
    # English stimuli had no "-0" after the syllable in their filenames (that
    # fix is also built into the asciify function).
    mapfun = (partial(asciify, strip_digit=True) if stimdir in testdirs else
              asciify)
    ascii_wavfiles = list(map(mapfun, wavfiles))
    wavfiles = np.array(wavfiles)[np.argsort(ascii_wavfiles)].tolist()
    # sort into train/test
    if stimdir in traindirs:  # include only 3 tokens (a few CVs have 4)
        wavfiles = [x for x in wavfiles if x[-5] in ('0', '1', '2')]
        trainfiles.extend(wavfiles)
    else:
        if stimdir in engtestdirs:  # include only 1 token
            wavfiles = [x for x in wavfiles if x[-5] == '0']
        testfiles.extend(wavfiles)
allfiles = trainfiles + testfiles
talkers = [x.split(os.sep)[-2] for x in allfiles]
syllables = [os.path.splitext(os.path.split(x)[1])[0] for x in allfiles]
train_mask = np.array([x in traindirs for x in talkers])

# read in wav data
wav_and_fs = [read_wav(x) for x in allfiles]
fs_in = [x[1] for x in wav_and_fs]
wavs = [x[0] for x in wav_and_fs]
nchan = np.array([x.shape[0] for x in wavs])
assert len(wavs) == len(allfiles)  # make sure they all loaded
assert len(set(fs_in)) == 1  # make sure sampling rate consistent
assert len(set(nchan)) == 1  # make sure all mono or all stereo
nchan = nchan[0]
fs = float(fs_in[0])
# resample to fs_out
if do_resample:
    wavs = [resample(x, fs_out, float(fs_in[0]), n_jobs='cuda') for x in wavs]
    fs = fs_out
# store wav data in one big array (shorter wavs zero-padded at end)
wav_nsamps = np.array([x.shape[-1] for x in wavs])
wav_array = np.zeros((len(wavs), nchan, wav_nsamps.max()))
for ix, (wav, dur) in enumerate(zip(wavs, wav_nsamps)):
    wav_array[ix, :, :dur] = wav
del wavs, wav_and_fs

# read in videos to get block durations
videonames = sorted(os.listdir(videodir))
videopaths = sorted(glob(os.path.join(videodir, '*.m4v')))
videodurs = np.array([pygload(vp).duration for vp in videopaths])

# initialize dataframe to store various params
df = pd.DataFrame()

for subj in range(n_subj):
    print(f'subj {subj:02}, block', end=' ')
    # init some vars
    all_blocks_syll_order = list()
    all_blocks_onset_samp = list()
    all_blocks_offset_samp = list()
    # set up output directory
    subjdir = f'subj-{subj:02}'
    if not os.path.isdir(os.path.join(outdir, subjdir)):
        os.makedirs(os.path.join(outdir, subjdir))
    # select which videos to show
    video_order = rand.permutation(len(videodurs))
    this_video_names = np.array(videonames)[video_order]
    this_video_durs = videodurs[video_order]
    # handle subject-specific language selection
    this_testdirs = foreign_talker_combos[subj % len(foreign_talker_combos)]
    this_test_talkers = this_testdirs + engtestdirs
    this_test_mask = np.array([x in this_test_talkers for x in talkers])
    # make sure there is zero overlap between training and test
    assert np.all(np.logical_not(np.logical_and(train_mask, this_test_mask)))
    # combine training and test sounds
    this_idx = np.r_[np.where(train_mask)[0], np.where(this_test_mask)[0]]
    this_nsamps = wav_nsamps[this_idx]
    # set stimulus order for whole experiment
    nsyll = this_idx.size
    order = np.ravel([rand.permutation(this_idx) for _ in range(n_token_reps)])
    # create ISIs (one more than we need, but makes zipping easier)
    isi_secs = np.linspace(*isi_range, num=len(order))
    isi_order = rand.permutation(len(isi_secs))
    isi_nsamps = np.round(fs * isi_secs).astype(int)[isi_order]
    # global onset/offset sample indices
    all_nsamps = wav_nsamps[order]
    syll_onset_samp = np.r_[0, np.cumsum(all_nsamps + isi_nsamps)][:-1]
    syll_offset_samp = np.cumsum(all_nsamps + np.r_[0, isi_nsamps[:-1]])
    assert np.array_equal(syll_offset_samp - syll_onset_samp, all_nsamps)
    # split into blocks
    all_blocks_max_nsamps = np.floor((this_video_durs - total_end_pad) *
                                     fs).astype(int)
    first_idx = 0
    for block_idx, this_block_max_nsamp in enumerate(all_blocks_max_nsamps):
        print(block_idx, end=' ')
        # calculate which syllables will fit in this block
        rel_offsets = syll_offset_samp - syll_onset_samp[first_idx]
        if rel_offsets.max() < this_block_max_nsamp:
            last_idx = nsyll * n_token_reps
        else:
            last_idx = np.where(rel_offsets > this_block_max_nsamp)[0].min()
        first_samp = syll_onset_samp[first_idx]
        last_samp = syll_offset_samp[last_idx - 1]
        nsamp = last_samp - first_samp
        assert this_block_max_nsamp >= nsamp
        this_block_wav = np.zeros((nchan, nsamp))
        this_block_syll_order = order[first_idx:last_idx]
        this_block_syll_onsets = (syll_onset_samp[first_idx:last_idx] -
                                  syll_onset_samp[first_idx])
        this_block_syll_offsets = (syll_offset_samp[first_idx:last_idx] -
                                   syll_onset_samp[first_idx])  # not a typo
        for onset, offset, idx in zip(this_block_syll_onsets,
                                      this_block_syll_offsets,
                                      this_block_syll_order):
            # patch syllable into block wav
            if write_wavs:
                this_block_wav[:, onset:offset] = \
                    wav_array[idx, :, :wav_nsamps[idx]]
            # append record to data frame
            is_training = talkers[idx] in traindirs
            record = dict(subj=subj, block=block_idx,
                          vid_dur=this_video_durs[block_idx],
                          vid_name=this_video_names[block_idx],
                          talker=talkers[idx], syll=syllables[idx],
                          train=is_training, onset=onset, offset=offset)
            df = df.append(record, ignore_index=True)
        # write out block wav
        if write_wavs:
            fname = f'block-{block_idx:02}.wav'
            write_wav(os.path.join(outdir, subjdir, fname), this_block_wav,
                      int(fs_out), overwrite=True)
        # iterate
        first_idx = last_idx
        if first_idx == nsyll * n_token_reps:
            break
    print()  # newline between subjs

# get just the talker (folder) and syllable (filename) part of each wav path
wav_ids = [os.path.split(x) for x in allfiles]
wav_ids = [os.path.join(os.path.split(x[0])[1], x[1]) for x in wav_ids]
# calculate trial-level params
df['wav_path'] = df['talker'] + '/' + df['syll'] + '.wav'  # NOT os.path.sep!
df['wav_idx'] = [wav_ids.index(x) for x in df['wav_path']]
df['wav_nsamp'] = [wav_nsamps[x] for x in df['wav_idx']]
df['onset_sec'] = df['onset'] / fs_out
df['offset_sec'] = df['offset'] / fs_out
# generate trial IDs and TTL stamps
df['wav_idx'] = [wav_ids.index(x) for x in df['wav_path']]
digits = np.ceil(np.log2(wav_array.shape[0])).astype(int)
df['ttl_id'] = df['wav_idx'].apply(np.binary_repr, width=digits)
df['ttl_id'] = df['ttl_id'].apply(lambda x: [int(y) for y in list(x)])
df['trial_id'] = (df['block'].astype(int).apply(format, args=('02',)) + '_' +
                  df['talker'] + '_' + df['syll'])
# save dataframe
column_order = ['subj', 'block', 'trial_id', 'ttl_id', 'talker', 'syll',
                'train', 'onset', 'offset', 'onset_sec', 'offset_sec',
                'wav_path', 'wav_idx', 'wav_nsamp', 'vid_name', 'vid_dur']
df.to_csv(os.path.join(paramdir, 'master-dataframe.tsv'), sep='\t',
          index=False, columns=column_order)
# save stimulus params
stimvars = dict(wav_array=wav_array, wav_nsamps=wav_nsamps, fs=fs_out,
                wav_names=wav_ids, isi_range=isi_range,
                pad=total_end_pad / 2.)
np.savez(os.path.join(paramdir, 'stim-params.npz'), **stimvars)

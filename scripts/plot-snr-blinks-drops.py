#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
Script 'plot-snr-blinks-drops.py'
===============================================================================

This script plots blinks, retained epochs, and SNR for each subject.
"""
# @author: drmccloy
# Created on Wed Jul 19 16:31:35 PDT 2017
# License: BSD (3-clause)

import os
import yaml
import mne
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# basic file I/O
paramdir = 'params'
eegdir = 'eeg-data-clean'
outdir = 'processed-data'
styledir = 'styles'

# style setup
sns.set_style('darkgrid')
plt.style.use(os.path.join(styledir, 'font-libertine.yaml'))

# load params
with open(os.path.join(paramdir, 'subjects.yaml'), 'r') as f:
    subjects = yaml.load(f)
rev_subjects = {subjects[s]: s for s in subjects}

with open(os.path.join(paramdir, 'params.yaml'), 'r') as f:
    params = yaml.load(f)
    brain_resp_dur = params['brain_resp_dur']
    blink_chan = params['blink_channel']
    bad_chan = params['bad_channels']
    ref_chan = params['ref_channel']
    reject = params['reject']
    n_jobs = params['n_jobs']

# load master dataframe
master_df = pd.read_csv(os.path.join(paramdir, 'master-dataframe.tsv'),
                        sep='\t')
master_df['subj'] = master_df['subj'].apply(int)
master_df['lang'] = master_df['talker'].str[:3]

# count total trials
n_trials = master_df.groupby('subj')['trial_id'].count()
n_trials.index = [rev_subjects[k + 1] for k in n_trials.index]
n_trials.index.name = 'subj'
n_trials.name = 'n_trials'

# count total English-talker trials
eng_df = master_df.loc[master_df['lang'] == 'eng']
n_eng_trials = eng_df.groupby('subj')['trial_id'].count()
n_eng_trials.index = [rev_subjects[k + 1] for k in n_eng_trials.index]
n_eng_trials.name = 'n_eng_trials'

# load SNR summary
snr = pd.read_csv(os.path.join(eegdir, 'snr-summary.csv'))
snr.set_index('subj', inplace=True)
# load retained epochs summary
ep = pd.read_csv(os.path.join(eegdir, 'epoch-summary.csv'))
ep = ep.set_index('subj')
# load retained english-talker epochs summary
n_eng_epochs = pd.read_csv(os.path.join(eegdir, 'eng-epoch-summary.csv'))
n_eng_epochs.set_index('subj', inplace=True)
# load blink summary
bl = pd.read_csv(os.path.join(eegdir, 'blinks', 'blink-summary.csv'))
bl = bl.set_index('subj')

# combine into one dataframe
df = pd.concat((snr, bl, ep, n_trials, n_eng_epochs, n_eng_trials), axis=1)
df.to_csv(os.path.join(outdir, 'blinks-epochs-snr.csv'))

# prettify column names for plotting
new_names = dict(n_blinks='Number of blinks detected',
                 n_epochs='Number of retained epochs',
                 snr='SNR: 10×$\log_{10}$(evoked power / baseline power)')
df.rename(columns=new_names, inplace=True)
df.index.name = 'Subject'

# plot
axs = df.iloc[:, :3].plot.bar(subplots=True, sharex=True, legend=False)
ax = axs[2]
xvals = ax.xaxis.get_ticklocs()
maxlines = ax.hlines(df['n_trials'], xvals - 0.4, xvals + 0.4,
                     colors='k', linestyles='dashed', linewidths=1)
ax.annotate(s='total trials', va='center',
            xy=(1, maxlines.get_segments()[-1][-1][-1]), xytext=(2, 0),
            xycoords=('axes fraction', 'data'), textcoords='offset points')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
for ax, ymax in zip(axs, [9, 2500, 5000]):
    ax.set_ylim([0, ymax])
fig = ax.get_figure()
fig.subplots_adjust(left=0.07, right=0.88, bottom=0.12, top=0.93, hspace=0.4)
# supplementary figure
fig.savefig(os.path.join('figures', 'supplement', 'subject-summary.pdf'))

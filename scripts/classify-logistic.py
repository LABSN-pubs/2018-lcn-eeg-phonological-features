#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
Script 'classify-logistic.py'
===============================================================================

This script runs EEG data through a classifier, and stores the classifier
object as well as its classifications and (pseudo-)probabilities.
"""
# @author: drmccloy
# Created on Thu Aug  3 16:08:47 PDT 2017
# License: BSD (3-clause)

import os
import sys
import yaml
from functools import partial
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import mne
from aux_functions import EER_score, EER_threshold, map_ipa_to_feature

rand = np.random.RandomState(seed=15485863)  # the one millionth prime


def train_split(data, mask, df, label_col):
    train_data = data[mask]
    train_labels = df.loc[mask, label_col].values
    return train_data, train_labels


def run_grid_search(train_data, train_labels):
    # hyperparameter setup
    param_grid = [dict(C=(2. ** np.arange(-10, 16)))]
    clf_kwargs = dict(tol=1e-4, solver='liblinear', random_state=rand)
    gridsearch_kwargs = dict(scoring=EER_score, n_jobs=n_jobs, refit=True,
                             pre_dispatch=pre_dispatch, cv=5, verbose=3)
    # run grid search
    classifier = LogisticRegression(**clf_kwargs)
    clf = GridSearchCV(classifier, param_grid=param_grid,
                       **gridsearch_kwargs)
    clf.fit(X=train_data, y=train_labels)
    return clf


def test_clf(clf, threshold, data, mask, df, lang, prob_colnames,
             pred_colname):
    test_data = data[mask]
    # convert probabilities to DataFrame
    probs = clf.predict_proba(test_data)
    ipa = df.loc[mask, 'ipa']
    df_out = pd.DataFrame(probs, index=ipa)
    # rename columns from 0,1 to something sensible
    df_out.columns = prob_colnames
    df_out['lang'] = lang
    # add in binary predictions
    df_out[pred_colname] = (probs[:, 1] >= threshold).astype(int)
    return df_out


subj_code = sys.argv[1]      # IJ, IQ, etc
subj_num = int(sys.argv[2])  # integer
trunc_dur = int(sys.argv[3])

# load params
paramdir = 'params'
with open(os.path.join(paramdir, 'subjects.yaml'), 'r') as f:
    subjects = yaml.load(f)
with open(os.path.join(paramdir, 'params.yaml'), 'r') as f:
    params = yaml.load(f)
    n_comp = params['n_dss_comp']
    n_jobs = params['n_jobs']
pre_dispatch = '2*n_jobs'

# load features
features_file = os.path.join(paramdir, 'feature-tables', 'all-features.tsv')
with open(features_file, 'r') as f:
   for line in f:
       if line.startswith('#'):
           continue
       else:
           features = line.strip().split('\t')
           break

# other file I/O
eegdir = 'eeg-data-clean'
indir = os.path.join(eegdir, f'td-redux-dss-{trunc_dur}')
outdir = os.path.join('processed-data', f'logistic-{trunc_dur}',
                      'classifiers')
subj_outdir = os.path.join(outdir, subj_code)
basename = f'{subj_num:03}-{subj_code}-cvalign-'
datafile = basename + f'redux-dss{n_comp}-data.npy'

# load the data
efile = os.path.join(eegdir, f'epochs-{trunc_dur}', basename + 'epo.fif.gz')
epochs = mne.read_epochs(efile, verbose=False)
data = np.load(os.path.join(indir, datafile))
event_ids = epochs.events[:, -1]

# load the trial params
df_cols = ['subj', 'block', 'talker', 'syll', 'train', 'wav_idx']
df_types = dict(subj=int, block=int, talker=str, syll=str, train=bool,
                wav_idx=int)
master_df = pd.read_csv(os.path.join('params', 'master-dataframe.tsv'),
                        sep='\t', usecols=df_cols, dtype=df_types)
# naively map syllable to just the onset consonant
master_df['ipa'] = master_df['syll'].map(lambda x: x.split('-')[0][:-1])
# now load the english-IPA mapping, and undo the strict phonetic coding of
# English phones done during stimulus annotation.
phoneme_mapping = os.path.join('params', 'eng-phones-to-phonemes.yaml')
with open(phoneme_mapping, 'r') as f:
    eng_phone_to_phoneme = yaml.load(f)
master_df['lang'] = master_df['talker'].str[:3]
eng = (master_df['lang'] == 'eng')
master_df.loc[eng, 'ipa'] = master_df.loc[eng, 'ipa'].map(eng_phone_to_phoneme)

# reduce to just this subj (NB: df['subj'] is 0-indexed, subj. dict is not)
master_df = master_df.loc[master_df['subj'] == (subj_num - 1)]

# remove dropped epochs (trials). The `while` loop is skipped for most subjects
# but should handle cases where the run was stopped and restarted, by cutting
# out trials from the middle of `df` until
# `df['wav_idx'].iloc[epochs.selection]` yields the stim IDs in `event_ids`
match = master_df['wav_idx'].iloc[epochs.selection].values == event_ids
unmatched = np.logical_not(np.all(match))
# i = 0
while unmatched:
    # print(f'iteration {i}; {match.sum()} / {len(match)} matches')
    first_bad_sel = np.where(np.logical_not(match))[0].min()
    first_bad = epochs.selection[first_bad_sel]
    mismatched_wavs = master_df['wav_idx'].iloc[first_bad:]
    mismatched_evs = event_ids[first_bad_sel:]
    new_start = np.where(mismatched_wavs == mismatched_evs[0])[0][0]
    master_df = pd.concat((master_df.iloc[:first_bad],
                           master_df.iloc[(first_bad + new_start):]))
    match = master_df['wav_idx'].iloc[epochs.selection].values == event_ids
    unmatched = np.logical_not(np.all(match))
    # i += 1
master_df = master_df.iloc[epochs.selection, :]
assert np.array_equal(master_df['wav_idx'].values, event_ids)


# # # # # # # # # # # # # # # # #
# FEATURE-BASED CLASSIFICATION  #
# # # # # # # # # # # # # # # # #

# initialize container
classifiers = dict()

for this_feature in features:
    this_df = master_df.copy()
    outfile_suffix = f'cvalign-redux-dss{n_comp}-{this_feature}'
    # merge in consonant features
    func = partial(map_ipa_to_feature, feature=this_feature,
                   features_file=features_file)
    this_df[this_feature] = this_df['ipa'].map(func)

    # make the data classifier-friendly
    train_data, train_labels = train_split(data=data, mask=this_df['train'],
                                           df=this_df, label_col=this_feature)
    # handle sparse feature sets (that have NaN cells)
    valued = np.isfinite(train_labels)
    train_labels = train_labels[valued].astype(int)
    train_data = train_data[valued]
    # run gridsearch
    clf = run_grid_search(train_data, train_labels)
    classifiers[f'{subj_code}-{this_feature}'] = clf

    # compute EER threshold (refits using best params from grid search object)
    threshold, eer = EER_threshold(clf, X=train_data, y=train_labels,
                                   return_eer=True)
    eer_fname = os.path.join(subj_outdir, 'eer-thresholds.csv')
    with open(eer_fname, 'a') as f:
        f.write(f'{subj_code},{this_feature},{threshold},{eer}\n')

    # test on new English talkers & foreign talkers
    for lang in set(this_df['lang']):
        mask = np.logical_and((this_df['lang'] == lang),
                              np.logical_not(this_df['train']))
        prob_colnames = [f'-{this_feature}', f'+{this_feature}']
        df_out = test_clf(clf, threshold, data, mask, this_df, lang,
                          prob_colnames, pred_colname=this_feature)
        # save
        fname = f'classifier-probabilities-{lang}-{outfile_suffix}.csv'
        df_out.to_csv(os.path.join(subj_outdir, fname))

# save classifier objects
np.savez(os.path.join(subj_outdir, 'classifiers.npz'), **classifiers)

# # # # # # # # # # # # # # # # #
# PAIRWISE / OVR CLASSIFICATION #
# # # # # # # # # # # # # # # # #

# don't run pairwise / OVR classification on truncated epochs
if trunc_dur == 0:
    # extract just the English consonants
    eng_phones = master_df.loc[eng, 'ipa'].unique().tolist()

    # OVR CLASSIFICATION

    # change output directory to "ovr"
    outdir = os.path.join('processed-data', 'ovr', 'classifiers')
    subj_outdir = os.path.join(outdir, subj_code)
    classifiers = dict()
    # loop over phones
    for this_phone in eng_phones:
        this_df = master_df.copy()
        outfile_suffix = f'cvalign-redux-dss{n_comp}-ovr_{this_phone}'
        # run gridsearch
        this_df['label'] = (this_df['ipa'] == this_phone)
        train_data, train_labels = train_split(data, mask=this_df['train'],
                                               df=this_df, label_col='label')
        clf = run_grid_search(train_data, train_labels)
        classifiers[f'{subj_code}-ovr_{this_phone}'] = clf
        # compute EER threshold
        # (refits using best params from grid search object)
        threshold, eer = EER_threshold(clf, X=train_data, y=train_labels,
                                       return_eer=True)
        eer_fname = os.path.join(subj_outdir, 'eer-thresholds.csv')
        with open(eer_fname, 'a') as f:
            f.write(f'{subj_code},{this_phone},{threshold},{eer}\n')
        # test on new English talkers & foreign talkers
        for lang in set(this_df['lang']):
            mask = np.logical_and((this_df['lang'] == lang),
                                  np.logical_not(this_df['train']))
            prob_colnames = [f'not_{this_phone}', this_phone]
            df_out = test_clf(clf, threshold, data, mask, this_df, lang,
                              prob_colnames, pred_colname=f'pred_{this_phone}')
            # save
            fname = f'classifier-probabilities-{lang}-{outfile_suffix}.csv'
            df_out.to_csv(os.path.join(subj_outdir, fname))
    # save classifier objects
    np.savez(os.path.join(subj_outdir, 'classifiers.npz'), **classifiers)

    # PAIRWISE CLASSIFICATION

    # change output directory to "pairwise"
    outdir = os.path.join('processed-data', 'pairwise', 'classifiers')
    subj_outdir = os.path.join(outdir, subj_code)
    classifiers = dict()
    # loop over pairs of phones
    while len(eng_phones) > 1:
        this_df = master_df.copy()
        phone_one = eng_phones.pop()
        this_df['label'] = (this_df['ipa'] == phone_one)
        for phone_two in eng_phones:
            this_contrast = f'{phone_one}_vs_{phone_two}'
            outfile_suffix = (f'cvalign-redux-dss{n_comp}-'
                              f'pairwise_{this_contrast}')
            # run gridsearch
            these_phones = np.in1d(this_df['ipa'], [phone_one, phone_two])
            mask = np.logical_and(this_df['train'], these_phones)
            train_data, train_labels = train_split(data, mask, df=this_df,
                                                   label_col='label')
            clf = run_grid_search(train_data, train_labels)
            classifiers[f'{subj_code}-pairwise_{this_contrast}'] = clf
            # compute EER threshold
            # (refits using best params from grid search object)
            threshold, eer = EER_threshold(clf, X=train_data, y=train_labels,
                                           return_eer=True)
            eer_fname = os.path.join(subj_outdir, 'eer-thresholds.csv')
            with open(eer_fname, 'a') as f:
                f.write(f'{subj_code},{this_contrast},{threshold},{eer}\n')
            # test on new English talkers & foreign talkers
            for lang in set(this_df['lang']):
                mask = np.logical_and((this_df['lang'] == lang),
                                      np.logical_not(this_df['train']))
                prob_colnames = [phone_two, phone_one]
                df_out = test_clf(clf, threshold, data, mask, this_df, lang,
                                  prob_colnames, pred_colname=this_contrast)
                # save
                fname = f'classifier-probabilities-{lang}-{outfile_suffix}.csv'
                df_out.to_csv(os.path.join(subj_outdir, fname))
    # save classifier objects
    np.savez(os.path.join(subj_outdir, 'classifiers.npz'), **classifiers)

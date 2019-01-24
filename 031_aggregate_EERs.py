#!/usr/bin/env python3

import os
import yaml
import numpy as np
import pandas as pd

# file I/O
paramdir = 'params'

# load subjects
with open(os.path.join(paramdir, 'subjects.yaml'), 'r') as f:
    subjects = yaml.load(f)

# load features
features_file = os.path.join(paramdir, 'feature-tables', 'all-features.tsv')
feature_table = pd.read_csv(features_file, sep='\t', index_col=0, comment='#')
features = feature_table.columns.tolist()

# load trunc durs
trunc_file = os.path.join('scripts', 'set-trunc-durs.sh')
with open(trunc_file, 'r') as f:
   for line in f:
       if line.startswith('trunc_durs'):
           trunc_durs = line.strip().split('=')[1].strip('() ').split()
           trunc_durs = list(map(int, trunc_durs))
           break

trunc_analyses = {f'logistic-{td}': 'feature' for td in trunc_durs}
analyses = dict(ovr='consonant', pairwise='contrast')
analyses.update(trunc_analyses)

# aggregate for OVR / pairwise classifications
for analysis, colname in analyses.items():
    this_df = pd.DataFrame()
    for subj_code in subjects:
        this_file = os.path.join('processed-data', analysis, 'classifiers',
                                 subj_code, 'eer-thresholds.csv')
        this_subj_df = pd.read_csv(this_file)
        this_df = pd.concat([this_df, this_subj_df], axis=0, ignore_index=True)
    this_df.reset_index(inplace=True, drop=True)
    # aggregate EERs / thresholds separately
    for val in ('eer', 'threshold'):
        path = os.path.join('processed-data', analysis, f'{val}s.csv')
        df = this_df.pivot(index=colname, columns='subj', values=val)
        df.columns.name = None
        df.to_csv(path)

# fixup sparse results for pairwise
for val in ('eer', 'threshold'):
    path = os.path.join('processed-data', 'pairwise', f'{val}s.csv')
    df = pd.read_csv(path, index_col='contrast')
    for subj in df.columns:
        for pair in df.index:
            if np.isnan(df.at[pair, subj]):
                altpair = '_'.join(pair.split('_')[::-1])
                df.at[pair, subj] = df.at[altpair, subj]
    df.to_csv(path)

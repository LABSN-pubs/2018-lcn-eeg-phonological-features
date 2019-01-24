#!/usr/bin/env bash

source scripts/set-trunc-durs.sh

for trunc in ${trunc_durs[@]}; do
    mkdir -p eeg-data-clean/dss-$trunc
    mkdir -p eeg-data-clean/td-redux-dss-$trunc
done

# loop over subjects
while read subject; do
    numb=$(echo $subject | cut -d " " -f 2 -)
    subj=$(echo $subject | cut -d " " -f 1 -)
    subj=${subj%:}
    python3 scripts/compute-dss.py $subj $numb ${trunc_durs[@]}
    python3 scripts/time-domain-redux.py $subj $numb ${trunc_durs[@]}
done <params/subjects.yaml

# plot DSS component power for each subject
mkdir -p figures
python3 scripts/plot-dss-power.py

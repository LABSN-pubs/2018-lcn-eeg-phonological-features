#!/usr/bin/env bash

source scripts/set-trunc-durs.sh

# make a place to save the non-CV-aligned epochs (need for SNR computation)
mkdir -p eeg-data-clean/epochs
# ...and a place to save the cv-aligned data with different truncations
for trunc in ${trunc_durs[@]}; do
    mkdir -p eeg-data-clean/epochs-$trunc
done

# initialize summary file
echo "subj,n_epochs" > eeg-data-clean/epoch-summary.csv

# loop over subjects
while read subject; do
    numb=$(echo $subject | cut -d " " -f 2 -)
    subj=$(echo $subject | cut -d " " -f 1 -)
    subj=${subj%:}
    python3 scripts/make-epochs.py $subj $numb ${trunc_durs[@]}
done <params/subjects.yaml

#!/usr/bin/env bash

mkdir -p processed-data
mkdir -p figures/supplement

# initialize summary files
echo "subj,snr" > eeg-data-clean/snr-summary.csv
echo "subj,n_eng_epochs" > eeg-data-clean/eng-epoch-summary.csv

# loop over subjects
while read subject; do
    numb=$(echo $subject | cut -d " " -f 2 -)
    subj=$(echo $subject | cut -d " " -f 1 -)
    subj=${subj%:}
    python3 scripts/compute-snr.py $subj $numb
done <params/subjects.yaml

# plot per-subject summary of SNR, blinks, and retained epochs
python3 scripts/plot-snr-blinks-drops.py

#!/usr/bin/env bash

# initialize output directory
projdir=eeg-data-clean/raws-with-projectors
bldir=eeg-data-clean/blinks
mkdir -p $projdir $bldir

# initialize summary file
echo "subj,n_blinks" > $bldir/blink-summary.csv

# loop over subjects
while read subject; do
    numb=$(echo $subject | cut -d " " -f 2 -)
    subj=$(echo $subject | cut -d " " -f 1 -)
    subj=${subj%:}
    python3 scripts/add-blink-projectors.py $subj $numb
done <params/subjects.yaml

#!/usr/bin/env bash

# initialize output directories
rootdir="eeg-data-clean"
mkdir -p $rootdir/raws $rootdir/events

# loop over subjects
while read subject; do
    numb=$(echo $subject | cut -d " " -f 2 -)
    subj=$(echo $subject | cut -d " " -f 1 -)
    subj=${subj%:}
    python3 scripts/merge-eeg-raws.py $subj $numb
done < params/subjects.yaml

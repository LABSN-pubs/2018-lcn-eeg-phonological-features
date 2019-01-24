#!/usr/bin/env bash

# initialize output directory
anndir=eeg-data-clean/raws-annotated
mkdir -p $anndir

# loop over subjects
while read subject; do
    numb=$(echo $subject | cut -d " " -f 2 -)
    subj=$(echo $subject | cut -d " " -f 1 -)
    subj=${subj%:}
    annfile=$anndir/$(printf "%03d" $numb)-$subj-raw.fif.gz
    if [ -f $annfile ]; then
        echo "File for $subj already exists in annotation folder. Your choices:"
        select reann in \
        "Reannotate that file & overwrite when done" \
        "Open that file for viewing; don't save when done" \
        "Annotate original RAW & overwrite existing annotated file" \
        "Skip this subject"; do
            case $REPLY in
            1) python scripts/annotate-eeg-interactive.py $subj $numb 1 1 ;;
            2) python scripts/annotate-eeg-interactive.py $subj $numb 1 0 ;;
            3) python scripts/annotate-eeg-interactive.py $subj $numb 0 1 ;;
            4) echo "skipping $subj"; break ;;
            *) echo "Invalid choice. Enter 1-4 (or Ctrl+C to abort)." ;;
            esac
        done </dev/tty
    else
        python scripts/annotate-eeg-interactive.py $subj $numb 0 1
    fi
done <params/subjects.yaml

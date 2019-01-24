#!/bin/sh

source scripts/set-trunc-durs.sh

# load the features (header line of all-features file) into an array called
# $features, after skipping comment lines. NOTE: this is no longer needed
# because we don't want to loop over features in Bash; instead we do it within
# python (so the epochs object is only loaded once per subject).
# while read line; do
#     case "$line" in \#*) continue ;; esac
#     read -a features <<< "$line"
#     break
# done < params/feature-tables/all-features.tsv

# loop over subjects
while read subject; do
    numb=$(echo $subject | cut -d " " -f 2 -)
    subj=$(echo $subject | cut -d " " -f 1 -)
    subj=${subj%:}
    # pairwise/OVR classification not done on truncated data, so outside loop
    pwdir="processed-data/pairwise/classifiers/$subj"
    ovrdir="processed-data/ovr/classifiers/$subj"
    mkdir -p $pwdir $ovrdir
	echo "subj,contrast,threshold,eer" > $pwdir/eer-thresholds.csv
	echo "subj,consonant,threshold,eer" > $ovrdir/eer-thresholds.csv
    # loop over truncation durations
    for trunc in ${trunc_durs[@]}; do
        outdir="processed-data/logistic-$trunc/classifiers/$subj"
        mkdir -p $outdir
		echo "subj,feature,threshold,eer" > $outdir/eer-thresholds.csv
        # run the "workhorse" script
        python3 scripts/classify-logistic.py $subj $numb $trunc
    done
done < params/subjects.yaml

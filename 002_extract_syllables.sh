#!/usr/bin/env bash

# clean up old versions
for SUBDIR in highpassed stimuli-extracted stimuli-rms; do
    rm -r stimuli/$SUBDIR
done

# apply highpass filter to remove low-frequency drifts in the raw recordings
find stimuli/recordings/*/*.wav -exec python scripts/highpass.py {} \;

# extract individual syllables as WAV files (this loop is 1 → many)
for RECORDING in $(find stimuli/highpassed/*/*.wav); do
    # replace folder "highpassed" with "textgrids", then change file extension
    # ("tmp" is workaround because can't nest ${} parameter expansions)
    TEXTGRID=$(TMP=${RECORDING/highpassed/textgrids};echo "${TMP%wav}TextGrid")
    OUTDIR=$(dirname ${RECORDING/highpassed/stimuli-extracted})
    # not all raw recordings are annotated (some had no usable tokens)
    if [ -f $TEXTGRID ]; then
        mkdir -p $OUTDIR
        praat --run scripts/extract-labeled-intervals.praat \
            ../$RECORDING ../$TEXTGRID ../$OUTDIR 1
    fi
done

# RMS normalize the extracted WAVs
for WAVFILE in $(find stimuli/stimuli-extracted/*/*.wav); do
    OUTFILE=${WAVFILE/extracted/rms}
    OUTDIR=$(dirname $OUTFILE)
    mkdir -p $OUTDIR
    (cd scripts; python rms-normalize.py ../$WAVFILE ../$OUTFILE 0.01)
done

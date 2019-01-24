# Comparing phonological feature systems using EEG data

This repository contains the materials for the scientific paper

> McCloy, DR & Lee, AKC. Investigating the fit between phonological feature
systems and brain responses to speech using EEG. *Language, Cognition and
Neuroscience*. http://dx.doi.org/10.1080/23273798.2019.1569246

Raw EEG data related to this project is hosted on the Open Science Framework:
https://osf.io/kxqtc/files/

To reproduce the analyses here, clone this repository, download the `.tar.gz`
files that are in the `eeg-data-raw` folder on
[the OSF project webpage](https://osf.io/kxqtc/files/), put them in the folder
of the same name that is part of this repository, and run
`000_uncompress_data.sh` to unpack the data.  Then follow the instructions below.

### IMPORTANT NOTE

The original development repository for this project is at
https://github.com/drammock/eeg-phone-coding. As with many projects, it grew
organically and consequently that repo is not especially orderly. Refactoring
and cleanup is underway, and the permanent home of the code will be here.
This message will be updated accordingly when that process is complete.

# pipeline overview

The steps to reproduce the analysis and manuscript are documented below.
Scripts that must be run interactively (e.g., for manual marking of
consonant-vowel transitions in the stimuli, or for annotating bad sections of
EEG data) include `_interactive` in the filename.

## General setup
- `000_compress_data.sh` and `000_uncompress_data.sh` compress and uncompress
  the raw EEG data and metadata.
- `001_setup_expyfun.py` downloads a particular historical version of `expyfun`
  into a local directory, to ensure that the stimulus generation and experiment
  runner script are always run with the version of `expyfun` against which it
  they were developed and tested.

## Stimulus generation

**NOTE** The results of all of the manual/interactive steps of stimulus
generation are included in the repository, so it is possible to skip this
section entirely to reproduce the analysis. These scripts are included as a
record of the procedure, and as a starting point for future stimulus generation
efforts that start from different recordings.

- The first step of stimulus generation involved annotating the raw recordings
  with `.TextGrid` files using [praat](https://www.praat.org), to mark which
  syllables should be extracted for use as stimuli. Subsequent steps assume
  that recordings are in `stimuli/recordings`, associated TextGrids are in
  `recordings/textgrids`, with WAV/TextGrid correspondences determined by
  parallel subfolders/filenames within each.
- `002_extract_syllables.sh` applies a high-pass filter to the raw recordings
  (to ameliorate very low frequency noise in the recording), extracts the
  annotated syllables into individual WAV files, and root-mean-square
  normalizes the extracted syllables.
- `003_mark_syllable_boundaries_interactive.praat` opens each syllable in praat
  for annotation of the consonant-vowel transition point. This should be run
  from an open instance of the praat GUI (`Praat → Open Praat script...`;
  `Run → Run`). The result is a new set of textgrids in `stimuli/stimuli-tg`.
- `004_make_syllable_boundary_table.praat` parses the syllable textgrids and
  writes them to a table (`params/syllable-boundary-times.tsv`). This can be
  run through the praat GUI, or via
  `praat --run 004_make_syllable_boundary_table.praat stimuli/stimuli-tg params/cv-boundary-times.tsv`.
  ***WARNING*** Before running this step, make sure praat’s text writing
  setting is set to UTF-8 (menu command Praat → Preferences → Text writing
  preferences) or else subsequent scripts will not be able to read the CV
  boundary times table.
- `005_make_block_length_stimuli.py` assembles the syllables into stimulus
  blocks (with different randomizations for each subject). Each block is
  written out as a single WAV file whose duration is matched to the video
  stimulus used in that block. The record of subject/block/video/syllable
  correspondences is written to `params/master-dataframe.tsv`. Note that the
  block-length WAV files are for reference only; when the experiment is
  actually run, individual syllable WAVs are loaded and played individually.

## Data collection

- `006_run_experiment.py` runs the experiment. It expects to connect to a TDT
  external sound processor (and thus must run on a Windows computer), and sends
  synchronization timestamps to both the TDT and the EEG acquisition device,
  via TTL.  The raw EEG data were saved continuously on a separate computer,
  using “BrainVision Recorder” software.

## EEG preprocessing

- `011_merge_eeg_raws.sh` is a Bash wrapper around `scripts/merge-eeg-raws.py`,
  which converts the Brainvision `.vhdr`, `.vmrk`, and `.eeg` files into `.fif`
  files, and handles cases of mid-experiment recording failure where multiple
  raw recordings were generated and partial blocks had to be repeated.
- `012_annotate_eeg_interactive.sh` This is another Bash wrapper, around
  `scripts/annotate-eeg-interactive.py`.  The wrapper allows skipping subjects
  and/or re-opening files that have already been annotated (e.g., if
  corrections are necessary).
- `013_remove_blinks.sh` is a Bash wrapper around
  `scripts/add-blink-projectors.py`, which in addition to a new raw file with
  projectors added, also creates epoch and event objects for each subject’s
  blinks, and a summary CSV of blink detection across all subjects.
- `014_make_epochs.sh` is a Bash wrapper around `scripts/make-epochs.py`. This
  creates 3 versions of epochs objects for each subject: one with traditional
  stimulus-onset-alignment (just in case), one with epochs aligned at the
  consonant-vowel transition point (“CV alignment”), and one with CV alignment
  and also the early part of all epochs truncated (to exclude the brain’s
  early response to the stimuli, thought to be dominated by acoustic-phonetic
  representations, so that only the later (hopefully phonological)
  representations are all that is left).  This also generates a summary file
  telling how many epochs were retained for each subject (to make sure the
  epoch rejection criteria are not too severe).
- `015_check_snr.sh` wraps `scripts/compute-snr.py` and
  `scripts/plot-snr-blinks-drops.py`.  It generates a summary figure
  (`figures/supplement/subject-summary.pdf`) and table
  (`processed-data/blinks-epochs-snr.csv`) useful for checking for a systematic
  relationship between number of retained epochs and a subject’s blink
  behavior and data SNR.
- `020_reduce_dimensionality.sh`

## Supervised learning
- `030_run_classification.sh` creates the necessary output directories and
  loops over subjects and truncation durations, running
  `scripts/classify-logistic.py` each time. The output goes into 
  `processed-data/{analysis_type}/classifiers/{subject_ID}`. Note that
  `scripts/classify-logistic.py` is set up to run phonological-feature-based
  classification for *each value* of the trucation duration defined in
  `scripts/set-trunc-durs.sh`, but will only run OVR and pairwise
  classification when truncation duration is `0` (i.e., on the untruncated
  epochs).
- `031_aggregate_EERs.py` aggregates the Equal Error Rate and threshold values
  from each subject into a single table, in `processed-data/{analysis_type}`.

## TODO: rest of analysis pipeline

TODO

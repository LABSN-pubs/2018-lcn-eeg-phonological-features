#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
highpass.py

This script applies a highpass filter to a WAV file.
"""
# @author: drmccloy
# Created on Tue Oct 30 15:18:32 PDT 2018
# License: BSD (3-clause)

import os
import sys
import scipy.signal as ss
from expyfun.stimuli import read_wav, write_wav

infile = sys.argv[1]
cutoff = 50.

# generate output file path
outfile = infile.replace('recordings', 'highpassed')
outdir = os.path.split(outfile)[0]
os.makedirs(outdir, exist_ok=True)

# read in wavfile
wav, fs = read_wav(infile)
mono = wav[0] if wav.ndim > 1 else wav  # convert to mono
b, a = ss.butter(4, cutoff / (fs/2.), btype='high')
hp = ss.lfilter(b, a, mono)
write_wav(outfile, hp, fs, overwrite=True)

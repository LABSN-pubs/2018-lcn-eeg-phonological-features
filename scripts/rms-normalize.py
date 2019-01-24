#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rms-normalize.py

This script normalizes root-mean-square amplitude of WAV files. There is no
built-in protection against clipping.
"""
# @author: drmccloy
# Created on Wed Oct 31 12:48:41 PDT 2018
# License: BSD (3-clause)

import sys
from expyfun.stimuli import read_wav, write_wav, rms

infile = sys.argv[1]
outfile = sys.argv[2]
rms_out = float(sys.argv[3])

wav, fs = read_wav(infile)
rms_in = rms(wav)
wavout = wav * rms_out / rms_in
write_wav(outfile, wavout, fs, overwrite=True)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
Script 'setup_expyfun.py'
===============================================================================

This script makes sure the right version of expyfun is available to run the
experiment
"""
# @author: drmccloy
# Created on Wed Nov  7 12:40:35 PST 2018
# License: BSD (3-clause)

import expyfun

# this version string must match what is asserted in the run_experiment script
expyfun.download_version('0ee0951')

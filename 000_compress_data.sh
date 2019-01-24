#!/usr/bin/env bash

cd eeg-data-raw
find -name "*.eeg" -exec tar -cz {} -f {}.tar.gz \;
ls *.v[hm][dr][rk] | tar -cz -T - -f header-and-marker-files.tar.gz
cd ..

#!/usr/bin/env bash

cd eeg-data-raw
find -name "*.eeg.tar.gz" -exec tar -xzf {} \;
tar -xzf header-and-marker-files.tar.gz
#rm *.tar.gz
cd ..

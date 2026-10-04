#!/bin/bash

mkdir audio

find . -name "audio-*.tar.gz" -exec tar xvzf {} \;

find audio-* -name "*.wav" -exec mv {} audio/. \;

rmdir audio-*/

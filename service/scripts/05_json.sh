#!/bin/bash

. ./scripts/config.sh

VIDEO=${1}

python ./jsonResults.py \
    -o ./uploads/${VIDEO}.json \
    -e ./processes/${VIDEO}/emotion.txt \
    -g ./processes/${VIDEO}/genderLabels.txt \
    -a ./processes/${VIDEO}/ageLabels.txt \
    -L ./processes/${VIDEO}/oface_landmarks.txt \
    -A ./processes/${VIDEO}/oface_AUs.txt

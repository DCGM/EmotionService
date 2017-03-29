#!/bin/bash

. ./scripts/config.sh

VIDEO=${1}
$SCRIPT_PATH/processVideo2.py \
    --landmark-file ./processes/${VIDEO}/processVideo_landmarks.txt \
    --out-dir ./processes/${VIDEO}/processVideo \
    --input-video ./uploads/${VIDEO} \
    --landmark-model ${DATA_PATH}/shape_predictor_68_face_landmarks.dat \
    --crop-resolution 252 --crop-scale 1.8 --downscale-factor 0



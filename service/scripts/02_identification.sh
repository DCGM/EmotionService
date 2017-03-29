#!/bin/bash

. ./scripts/config.sh

VIDEO=$1

python ${SCRIPT_PATH}/clusterFaces.py \
    --net '{"net":"'$DATA_PATH'/vgg_face_caffe/VGG_FACE.caffemodel", "deploy":"'$DATA_PATH'/vgg_face_caffe/VGG_FACE_deploy.prototxt", "cropScaleNew":"1.6", "mean":"105"}' \
    --crop-scale 1.8 --per-track 6  \
    --input-list ./processes/${VIDEO}/processVideo_landmarks.txt \
    --image-dir ./processes/${VIDEO}/processVideo \
    -o ./processes/${VIDEO}/ID \
    -t 0.73 \
    --layer-extract fc7

$SCRIPT_PATH/mapIdentities.py \
    -i ./processes/${VIDEO}/processVideo_landmarks.txt \
    -m ./processes/${VIDEO}/ID.mapping >./processes/${VIDEO}/dlib_landmarks.txt


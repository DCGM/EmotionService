#!/bin/bash

. ./scripts/config.sh

VIDEO=${1}

python $SCRIPT_PATH/classifySoftmax.py \
    --class-file ./processes/${VIDEO}/genderLabels  \
    --landmark-file ./processes/${VIDEO}/processVideo_landmarks.txt \
    --in-dir ./processes/${VIDEO}/processVideo \
    --net $DATA_PATH/Nets/genderDEX.caffemodel \
    --deploy $DATA_PATH/genderDEX.prototxt \
    --mean 127 -bs 32 -so 1.8 -sn 1.6 --skip 3

python $SCRIPT_PATH/classifySoftmax.py \
    --class-file ./processes/${VIDEO}/ageLabels \
    --landmark-file ./processes/${VIDEO}/processVideo_landmarks.txt \
    --in-dir ./processes/${VIDEO}/processVideo \
    --net $DATA_PATH/Nets/ageDEX.caffemodel \
    --deploy  $DATA_PATH/ageDEX.prototxt \
    --mean 127 -bs 32 -so 1.8001 -sn 1.8 --skip 3

python $SCRIPT_PATH/classifySoftmax.py \
    --class-file ./processes/${VIDEO}/emotion \
    --landmark-file ./processes/${VIDEO}/processVideo_landmarks.txt \
    --in-dir ./processes/${VIDEO}/processVideo \
    --net $DATA_PATH/Nets/emotionsAllNew100k.caffemodel \
    --deploy  $DATA_PATH/emotions.prototxt \
    --mean 127 -bs 32 -s 127 -so 1.8 -sn 1.5

python $SCRIPT_PATH/countAge.py \
    --class-file ./processes/${VIDEO}/ageLabels.txt \
    --output-file ./processes/${VIDEO}/ageLabels.txt

for i in ageLabels emotion genderLabels
do
    $SCRIPT_PATH/mapIdentities.py \
        -i ./processes/${VIDEO}/${i}.txt \
        -m ./processes/${VIDEO}/ID.mapping >./processes/${VIDEO}/${i}.tmp
    mv ./processes/${VIDEO}/${i}.tmp ./processes/${VIDEO}/${i}.txt
done

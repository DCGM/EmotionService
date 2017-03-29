#!/bin/bash

. ./scripts/config.sh

VIDEO=$1

IDS=$(cut -f 2 -d \  ./processes/${VIDEO}/ID.mapping | sort | uniq)
for ID in ${IDS}
do
    grep -P "\t$ID\t" ./processes/${VIDEO}/dlib_landmarks.txt | cut -f 1,3 | sed 's/:/ /g' | gawk '{print $1, $2, $3, $4-$2, $5-$3}'  >${VIDEO}.bb.tmp
    lines=$(cat ${VIDEO}.bb.tmp | wc -l)
    echo LINES $VIDEO $ID "'$lines'"
    if (( 10 < lines ))
    then
        echo WORKING ON $VIDEO $ID
        $O_FACE -q -bb ${VIDEO}.bb.tmp -f ./uploads/${VIDEO} -of ./processes/${VIDEO}/id_${ID}_oface
        sed "s/\t0\t/\t${ID}\t/" ./processes/${VIDEO}/id_${ID}_oface_landmarks.txt >>./processes/${VIDEO}/oface_landmarks.txt

        sed "s/ / $ID /" ./processes/${VIDEO}/id_${ID}_oface_AUs.txt >> ./processes/${VIDEO}/oface_AUs.txt
    fi
done
cat ./processes/${VIDEO}/oface_AUs.txt | sort -n -k 1 > ./processes/${VIDEO}/oface_AUs.tmp
mv ./processes/${VIDEO}/oface_AUs.tmp ./processes/${VIDEO}/oface_AUs.txt

sort -n -k 1,2 ./processes/${VIDEO}/oface_landmarks.txt >./processes/${VIDEO}/oface_landmarks.tmp
mv ./processes/${VIDEO}/oface_landmarks.tmp ./processes/${VIDEO}/oface_landmarks.txt

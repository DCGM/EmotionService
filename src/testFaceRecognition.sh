DATA_DIR=/home/ihradis/projects/2016-08-31_face_service/MixedEmotions/data
IMAGE_LIST=CACD.10k.ann
IMAGE_ROOT=./CACD/
ORIGIN_CROP_SCALE=2.0

for CLASSIFY_CROP_SCALE in 1.6 #1.2 1.4 1.6 1.8 2.0 3 4
do
# NET=$DATA_DIR/nn4.small2.v1.t7
#
# python extractOpenFaceFeatures.py -m "$NET" \
#                           -i "$IMAGE_ROOT" \
#                           -l "$IMAGE_LIST" \
#                           -o featuresOpenFace.npy \
#                           --crop-scale-origin $ORIGIN_CROP_SCALE \
#                           --crop-scale-new $CLASSIFY_CROP_SCALE


NET=$DATA_DIR/vgg_face_caffe/VGG_FACE.caffemodel
NET_DEPLOY=$DATA_DIR/vgg_face_caffe/VGG_FACE_deploy.prototxt
IMAGE_LIST=CACD.10k.ann

python extractFeatures.py --net "$NET" --deploy "$NET_DEPLOY" \
                          -i "$IMAGE_ROOT" \
                          --extract_layer fc7 \
                          -l "$IMAGE_LIST" \
                          -o features.npy \
                          --mean 105 \
                          --crop-scale-origin $ORIGIN_CROP_SCALE \
                          --crop-scale-new $CLASSIFY_CROP_SCALE

    #./testFaceRecognition.py -a CACD.10k.ann -f features.npy -i 10
 results=$(./testFaceRecognition.py -a CACD.10k.ann -f features.npy -i 10 | tail -n 1)
 echo $CLASSIFY_CROP_SCALE $results >> TEST_FACE_RECOGNITION.results
 echo $CLASSIFY_CROP_SCALE $results

 done

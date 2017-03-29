DATA_DIR=/home/ihradis/projects/2016-08-31_face_service/MixedEmotions/data

# python classifyStream.py -i out.avi --landmark-model $DATA_DIR/shape_predictor_68_face_landmarks.dat \
#                          --crop-scale 2.0 \
#                          --crop-resolution 320 \
#                          --emoticons-dir $DATA_DIR/Emoticons \
#                       	 --render-bbox \
# 			 --render-landmarks \
# 		         --emotions-net '{"net":"'$DATA_DIR'/Nets/emotions.caffemodel", "deploy":"'$DATA_DIR'/emotions.prototxt", "cropScaleNew":"1.6", "mean":"127", "scale":"127", "RGBGray":"True"}' \
#                          --gender-net '{"net":"'$DATA_DIR'/Nets/genderDEX.caffemodel", "deploy":"'$DATA_DIR'/genderDEX.prototxt", "mean":"127", "cropScaleNew":"1.6"}' \
#                          --age-net '{"net":"'$DATA_DIR'/Nets/ageDEX.caffemodel", "deploy":"'$DATA_DIR'/ageDEX.prototxt", "mean":"127", "cropScaleNew":"1.8"}'


python classifyStream.py --landmark-model $DATA_DIR/shape_predictor_68_face_landmarks.dat \
        --crop-scale 2.0 \
        --crop-resolution 256 \
        --emoticons-dir $DATA_DIR/Emoticons \
        --render-bbox \
        --render-landmarks \
        --identification-features features.npy \
        --identification-directory ../data/CACD/ \
        --identification-layer fc7 \
        --identification-annotation CACD.10k.ann \
        --age-net '{"net":"'$DATA_DIR'/Nets/ageDEX.caffemodel", "deploy":"'$DATA_DIR'/ageDEX.prototxt", "mean":"127", "cropScaleNew":"1.8"}' \
        --gender-net '{"net":"'$DATA_DIR'/Nets/genderDEX.caffemodel", "deploy":"'$DATA_DIR'/genderDEX.prototxt", "mean":"127", "cropScaleNew":"1.6"}' \
        --emotions-net '{"net":"'$DATA_DIR'/Nets/emotionsAllNew100k.caffemodel", "deploy":"'$DATA_DIR'/emotions.prototxt", "cropScaleNew":"1.5", "mean":"127", "scale":"127", "RGBGray":"True"}' \
        --identification-net '{"net":"'$DATA_DIR'/vgg_face_caffe/VGG_FACE.caffemodel", "deploy":"'$DATA_DIR'/vgg_face_caffe/VGG_FACE_deploy.prototxt", "cropScaleNew":"1.6", "mean":"105"}'


#--emotions-net '{"net":"'$DATA_DIR'/Nets/emotions.caffemodel", "deploy":"'$DATA_DIR'/emotions.prototxt", "cropScaleNew":"1.6", "mean":"127", "scale":"127", "RGBGray":"True"}' \
#         --gender-net '{"net":"'$DATA_DIR'/Nets/genderDEX.caffemodel", "deploy":"'$DATA_DIR'/genderDEX.prototxt", "mean":"127", "cropScaleNew":"1.6"}' \
#         --age-net '{"net":"'$DATA_DIR'/Nets/ageDEX.caffemodel", "deploy":"'$DATA_DIR'/ageDEX.prototxt", "mean":"127", "cropScaleNew":"1.8"}'

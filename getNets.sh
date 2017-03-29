wget http://www.robots.ox.ac.uk/~vgg/software/vgg_face/src/vgg_face_caffe.tar.gz
tar xzf vgg_face_caffe.tar.gz -C ./data
rm vgg_face_caffe.tar.gz

mkdir -p -- "./data/Nets"
wget -O ./data/Nets/genderDEX.caffemodel https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/gender.caffemodel
wget -O ./data/Nets/ageDEX.caffemodel https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/dex_imdb_wiki.caffemodel
wget -O ./data/Nets/emotionsAllNew100k.caffemodel https://www.dropbox.com/s/3dyw18e5y6m6p3i/emotionsAllNew100k.caffemodel?dl=1

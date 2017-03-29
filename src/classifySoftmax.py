#!/usr/bin/env python
import numpy as np
import caffe
import cv2
import os
import sys
import argparse
from tools import natural_sort

def parse_args():
    print( ' '.join(sys.argv))

    parser = argparse.ArgumentParser(epilog="Classify Softmax")


    parser.add_argument('--skip',
                        default=0, type=int,
                        help='Caffe net model.')
    parser.add_argument('-n', '--net',
                        required=True,
                        help='Caffe net model.')
    parser.add_argument('-d', '--deploy',
                        required=True,
                        help='Deploy for caffe net.')
    parser.add_argument('-bs', '--batch-size',
                        type=int)
    parser.add_argument('-i', '--in-dir',
                        required=True,
                        help='Input directory with images for classification.')
    parser.add_argument('-cf', '--class-file',
                        required=True,
                        help='Output file for predictions.')
    parser.add_argument('-lf', '--landmark-file',
                        required=True,
                        help='Landmark file.')
    parser.add_argument('-m', '--mean',
                        type=float,
                        help='Substract mean from images.')
    parser.add_argument('-s', '--scale',
                        type=float,
                        help='Scale images.')
    parser.add_argument('-so', '--crop-scale-origin',
                        type=float,
                        help='Origin crop scale of images.')
    parser.add_argument('-sn', '--crop-scale-new',
                        type=float,
                        help='New crop scale of images.')
    parser.add_argument('-rg', '--RGB-gray',
                        action='store_true',
                        help='Convert loaded images to gray')
    parser.add_argument('-c', '--cpu',
                        action='store_true',
                        help='Set cpu mode for classification (default gpu mode)')

    args = parser.parse_args()

    if (args.crop_scale_origin is None and args.crop_scale_new is not None) or (args.crop_scale_origin is not None and args.crop_scale_new is None):
        parser.error("Either both or none of the arguments crop_scale_origin and crop_scale_new can be used.")

    if (args.crop_scale_origin is not None and args.class_file is not None):
        if args.crop_scale_origin <= args.crop_scale_new:
            parser.error("Argument crop_scale_new must be smaller or equal to crop_scale_origin.")

    return args


class Net(object):

    def __init__(self, netModel, deploy, mean, scale, cropScaleOrigin, cropScaleNew, RGBGray):
        self.net = None
        self.netModel = netModel
        self.deploy = deploy
        self.mean = mean
        self.scale = scale
        self.cropScaleOrigin = cropScaleOrigin
        self.cropScaleNew = cropScaleNew
        self.RGBGray = RGBGray

        self.netImageShape = None
        self.netGrayColor = None

    def loadNet(self, batchSize=None):
        self.net = caffe.Net(self.deploy, self.netModel, caffe.TEST)
        if batchSize != self.net.blobs['data'].data.shape[0] and batchSize is not None:
            self.net.blobs['data'].reshape(batchSize, self.net.blobs['data'].data.shape[1], self.net.blobs['data'].data.shape[2], self.net.blobs['data'].data.shape[3])
            self.net.reshape()
        self.netImageShape = self.net.blobs['data'].data.shape[2]
        if self.net.blobs['data'].data.shape[1] == 1:
            self.netGrayColor = True
        else:
            self.netGrayColor = False

    def classifyImages(self, images):

        for index in range(0, len(images)):

            image = images[index]

            if len(image.shape) == 2:
                grayColor = True
            else:
                grayColor = False

            if grayColor != self.netGrayColor:
                if self.netGrayColor:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                else:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

            if self.RGBGray and self.netGrayColor is False:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

            if ((self.cropScaleNew is not None) and (self.cropScaleNew != self.cropScaleOrigin)):
                newImageShape = (image.shape[0] / self.cropScaleOrigin) * self.cropScaleNew
                leftTop = int((image.shape[0] - newImageShape) / 2)
                rightBottom = int(image.shape[0] - (image.shape[0] - newImageShape) / 2)
                image = image[leftTop:rightBottom, leftTop:rightBottom]

            if self.netImageShape != image.shape[0]:
                image = cv2.resize(image, (self.netImageShape, self.netImageShape), interpolation=cv2.INTER_AREA)

            if self.mean is not None:
                image = image - self.mean

            if self.scale is not None:
                image = image / self.scale

            if self.netGrayColor is False:
                image = np.rollaxis(image, 2, 0)

            self.net.blobs['data'].data[index] = image

        out = self.net.forward()

        return out


def main():
    args = parse_args()

    if args.cpu:
        print("CPU mode set")
        caffe.set_mode_cpu()
    else:
        print("GPU mode set")
        caffe.set_mode_gpu()

    net = Net(
        args.net, args.deploy, args.mean, args.scale,
        args.crop_scale_origin, args.crop_scale_new, args.RGB_gray)
    net.loadNet(batchSize=args.batch_size)

    if os.path.isfile(os.path.join(os.getcwd(), args.class_file + '.txt')):
        os.remove(os.path.join(os.getcwd(), args.class_file + '.txt'))
    f = open(os.path.join(os.getcwd(), args.class_file + '.txt'), 'w+')

    with open(args.landmark_file) as lFile:
        landmarks = lFile.read().splitlines()
        if args.skip:
            landmarks = landmarks[::args.skip + 1]

    batchSize = net.net.blobs['data'].data.shape[0]

    counter = 0

    color = 1
    if net.net.blobs['data'].data.shape[1] == 1 or args.RGB_gray:
        color = 0

    print("Classifying images...")

    while (counter < len(landmarks)):

        if not ((len(landmarks) - counter) >= batchSize):
            sizeOfRange = len(landmarks) - counter
        else:
            sizeOfRange = batchSize

        images = []

        for index in range(0, sizeOfRange):
            words = landmarks[index + counter].split()
            imageName = 'face_{:06d}_{:04d}.jpg'.format(
                int(words[0]), int(words[1]))

            imgPath = os.path.join(args.in_dir, imageName)
            images.append(cv2.imread(imgPath, color))

        out = net.classifyImages(images)

        for index in range(0, sizeOfRange):
            labels = landmarks[index + counter].split()
            f.write("{} {} {}\n".format(labels[0], labels[1],' '.join(map(str, out['prob'][index]))))
            if (index + counter) != 0:
                if (index + counter) % 100 == 0:
                    print('{} images classified'.format(index + counter))

        counter += sizeOfRange

    f.close()

if __name__ == "__main__":
    main()

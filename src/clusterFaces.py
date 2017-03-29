#!/usr/bin/env python

from __future__ import print_function

import time
import argparse
import cv2
import os
from operator import itemgetter
from collections import defaultdict

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
import numpy as np

import caffe
from classifySoftmax import Net
from classifyStream import createNet


def parseArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument('-n', '--net',
                        required=True,
                        help='Caffe net model.')
    parser.add_argument('--per-track',
                        default=8,
                        type=int,
                        help='Images used per face track.')
    parser.add_argument('--crop-scale',
                        default=1,
                        type=float,
                        help='Face scale factor of the input facial crops.')
    parser.add_argument('-i', '--input-list',
                        required=True,
                        help='File produced by processVideo.py')
    parser.add_argument(
        '-c', '--cpu', action='store_true',
        help='Set cpu mode for classification (default gpu mode)')
    parser.add_argument(
        '--image-dir', required=True,
        help='Directory with facial crops produced by processVideo.py.')
    parser.add_argument(
        '-o', '--output-file', required=True,
        help='Output file name.')
    parser.add_argument(
        '--layer-extract', required=True,
        help='Network layer which will be used for identification.')

    parser.add_argument('-t', '--threshold',
                        required=False, default=0.7, type=float,
                        help='Threshold controlls size of clusters.')

    return parser.parse_args()


def compactDistanceMatrix(distances, identities):
    uniqeIdentities = np.unique(identities)
    outputDistances = []
    for i, identity in enumerate(uniqeIdentities):
        outputDistances.append(
            distances[:, identities == identity].min(axis=1))

    distances = np.vstack(outputDistances)
    outputDistances = []
    for i, identity in enumerate(uniqeIdentities):
        outputDistances.append(
            distances[:, identities == identity].min(axis=1))
    outputDistances = np.vstack(outputDistances)
    return outputDistances, uniqeIdentities


def makeCollage(images):
    side = int((len(images)) ** 0.5 + 1)
    res = images[0].shape
    collage = np.zeros(
        (side * res[0], side * res[1], res[2]), dtype=images[0].dtype)

    for pos, image in enumerate(images):
        p0 = (pos / side) * res[0]
        p1 = (pos % side) * res[1]
        print(pos, p0, p1)
        collage[p0:p0 + res[0], p1:p1 + res[1], :] = image

    return collage


def readDetection(input_list):
    faces = defaultdict(list)
    with open(input_list, 'r') as f:
        for line in f:
            words = line.split()
            face = [int(x) for x in words[0:2]]
            faces[face[1]].append(face)
    return faces


def main():
    args = parseArgs()

    # read detections
    faces = readDetection(args.input_list)

    if args.cpu:
        print("CPU mode set")
        caffe.set_mode_cpu()
    else:
        print("GPU mode set")
        caffe.set_mode_gpu()

    net = createNet(args.net, args.crop_scale)
    net.loadNet(batchSize=args.per_track)
    dim = net.net.blobs[args.layer_extract].data.reshape(args.per_track, -1).shape[1]
    trackFeatures = np.zeros((len(faces), dim)).astype(np.float32)
    allImages = []

    # create representations for face tracks
    for key in faces:
        face_id = int(key)
        print('Working on', key)
        cropList = faces[key]
        step = max(1, int(len(cropList) / args.per_track + 0.5))
        cropList = cropList[step / 2::step]

        images = []
        for cropFrame, cropID in cropList:
            cropName = 'face_{:06d}_{:04}.jpg'.format(cropFrame, key)
            cropName = os.path.join(args.image_dir, cropName)
            img = cv2.imread(cropName)
            images.append(img)
        images = images[0:args.per_track]
        for i in range(args.per_track - len(images)):
            images.append(images[0])
        net.classifyImages(images)
        trackFeatures[face_id, :] = np.mean(
            net.net.blobs[args.layer_extract].data.reshape(args.per_track, -1),
            axis=0)
        allImages.append(images[args.per_track / 2])

    print('Have features. ', trackFeatures.shape)
    distances = pdist(trackFeatures, 'cosine')
    Z = linkage(distances, method='complete')
    print('Max distance: ', trackFeatures.max())
    clusterCount = 2
    clusters = fcluster(Z, t=args.threshold, criterion='distance') #clusterCount, criterion='maxclust')

    with open(args.output_file + '.mapping', 'w') as outF:
        for i in range(len(clusters)):
            print(i, clusters[i], file=outF)

    for ID in range(np.unique(clusters).size):
        try:
            idx = np.nonzero(clusters == ID + 1)[0].tolist()
            collage = [allImages[i] for i in idx]

            collage = makeCollage(collage)

            cv2.imwrite(
                '{}_cluster_{:04d}.jpg'.format(args.output_file,ID),
                collage)
            #cv2.imshow('cluster {}'.format(ID), collage)
            #cv2.waitKey(1000)
        except:
            pass



if __name__ == '__main__':
    main()

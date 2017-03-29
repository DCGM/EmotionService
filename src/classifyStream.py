#!/usr/bin/env python
from __future__ import print_function
import sys
import numpy as np
import caffe
import multiprocessing

import os
import json

from processVideo2 import faceDetector, multipleFaceTracker, faceAligner
from classifySoftmax import Net
from renderVideo import renderEmotionsLabelToFrame, renderGenderLabelToFrame, renderAgeLabelToFrame, renderBoxToFrame, renderLandmarksToFrame

from testFaceRecognition import readAnnotation
from sklearn.metrics import pairwise_distances

import zmqnpy
import zmq


def parse_args():
    print(' '.join(sys.argv))
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input-video',
                        required=False,
                        help='Path to input video.')

    parser.add_argument('--landmark-model',
                        help='Path to shape predictidentification-featuresor model.')
    parser.add_argument('--downscale-factor',
                        default=3,
                        type=int,
                        help='Downslace factor of the image.')
    parser.add_argument('--align-version',
                        default=1,
                        type=int,
                        help='Version of alignment.')
    parser.add_argument('--crop-resolution',
                        default=128,
                        type=int,
                        help='Resolution of cropped faces.')
    parser.add_argument('--crop-scale',
                        default=1,
                        type=float,
                        help='Scale factor of the facial crop box.')
    parser.add_argument('--skip-frames',
                        default=0, type=int,
                        help='How many frames to skip between two processed frames.')
    parser.add_argument('-v', '--verbose',
                        action='store_true')

    parser.add_argument('-b', '--render-bbox',
                        action='store_true')
    parser.add_argument('-l', '--render-landmarks',
                        action='store_true')

    parser.add_argument('-en', '--emotions-net',
                        type=str,
                        help='Caffe emotions net model.')
    parser.add_argument('-gn', '--gender-net',
                        type=str,
                        help='Caffe gender net model.')
    parser.add_argument('-an', '--age-net',
                        type=str,
                        help='Caffe age net model.')

    parser.add_argument('-c', '--cpu',
                        action='store_true',
                        help='Set cpu mode for classification (default gpu mode)')
    parser.add_argument('-er', '--emoticons-dir',
                        help='Directory with images of emoticons for each emotion (angry.png, disgust.png, fear.png, happy.png, sad.png, surprise.png, neutral.png, male.png, female.png)')

    parser.add_argument('--identification-net',
                        type=str,
                        help='Caffe person identification network definition.')
    parser.add_argument('--identification-features',
                        type=str,
                        help='Computed features for the images.')
    parser.add_argument('--identification-annotation',
                        type=str,
                        help='File with image names.')
    parser.add_argument('--identification-directory',
                        type=str,
                        help='Directory with images.')
    parser.add_argument('--identification-layer',
                        type=str,
                        help='Layer which is used for person identification.')

    args = parser.parse_args()

    if args.identification_net:
        for attr in ['identification_features', 'identification_annotation',
                     'identification_directory', 'identification_layer']:
            if not hasattr(args, attr) or not getattr(args, attr):
                print('If param --identification-net is specified, you have to specifie --{} as well.'.format(attr.replace('_', '-')))
                exit(-1)

    return args


def createNet(jsonString, cropScale):
    stringProperties = json.loads(jsonString)
    netProperties = {}
    propertiesNames = ['net', 'deploy', 'mean', 'scale', 'cropScaleNew', 'RGBGray']

    for propertyName in propertiesNames:
        try:
            netProperties[propertyName] = stringProperties[propertyName]
            if propertyName == 'net' or propertyName == 'deploy':
                netProperties[propertyName] = str(netProperties[propertyName])
            elif propertyName == 'RGBGray':
                netProperties[propertyName] = bool(netProperties[propertyName])
            else:
                netProperties[propertyName] = float(netProperties[propertyName])
        except:
            if propertyName == 'RGBGray':
                netProperties[propertyName] = False
            else:
                netProperties[propertyName] = None

    return Net(netProperties['net'], netProperties['deploy'], netProperties['mean'], netProperties['scale'], cropScale, netProperties['cropScaleNew'], netProperties['RGBGray'])


class NetProcess(multiprocessing.Process):
    def __init__(self, jsonString, cropScale, inQueue, outQueue):
        multiprocessing.Process.__init__(self)
        self.daemon = True
        self.jsonString = jsonString
        self.cropScale = cropScale
        self.inQueue = inQueue
        self.outQueue = outQueue

    def run(self):
            self.inQueue.start()
            self.outQueue.start()
            self.net = createNet(self.jsonString, self.cropScale)
            self.net.loadNet(1)
            while True:
                try:
                    img = self.inQueue.getNewest()
                    out = self.net.classifyImages([img])
                    out = out['prob'][0]
                    self.outQueue.put(out)
                except:
                    pass


class PersonIdentification(multiprocessing.Process):
    def __init__(self, netConfigString, featureFile, annotationFile, imageDirectory, inputCropScale, layer, queue=None):
        multiprocessing.Process.__init__(self)
        self.daemon = True
        self.netConfigString = netConfigString
        self.featureFile = featureFile
        self.annotationFile = annotationFile
        self.imageDirectory = imageDirectory
        self.inputCropScale = inputCropScale
        self.layer = layer
        self.inQueue = queue
        self.features = np.load(featureFile)
        self.annotation = readAnnotation(self.annotationFile)
        self.identities = [''] * self.features.shape[0]
        self.files = [''] * self.features.shape[0]
        for person in self.annotation:
            for rec in self.annotation[person]:
                self.identities[rec['line']] = person
                self.files[rec['line']] = rec['file']
        self.reset()

    def run(self):
        self.inQueue.start()
        self.net = createNet(self.netConfigString, self.inputCropScale)
        self.net.loadNet(1)
        import cv2
        while True:
            img = self.inQueue.getNewest()
            self.identify(img)
            self.deprecate()
            cv2.waitKey(10)

    def identify(self, img):
        import cv2
        out = self.net.classifyImages([img])
        fingerprint = self.net.net.blobs[self.layer].data[0].reshape(-1)
        distances = pairwise_distances(fingerprint.reshape(1, -1), self.features, metric='cosine')
        bestID = np.argmin(distances)
        matchImg = cv2.imread(os.path.join(self.imageDirectory, self.files[bestID]))
        print('Best match', self.files[bestID], self.identities[bestID])

        sortedMatches = np.argsort(distances).reshape(-1)
        for score, position in enumerate(sortedMatches):
            score = 1.0 / (score + 4)
            self.scores[self.identities[position]] += score

        names = self.scores.keys()
        scores = [0 - self.scores[x] for x in names]
        bestMatches = np.argsort(scores).reshape(-1)

        toShow = 6
        images = []
        for match in bestMatches[0:toShow]:
            name = names[match]
            fileName = self.annotation[name][0]['file']
            img = cv2.imread(os.path.join(self.imageDirectory, fileName), 1)
            images.append(img)

        collage = np.concatenate(images, axis=1)
        cv2.imshow('Best matches collage', collage)
        cv2.imshow('Best match', matchImg)

    def reset(self):
        self.scores = {}
        for person in self.annotation:
            self.scores[person] = 0

    def deprecate(self):
        for person in self.scores:
            self.scores[person] *= 0.8


def main():

    args = parse_args()

    if args.cpu:
        print("CPU mode set")
        caffe.set_mode_cpu()
    else:
        print("GPU mode set")
        caffe.set_mode_gpu()

    cropQueue = zmqnpy.plug(url='ipc://facecrop', socket_type=zmq.PUB,
                            bind=True, hwm=2)
    emotionQueue = zmqnpy.plug(url='ipc://emotion', socket_type=zmq.SUB,
                               bind=False, blockRead=False)
    genderQueue = zmqnpy.plug(url='ipc://gender', socket_type=zmq.SUB,
                              bind=False, blockRead=False)
    ageQueue = zmqnpy.plug(url='ipc://age', socket_type=zmq.SUB,
                           bind=False, blockRead=False)
    emotionQueue.start()
    genderQueue.start()
    ageQueue.start()

    if args.emotions_net is not None:
        emotionsNet = NetProcess(
            args.emotions_net, args.crop_scale,
            zmqnpy.plug(url='ipc://facecrop', socket_type=zmq.SUB,
                        bind=False, blockRead=True),
            zmqnpy.plug(url='ipc://emotion', socket_type=zmq.PUB, bind=True))
        emotionsNet.start()

    if args.gender_net is not None:
        genderNet = NetProcess(
            args.gender_net, args.crop_scale,
            zmqnpy.plug(url='ipc://facecrop', socket_type=zmq.SUB,
                        bind=False, blockRead=True),
            zmqnpy.plug(url='ipc://gender', socket_type=zmq.PUB, bind=True))
        genderNet.start()

    if args.age_net is not None:
        ageNet = NetProcess(
            args.age_net, args.crop_scale,
            zmqnpy.plug(url='ipc://facecrop', socket_type=zmq.SUB,
                        bind=False, blockRead=True),
            zmqnpy.plug(url='ipc://age', socket_type=zmq.PUB, bind=True))
        ageNet.start()


    if args.identification_net:
        personIdentifier = PersonIdentification(
            netConfigString=args.identification_net,
            featureFile=args.identification_features,
            annotationFile=args.identification_annotation,
            imageDirectory=args.identification_directory,
            inputCropScale=args.crop_scale,
            layer=args.identification_layer,
            queue=zmqnpy.plug(url='ipc://facecrop', socket_type=zmq.SUB,
                              bind=False, blockRead=True))
        personIdentifier.start()

    frameId = 0

    detector = faceDetector(detector_downscale=args.downscale_factor)
    tracker = multipleFaceTracker(landmark_model=args.landmark_model, detector=detector, verbose=args.verbose)

    import cv2
    cv2.namedWindow("classifyStream", cv2.WINDOW_AUTOSIZE)
    if args.input_video:
        cap = cv2.VideoCapture(args.input_video)
    else:
        cap = cv2.VideoCapture(0)

    aligner = faceAligner(
        faceResolution=np.array((int(args.crop_resolution / args.crop_scale), int(args.crop_resolution / args.crop_scale))),
        cropResolution=np.array((args.crop_resolution, args.crop_resolution)))

    emotionsLabel = None
    genderLabel = None
    ageLabel = None
    cropQueue.start()

    while(True):
        ret, frame = cap.read()
        if not ret:
            break
        frameId += 1
        if frameId % (args.skip_frames + 1) != 0:
            continue
        faces = tracker.processFrame(frame)

        maxBboxSize = 0
        faceToRender = None
        for k, face in faces:
            tmpBboxSize = abs(face.rect.top() - face.rect.bottom())
            if (tmpBboxSize > maxBboxSize):
                maxBboxSize = tmpBboxSize
                faceToRender = face

        if faceToRender is not None:
            H = aligner.getTranformation(faceToRender)
            crop = aligner.transform(frame, H)
            cropQueue.put(crop)
            print('PUT')
            # img = cropQueueTest.get()
            # if img is not None:
            #     print('GOT IMAGE')


            if args.render_bbox:
                frame = renderBoxToFrame(frame, [(faceToRender.rect.left(), faceToRender.rect.top()), (faceToRender.rect.right(), faceToRender.rect.bottom())])

            if args.render_landmarks:
                frame = renderLandmarksToFrame(frame, [(point.x, point.y) for point in faceToRender.parts()])

            emotionsLabel = emotionQueue.getNewest(emotionsLabel)
            if emotionsLabel is not None:
                frame = renderEmotionsLabelToFrame(frame, emotionsLabel, args.emoticons_dir)

            genderLabel = genderQueue.getNewest(genderLabel)
            if genderLabel is not None:
                frame = renderGenderLabelToFrame(frame, genderLabel, args.emoticons_dir)

            ageLabel = ageQueue.getNewest(ageLabel)
            if ageLabel is not None:
                meanAge = 0
                valSum = 0
                for i, age in enumerate(ageLabel):
                    meanAge += i * (age**1.5)
                    valSum += (age**1.5)
                meanAge /= valSum
                ageLabel = np.append(ageLabel, meanAge)
                frame = renderAgeLabelToFrame(frame, ageLabel)


        # if args.identification_net:
        #     personIdentifier.deprecate()
        cv2.imshow('classifyStream', cv2.resize(frame, (0,0), fx=2, fy=2))
        key = cv2.waitKey(1)
        key = chr(key & 255)
        if key == ' ':
            # personIdentifier.reset()
            pass
        if key == chr(27):
            break

    cap.release()


if __name__ == "__main__":
    main()

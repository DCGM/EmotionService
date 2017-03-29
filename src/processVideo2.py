#!/usr/bin/env python
from __future__ import print_function

import sys
import numpy as np

import cv2
from time import time
import os
import dlib

INNER_EYES_AND_BOTTOM_LIP = [39, 42, 57, 58]
OUTER_EYES_AND_NOSE = [36, 45, 33, 58]

TEMPLATE = np.float32([
    (0.0792396913815, 0.339223741112), (0.0829219487236, 0.456955367943),
    (0.0967927109165, 0.575648016728), (0.122141515615, 0.691921601066),
    (0.168687863544, 0.800341263616), (0.239789390707, 0.895732504778),
    (0.325662452515, 0.977068762493), (0.422318282013, 1.04329000149),
    (0.531777802068, 1.06080371126), (0.641296298053, 1.03981924107),
    (0.738105872266, 0.972268833998), (0.824444363295, 0.889624082279),
    (0.894792677532, 0.792494155836), (0.939395486253, 0.681546643421),
    (0.96111933829, 0.562238253072), (0.970579841181, 0.441758925744),
    (0.971193274221, 0.322118743967), (0.163846223133, 0.249151738053),
    (0.21780354657, 0.204255863861), (0.291299351124, 0.192367318323),
    (0.367460241458, 0.203582210627), (0.4392945113, 0.233135599851),
    (0.586445962425, 0.228141644834), (0.660152671635, 0.195923841854),
    (0.737466449096, 0.182360984545), (0.813236546239, 0.192828009114),
    (0.8707571886, 0.235293377042), (0.51534533827, 0.31863546193),
    (0.516221448289, 0.396200446263), (0.517118861835, 0.473797687758),
    (0.51816430343, 0.553157797772), (0.433701156035, 0.604054457668),
    (0.475501237769, 0.62076344024), (0.520712933176, 0.634268222208),
    (0.565874114041, 0.618796581487), (0.607054002672, 0.60157671656),
    (0.252418718401, 0.331052263829), (0.298663015648, 0.302646354002),
    (0.355749724218, 0.303020650651), (0.403718978315, 0.33867711083),
    (0.352507175597, 0.349987615384), (0.296791759886, 0.350478978225),
    (0.631326076346, 0.334136672344), (0.679073381078, 0.29645404267),
    (0.73597236153, 0.294721285802), (0.782865376271, 0.321305281656),
    (0.740312274764, 0.341849376713), (0.68499850091, 0.343734332172),
    (0.353167761422, 0.746189164237), (0.414587777921, 0.719053835073),
    (0.477677654595, 0.706835892494), (0.522732900812, 0.717092275768),
    (0.569832064287, 0.705414478982), (0.635195811927, 0.71565572516),
    (0.69951672331, 0.739419187253), (0.639447159575, 0.805236879972),
    (0.576410514055, 0.835436670169), (0.525398405766, 0.841706377792),
    (0.47641545769, 0.837505914975), (0.41379548902, 0.810045601727),
    (0.380084785646, 0.749979603086), (0.477955996282, 0.74513234612),
    (0.523389793327, 0.748924302636), (0.571057789237, 0.74332894691),
    (0.672409137852, 0.744177032192), (0.572539621444, 0.776609286626),
    (0.5240106503, 0.783370783245), (0.477561227414, 0.778476346951)])

TPL_MIN, TPL_MAX = np.min(TEMPLATE, axis=0), np.max(TEMPLATE, axis=0)
MINMAX_TEMPLATE = (TEMPLATE - TPL_MIN) / (TPL_MAX - TPL_MIN)


def parse_args():
    print(' '.join(sys.argv))
    import argparse
    parser = argparse.ArgumentParser(
        epilog="Output file format: frame_id\\tface_id\\tbbox_left:bbox_top:bbox_right:bbox_bottom\\tlandmark1_x:landmark1_y\\tlandmark2_x:landmark2_y\\t...landmarkn_x:landmarkn_y")

    parser.add_argument('--input-video',
                        default='./dev_1.mp4',
                        help='Input video.')
    parser.add_argument('--landmark-model',
                        default='/home/alena/work/openface/models/dlib/shape_predictor_68_face_landmarks.dat',
                        help='Path to shape predictor model.')
    parser.add_argument('--downscale-factor',
                        default=4,
                        type=float,
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
    parser.add_argument('-o', '--out-dir',
                        required=True,
                        help='Output directory.')
    parser.add_argument('--landmark-file',
                        help='Where to write detected faces, landmarks, ... If no specified out-dir/landmarks.txt')
    parser.add_argument('--skip-frames',
                        default=0, type=int,
                        help='How many frames to skip between two processed frames.')
    parser.add_argument('-v', '--verbose',
                        action='store_true')

    args = parser.parse_args()

    if not args.landmark_file:
        args.landmark_file = os.path.join(args.out_dir, "landmarks.csv")

    return args


def loadBBoxFile(fileName):
    inputFileTxt = open(fileName, "r")
    lines = inputFileTxt.readlines()
    lastFrameId = -1
    BBoxList = []
    for i in range(len(lines)):
        line = lines[i].rstrip().split('\t')
        dRect = dlib.rectangle(
            left=int(line[2]), top=int(line[3]),
            right=int(line[4]), bottom=int(line[5]))
        frameId = int(line[0])
        if lastFrameId != frameId:
            dRectList = dlib.rectangles()
            BBoxList.append(dRectList)
        dlib.rectangles.append(dRectList, dRect)
        lastFrameId = frameId
    print("Loaded {} bounding boxes for {} frames.".format(len(lines),
          len(BBoxList)))
    return BBoxList


def loadLandmarkFile(fileName):
    inputFileTxt = open(fileName, "r")
    lines = inputFileTxt.readlines()
    lastFrameId = -1
    landmarksList = []
    for i in range(len(lines)):
        line = lines[i].rstrip().split('\t')

        rectStr = line[2].split(':')
        rect = dlib.rectangle(
            left=int(rectStr[0]), top=int(rectStr[1]),
            right=int(rectStr[2]), bottom=int(rectStr[3]))

        parts = []
        for j in range(len(line) - 5):
            pointStr = line[j + 3].split(':')
            parts.append(dlib.point(x=int(pointStr[0]), y=int(pointStr[1])))

        shape = dlib.full_object_detection(rect, parts)
        frameId = int(line[0])
        if lastFrameId != frameId:
            shapeList = []
            landmarksList.append(shapeList)
        shapeList.append(shape)
        lastFrameId = frameId

    print("Loaded landmarks for {} faces in {} frames.".format(len(lines),
          len(landmarksList)))
    return landmarksList


def saveFaces(fileLandmarks, frameId, k, shape):
    rectStr = ":".join((
        str(shape.rect.left()), str(shape.rect.top()),
        str(shape.rect.right()), str(shape.rect.bottom())))
    partsStr = '\t'.join(
        [':'.join((str(i.x), str(i.y))) for i in shape.parts()])
    fileLandmarks.write(
        "{}\t{}\t{}\t{}\n".format(frameId, k, rectStr, partsStr))


def getRigidTransform(p1, p2):
    p2 = p2.reshape(p2.size)
    lines = []
    for p in p1:
        lines.append(np.asarray([p[0], p[1], 1, 0]))
        lines.append(np.asarray([p[1], -p[0], 0, 1]))
    lines = np.vstack(lines)
    m, residual, rank, s = np.linalg.lstsq(lines, p2)
    H = np.identity(3)
    H[0, 0] = m[0]
    H[0, 1] = m[1]
    H[0, 2] = m[2]
    H[1, 0] = -m[1]
    H[1, 1] = m[0]
    H[1, 2] = m[3]
    return H


# not used at the moment
def alignFace_v1(shape, frame, landmarkIndices=OUTER_EYES_AND_NOSE, imgDim=260, cropScale=1.0):
    landmarks= list(map(lambda p: (p.x, p.y), shape.parts()))

    npLandmarks = np.float32(landmarks)
    npLandmarkIndices = np.array(landmarkIndices)

    faceResolution = imgDim * 1/cropScale
    faceBorder = faceResolution * (cropScale - 1) / 2

    P1 = npLandmarks
    P1 = P1.reshape((1, -1, 2))
    P1 = P1
    P2 = faceBorder + faceResolution * MINMAX_TEMPLATE
    P2 = P2.reshape((1, -1, 2))
    H = cv2.estimateRigidTransform( P1.astype(np.float32), P2.astype(np.float32), fullAffine=False)

    H2 = getRigidTransform(npLandmarks, faceBorder + faceResolution * MINMAX_TEMPLATE)

    thumbnail = cv2.warpAffine(frame, H2[0:2,:], (imgDim, imgDim), borderMode=cv2.BORDER_REPLICATE)
    return thumbnail


class faceAligner(object):
    '''
    Is able to compute precise and axis-aligned face bounding boxes,
    crop faces, draw BB, ...
    '''
    def __init__(self, faceResolution, cropResolution, landmarkIndices=np.array(range(60))):
        self.landmarkIndices = landmarkIndices
        self.faceResolution = faceResolution
        self.faceBorder = (cropResolution - faceResolution) / 2
        self.cropResolution = tuple(cropResolution.tolist())
        self.bBox = dlib.rectangle(
            left=self.faceBorder[1], top=self.faceBorder[0],
            right=self.faceBorder[1] + self.faceResolution[1],
            bottom=self.faceBorder[0] + self.faceResolution[0])
        self.bBoxMat = np.array([
            [self.faceBorder[1], self.faceBorder[0], 1],
            [self.faceBorder[1] + self.faceResolution[1], self.faceBorder[0], 1],
            [self.faceBorder[1] + self.faceResolution[1], self.faceBorder[0] + self.faceResolution[0], 1],
            [self.faceBorder[1], self.faceBorder[0] + self.faceResolution[0], 1],
            ]).T

    def getTranformation(self, shape):
        ''' Computes rigid body face trasformation '''
        if not isinstance(shape, list):
            landmarks = [(p.x, p.y) for p in shape.parts()]
        else:
            landmarks = shape
        landmarks = np.float32(landmarks)
        H = getRigidTransform(
            landmarks[self.landmarkIndices],
            self.faceBorder + self.faceResolution * MINMAX_TEMPLATE[self.landmarkIndices])
        return H

    def transform(self, frame, H):
        ''' Crop face '''
        return cv2.warpAffine(
            frame, H[0:2, :], self.cropResolution,
            borderMode=cv2.BORDER_REPLICATE)

    def getCropBBox(self):
        ''' Return tight face bounding box in the face crop. '''
        return self.bBox

    def getPreciseBBox(self, H):
        ''' Return tight face bounding box in the face crop. '''
        bbox = np.linalg.inv(H).dot(self.bBoxMat)
        bbox = bbox[0:2, :] / bbox[2, :]
        return bbox

    def getSimpleBBox(self, H):
        bbox = self.getPreciseBBox(H)
        tl = bbox[:, 0].reshape(-1)
        tr = bbox[:, 1].reshape(-1)
        br = bbox[:, 2].reshape(-1)
        center = (tl + br) / 2
        radius = np.linalg.norm((tl - tr) / 2) * 1.1
        rect = dlib.rectangle(
            left=int(center[0] - radius), top=int(center[1] - radius),
            right=int(center[0] + radius), bottom=int(center[1] + radius))
        return rect

    def drawPreciseBBox(self, frame, H, color=(0, 255, 0), thickness=1):
        bbox = self.getPreciseBBox(H).astype(np.int32)
        for p1 in range(4):
            p2 = (p1 + 1) % 4
            cv2.line(
                frame, pt1=(bbox[0, p1], bbox[1, p1]),
                pt2=(bbox[0, p2], bbox[1, p2]),
                color=color, thickness=thickness)


# not used at the moment
class faceTracker(object):
    instanceCounter = 0
    def __init__(self, faceResolution, cropResolution, facePredictor, detector, trackingIterations=1):
        self.aligner = faceAligner(faceResolution, cropResolution)
        self.trackingIterations = trackingIterations
        self.facePredictor = facePredictor
        self.frameBBox = None
        self.ID = faceTracker.instanceCounter
        self.detector = detector
        faceTracker.instanceCounter += 1

    def localizeFeatures(self, frame, bbox):
        return self.facePredictor(frame, bbox)

    def start(self, frame, bbox):
        shape = self.localizeFeatures(frame, bbox)
        self.H = self.aligner.getTranformation(shape)
        self.nextFrame( frame)
        return self

    def nextFrame(self, frame):
        for i in range(self.trackingIterations):
            crop = self.aligner.transform(frame, self.H)
            shape = self.localizeFeatures(crop, self.aligner.getCropBBox())
            H = self.aligner.getTranformation(shape)
            self.H = H.dot(self.H)
        self.frameBBox = self.aligner.getPreciseBBox(self.H)

    def getCrop(self, frame):
        return self.aligner.transform(frame, self.H)


class faceTrackerOrig(object):
    instanceCounter = 0
    def __init__(self, faceResolution, cropResolution, facePredictor, trackingIterations=1):
        self.trackingIterations = trackingIterations
        self.faceAligner = faceAligner(faceResolution, cropResolution)
        self.facePredictor = facePredictor
        self.frameBBox = None
        self.ID = faceTracker.instanceCounter
        self.shape = None
        faceTracker.instanceCounter += 1

    def localizeFeatures(self, frame, bbox):
        return self.facePredictor(frame, bbox)

    def start(self, frame, bbox):
        self.shape = self.localizeFeatures(frame, bbox)
        self.H = self.faceAligner.getTranformation(self.shape)
        self.frameBBox = self.faceAligner.getPreciseBBox(self.H)
        return self

    def nextFrame(self, frame):
        for i in range(self.trackingIterations):
            self.shape = self.localizeFeatures(frame, self.faceAligner.getSimpleBBox(self.H))
            self.H = self.faceAligner.getTranformation(self.shape)
        self.frameBBox = self.faceAligner.getPreciseBBox(self.H)

    def getCrop(self, frame):
        return self.faceAligner.transform(frame, self.H)


def bboxSize(bbox):
    tl = bbox[:, 0].reshape(-1)
    br = bbox[:, 2].reshape(-1)
    return np.linalg.norm((tl - br) / 2)

class faceDetector(object):
    def __init__(self, detector_downscale=1.0):
        self.detector = dlib.get_frontal_face_detector()
        self.detector_downscale = detector_downscale

    def __call__(self, frame):
        if self.detector_downscale != 1.0:
            frame = cv2.resize(
                frame, (0, 0), fx=1. / self.detector_downscale,
                fy=1. / self.detector_downscale, interpolation=cv2.INTER_AREA)

        # The second paprameter is upscale factor
        dets = self.detector(frame, 1)
        for k, bbox in enumerate(dets):
            dets[k] = dlib.rectangle(
                left=int(bbox.left() * self.detector_downscale + 0.5),
                top=int(bbox.top() * self.detector_downscale + 0.5),
                right=int(bbox.right() * self.detector_downscale + 0.5),
                bottom=int(bbox.bottom() * self.detector_downscale + 0.5))
        return dets


class multipleFaceTracker(object):
    def __init__(self, landmark_model, detector, verbose=True):
        self.detector = detector
        self.facePredictor = dlib.shape_predictor(landmark_model)
        self.verbose = verbose
        if self.verbose:
            self.win = dlib.image_window()
        self.faceAligner = faceAligner(
            faceResolution=np.array((64, 64)),
            cropResolution=np.array((96, 96)))

        self.trackedFaces = []

        self.trackedFaceResolution = np.array((128, 128))
        self.trackedCropResolution = np.array((192, 192))
        self.overlapThreshold = 0.35
        self.frameID = 0
        self.detectStep = 1
        self.faceID = 0

    def getFacePositions(self):
        trackedCenters = []
        trackedRadiuses = []
        if self.trackedFaces:
            for face in self.trackedFaces:
                tl = face.frameBBox[:, 0].reshape(-1)
                br = face.frameBBox[:, 2].reshape(-1)
                trackedCenters.append((tl + br) / 2)
                trackedRadiuses.append(np.linalg.norm((tl - br) / 2))
            trackedCenters = np.vstack(trackedCenters)
            trackedRadiuses = np.array(trackedRadiuses)
        return trackedCenters, trackedRadiuses

    def computeOverlapMatrix(self, detections):
        trackedCenters, trackedRadiuses = self.getFacePositions()
        overlapMatrix = np.zeros(
            (len(trackedCenters) + 1, len(detections) + 1))
        if len(trackedCenters) > 0:
            for num, bbox in enumerate(detections):
                tl = np.array([bbox.left(), bbox.top()])
                br = np.array([bbox.right(), bbox.bottom()])
                detCenter = (tl + br) / 2
                detRadius = np.linalg.norm((tl - br) / 2)
                distances = np.sqrt(
                    np.sum((detCenter - trackedCenters) ** 2, axis=1))
                r = np.minimum(trackedRadiuses, detRadius)
                R = np.maximum(trackedRadiuses, detRadius)
                overlaps = r / R * (1 - distances / (r + R))
                overlapMatrix[0:-1, num] = overlaps

        return overlapMatrix

    def terminateTracksAddDetections(self, frame, detections):

        # colums are for detections (dim2 indexes detections)
        # rows are for tracked faces (dim1 indexes faces)
        overlapMatrix = self.computeOverlapMatrix(detections)

        # remove and update existing tracks
        newTracks = []
        for trackOverlaps, trackedFace in zip(overlapMatrix, self.trackedFaces):
            closestID = np.argmax(trackOverlaps)
            if trackOverlaps[closestID] >= self.overlapThreshold:
                newTracks.append(trackedFace.start(frame, detections[closestID]))

        # add non-tracked detections
        for detectionOverlaps, detection in zip(overlapMatrix.T, detections):
            if detectionOverlaps.max() < self.overlapThreshold:
                faceTrack = faceTrackerOrig(
                    faceResolution=self.trackedFaceResolution,
                    cropResolution=self.trackedCropResolution,
                    facePredictor=self.facePredictor)
                newTracks.append(faceTrack.start(frame, detection))

        self.trackedFaces = newTracks

    def processFrame(self, frame):
        origFrame = frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self.frameID % self.detectStep == 0:
            dets, points = self.detector(frame)
            self.terminateTracksAddDetections(frame, dets)
        else:
            for face in self.trackedFaces:
                face.nextFrame(frame)

        self.frameID += 1

        if self.verbose:
            visFrame = origFrame.copy()
            for face in self.trackedFaces:
                face.faceAligner.drawPreciseBBox(
                    visFrame, face.H, color=(0, 255, 0), thickness=1)
            cv2.imshow('vis', visFrame)
            cv2.waitKey(1)

        return [(f.ID, f.shape) for f in self.trackedFaces]


def cropFace(frame, shape, align_version=1, cropSize=128, cropScale=1.5):
    if align_version == 1:
        crop = alignFace_v1(shape, frame, imgDim=cropSize, cropScale=cropScale)
    else:
        print(
            "Alignment number {} is not implemented.".format(align_version),
            file=sys.stderr)
        exit(-1)

    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    return crop

from CNNdetector import mtcnnFaceDetector

####################################################################
# main
####################################################################
def main():

    frameId = 0
    args = parse_args()

    cap = cv2.VideoCapture(args.input_video)
    resolution = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    pixels = resolution[0] * resolution[1]
    targetPixels = 80000
    if not args.downscale_factor:
        args.downscale_factor = max((pixels / targetPixels) ** 0.5, 1)
    print('INFO downscale_factor {}'.format(args.downscale_factor))

    detector = mtcnnFaceDetector('/home/ihradis/projects/2016-08-31_face_service/MixedEmotions/data/mtcnn/')
    tracker = multipleFaceTracker(
        landmark_model=args.landmark_model,
        detector=detector, verbose=args.verbose)

    # used to make output face crops
    aligner = faceAligner(
        faceResolution=np.array((
            int(args.crop_resolution / args.crop_scale),
            int(args.crop_resolution / args.crop_scale))),
        cropResolution=np.array((args.crop_resolution, args.crop_resolution)))

    # create output dir
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    outputFile = open(args.landmark_file, 'w')

    t1 = time()
    while(True):
        ret, frame = cap.read()
        if not ret:
            break
        frameId += 1
        if frameId % (args.skip_frames + 1) != 0:
            continue

        # track and detect faces
        faces = tracker.processFrame(frame)

        for k, face in faces:
            saveFaces(outputFile, frameId, k, face)

        alignedFaces = []
        for k, face in faces:
            H = aligner.getTranformation(face)
            crop = aligner.transform(frame, H)
            alignedFaces.append((k, crop))

        for k, crop in alignedFaces:
            crop_name = 'face_{:06d}_{:04d}.jpg'.format(frameId, k)
            cv2.imwrite(
                os.path.join(args.out_dir, crop_name),
                crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    print('DONE with average fps: ', frameId / (time()-t1))

    outputFile.close()
    cap.release()


if __name__ == "__main__":
    main()

#!/usr/bin/env python

import sys
import numpy as np
import shutil
import re
import itertools
import operator

import cv2
import os

def parse_args():
    print( ' '.join(sys.argv))
    import argparse
    parser = argparse.ArgumentParser(epilog="Render Labels To Video")

    parser.add_argument('-iv', '--input-video',
                        required=True,
                        help='Input video.')
    parser.add_argument('-ov', '--output-video',
                        required=True,
                        help='Output video.')
    parser.add_argument('-ed', '--emoticons-dir',
                        help='Directory with images of emoticons for each emotion (angry.png, disgust.png, fear.png, happy.png, sad.png, surprise.png, neutral.png, male.png, female.png)')
    parser.add_argument('-ef', '--emotions-file',
                        help='File with emotions predictions.')
    parser.add_argument('-gf', '--gender-file',
                        help='File with gender predictions.')
    parser.add_argument('-af', '--age-file',
                        help='File with age predictions.')
    parser.add_argument('-lf', '--landmark-file',
                        help='Landmark file.')
    parser.add_argument('-b', '--render-bbox',
                        action='store_true')
    parser.add_argument('-l', '--render-landmarks',
                        action='store_true')
    parser.add_argument('-of', '--orientation-file',
                        help='File with pose of head.')
    parser.add_argument('-g', '--render-gaze',
                        action='store_true')
    parser.add_argument('-p', '--render-pose',
                        action='store_true')
    parser.add_argument('-nf', '--number-frames',
                        type=int,
                        help='Number of frames for prediction.')


    args = parser.parse_args()

    if (args.emotions_file is not None or args.gender_file is not None) and (args.emoticons_dir is None):
        parser.error("Arguments emotions_file and gender_file must be used with argument emoticons_dir")

    if (args.render_bbox or args.render_landmarks) and (args.landmark_file is None):
        parser.error("Arguments render_bbox and render_landmarks must be used with argument landmark_file")

    if (args.render_gaze or args.render_pose) and (args.orientation_file is None):
        parser.error("Arguments render_gaze and render_pose must be used with argument orientation_file")

    if (args.emotions_file is None and args.gender_file is None and args.age_file is None and args.landmark_file is None and args.orientation_file is None):
        parser.error("Nothing to render")

    return args

def countLabel(numberOfFrames, frameId, labels):

    meanLabel = np.zeros(len(labels[0][2:]))
    counterOfLabelsFrames = 0

    for index in range(-numberOfFrames, numberOfFrames + 1):
        for label in labels:
            if (frameId + index) == int(label[0]):
                meanLabel += np.asarray([float(value) for value in label[2:]])
                counterOfLabelsFrames += 1
                break

    if counterOfLabelsFrames != 0:
        meanLabel = meanLabel/float(counterOfLabelsFrames)

    return meanLabel


def renderEmotionsLabelToFrame(frame, emotionsLabel, emoticonsDirectory):

    xShiftBar = frame.shape[1]/15
    yShiftBar = frame.shape[1]/25
    heightShiftBar = frame.shape[1]/20
    widhtShiftBar = frame.shape[1]/5
    shiftBar = frame.shape[1]/18

    emoticonFileNames = ["angry.png", "disgust.png", "fear.png", "happy.png", "sad.png", "surprise.png", "neutral.png"]
    emoticonIcons = [cv2.imread(os.path.join(emoticonsDirectory, name), 1) for name in emoticonFileNames]
    emoticonIcons = [cv2.resize(im, (heightShiftBar, heightShiftBar), interpolation=cv2.INTER_AREA) for im in emoticonIcons]

    for i, icon in enumerate(emoticonIcons):
        frame[yShiftBar + shiftBar * i : yShiftBar + shiftBar * i + icon.shape[0], 0 : icon.shape[1]] = icon
        cv2.rectangle(frame, (xShiftBar, yShiftBar + shiftBar * i), (xShiftBar + int(widhtShiftBar * float(emotionsLabel[i])), yShiftBar + heightShiftBar + shiftBar * i), (0, 255, 0), -1)

    return frame

def renderGenderLabelToFrame(frame, genderLabel, emoticonsDirectory):

    xShiftBar = frame.shape[1] - frame.shape[1]/15
    yShiftBar = frame.shape[1]/25
    heightShiftBar = frame.shape[1]/20
    widhtShiftBar = frame.shape[1]/5
    shiftBar = frame.shape[1]/18

    emoticonImageFiles = ["female.png", "male.png"]
    emoticonIcons = [cv2.imread(os.path.join(emoticonsDirectory, name), 1) for name in emoticonImageFiles]
    emoticonIcons = [cv2.resize(im, (heightShiftBar, heightShiftBar), interpolation=cv2.INTER_AREA) for im in emoticonIcons]

    for i, icon in enumerate(emoticonIcons):
        frame[yShiftBar + shiftBar * i : yShiftBar + shiftBar * i + icon.shape[0], frame.shape[1] - icon.shape[1] : frame.shape[1]] = icon
        cv2.rectangle(frame, (xShiftBar, yShiftBar + shiftBar * i), (xShiftBar - int(widhtShiftBar * float(genderLabel[i])), yShiftBar + heightShiftBar + shiftBar * i), (0, 255, 0), -1)
    return frame

def renderAgeLabelToFrame(frame, ageLabel):

    xShiftBar = frame.shape[1] - frame.shape[1]/15
    yShiftBar = frame.shape[1]/12
    heightShiftBar = frame.shape[1]/170
    widhtShiftBar = frame.shape[1]
    shiftBar = frame.shape[1]/170

    ageCategories = ["0", "10", "20", "30", "40", "50", "60", "70", "80", "90", "100"]
    sizeOfFont = frame.shape[0]/650.0
    textSize = cv2.getTextSize("100", cv2.FONT_HERSHEY_SIMPLEX, sizeOfFont, 2)[0]
    textCategoriesWidth = textSize[0]
    textCategoriesHeight = textSize[1]
    shiftText = int(shiftBar*101/10.8)

    for i, age in enumerate(ageLabel[:101]):
        cv2.rectangle(frame, (frame.shape[1] - textCategoriesWidth, 2 * yShiftBar - heightShiftBar + shiftBar * i), (frame.shape[1] - textCategoriesWidth - int(widhtShiftBar * float(age)), 2 * yShiftBar + shiftBar * (i + 2)), (0, 255, 0), -1)

    for i, age in enumerate(ageCategories):
        cv2.putText(frame, age, (frame.shape[1] - textCategoriesWidth, 2 * yShiftBar + textCategoriesHeight + shiftText * i), cv2.FONT_HERSHEY_SIMPLEX, sizeOfFont, (0, 255, 0), thickness=2)


    meanAge = round(ageLabel[101], 2)
    textMeanWidth = cv2.getTextSize('Mean Age: 00.00 ', cv2.FONT_HERSHEY_SIMPLEX, frame.shape[0]/520.0, 2)[0][0]
    cv2.putText(frame, 'Mean Age: ' + str(meanAge) + ' ', (frame.shape[1] - textMeanWidth, 2 * yShiftBar + shiftBar * 113), cv2.FONT_HERSHEY_SIMPLEX, frame.shape[0]/520.0, (0, 255, 0),  thickness=2)

    return frame

def renderEyeGaze(frame, gazeLabel, cameraProperities, eyesCenter):
    fx, fy, cx, cy = cameraProperities
    if gazeLabel[2] == 0:
        return frame
    gaze = np.asarray(gazeLabel)
    gazex2d = gazeLabel[0]/-gazeLabel[2] * (frame.shape[1]/7)
    gazey2d = gazeLabel[1]/-gazeLabel[2] * (frame.shape[1]/7)
    gaze = (int(gazex2d), int(gazey2d))

    cv2.circle(frame, eyesCenter, frame.shape[1]/100, (0, 255, 0))
    cv2.line(frame, eyesCenter, tuple(map(operator.add, eyesCenter, gaze)), (0, 255, 0), 2)

def renderGazeLabelToFrame(frame, gazeLabel, cameraProperities, eyesCenter):
    renderEyeGaze(frame, gazeLabel[0:3], cameraProperities, eyesCenter[0])
    renderEyeGaze(frame, gazeLabel[3:6], cameraProperities, eyesCenter[1])

    return frame


def renderHeadAxis(frame, headRotation, headDir, headPosition, f, c, pos2D):
    headDir = headRotation.dot(headDir)
    headDir = headDir*(100) + headPosition
    headDir2D = f * headDir[0:2] / headDir[2] + c
    cv2.line(frame, tuple(pos2D.astype(np.int32).tolist()), tuple(headDir2D.astype(np.int32).tolist()), (255, 0, 0), 2)

def renderCenteredHead(frame, headPosition, headOrientation, cameraProperities):
    fx, fy, cx, cy = cameraProperities
    f = np.array([fx,fy])
    c = np.array([cx,cy])
    pos = np.asarray([float(x) for x in headPosition])
    pos2D = f * pos[0:2] / pos[2] + c
    crop = cv2.getRectSubPix(frame, (400,400), (pos2D[0], pos2D[1]))
    ori = np.asarray([float(x) for x in headOrientation])
    headRotation, headJ = cv2.Rodrigues(ori)

    rot = np.arctan(pos[1]/ pos[2])
    xRot = np.asarray([
         [1,  0,   0],
         [0, np.cos(rot), -np.sin(rot)],
         [0, np.sin(rot),  np.cos(rot)]
         ])
    pos = xRot.dot(pos)

    rot = np.arctan(pos[0]/ pos[2])
    yRot = np.linalg.inv(np.asarray([
         [np.cos(rot), 0, np.sin(rot)],
         [0,  1,   0],
         [-np.sin(rot), 0,  np.cos(rot)]
         ]))
    headRotation = headRotation.dot(xRot.dot(yRot))
    pos = yRot.dot(pos)


    #pos = xRot.dot(yRot.dot(np.asarray([float(x) for x in headPosition])))



    #headPosition[0:2] = 0
    c = np.array([200,200])
    pos2D = f * pos[0:2] / pos[2] + c
    outRotation, j = cv2.Rodrigues(headRotation)
    print( ':'.join([str(x[0]) for x in outRotation]))

    renderHeadAxis(crop, headRotation, np.array([0,0,-1]), pos, f, c, pos2D)
    renderHeadAxis(crop, headRotation, np.array([0,1,0]), pos, f, c, pos2D)
    renderHeadAxis(crop, headRotation, np.array([1,0,0]), pos, f, c, pos2D)
    cv2.imshow('Centered', crop)

def renderHeadOrientaion(frame, headPosition, headOrientation, cameraProperities):
    fx, fy, cx, cy = cameraProperities
    f = np.array([fx,fy])
    c = np.array([cx,cy])
    pos = np.asarray([float(x) for x in headPosition])
    ori = np.asarray([float(x) for x in headOrientation])
    if pos[2] == 0:
        return frame

    #renderCenteredHead(frame, np.copy(headPosition), np.copy(headOrientation), np.copy(cameraProperities))
    pos2D = f * pos[0:2] / pos[2] + c
    cv2.circle(frame, tuple(pos2D.astype(np.int32).tolist()), frame.shape[1]/100, (255, 0, 0))
    headRotation, headJ = cv2.Rodrigues(ori)
    renderHeadAxis(frame, headRotation, np.array([0,0,-1]), headPosition, f, c, pos2D)
    renderHeadAxis(frame, headRotation, np.array([0,1,0]), headPosition, f, c, pos2D)
    renderHeadAxis(frame, headRotation, np.array([1,0,0]), headPosition, f, c, pos2D)
    #cv2.imshow( 'test', frame)
    #cv2.waitKey(1)

    return frame

def renderBoxToFrame(frame, boxPoints):

    cv2.rectangle(frame, boxPoints[0], boxPoints[1], (0, 0, 255), frame.shape[1]/600)

    return frame

def renderLandmarksToFrame(frame, landmarksPoints):

    rangeChin = 16
    rangeBrow = 4
    rangeNose = 8
    rangeEye = 5
    rangeOuterMouth = 11
    rangeInnerMouth = 7

    landCounter = 0

    thicknessOfLine = frame.shape[1]/600
    colorOfLine = (255, 0, 0)

    landmarksRange = [rangeChin, rangeBrow, rangeBrow, rangeNose, rangeEye, rangeEye, rangeOuterMouth, rangeInnerMouth]

    for landRange in landmarksRange:
        for index in range(0, landRange):
            cv2.line(frame, landmarksPoints[index + landCounter], landmarksPoints[index + 1 + landCounter], colorOfLine, thicknessOfLine)
        if landRange == rangeNose:
            cv2.line(frame, landmarksPoints[landCounter + 3], landmarksPoints[index + 1 + landCounter], colorOfLine, thicknessOfLine)
        if (landRange == rangeEye) or (landRange == rangeOuterMouth) or (landRange == rangeInnerMouth):
            cv2.line(frame, landmarksPoints[landCounter], landmarksPoints[index + 1 + landCounter], colorOfLine, thicknessOfLine)

        landCounter += landRange + 1

    return frame



def main():
    args = parse_args()

    print("Creating video...")

    cap = cv2.VideoCapture(args.input_video)
    resolution = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(args.output_video + '.avi', fourcc, fps, resolution, 1)

    labelsFiles = [args.emotions_file, args.gender_file, args.age_file, args.orientation_file, args.orientation_file]
    labelsFileKeys = ['emotions', 'gender', 'age', 'gaze', 'pose']
    labelsAll = {}

    if args.number_frames is None:
        numberOfFrames = 0
    else:
        numberOfFrames = args.number_frames

    if (args.render_gaze or args.render_pose):
        with open(args.orientation_file) as f:
            orientationLabel = f.readline().split()
            cameraProperities = orientationLabel[75].split(':')
            cameraProperities = [float(x) for x in cameraProperities]

    for labelsFile, labelsKey in itertools.izip(labelsFiles, labelsFileKeys):
        if labelsFile is not None:
            with open(labelsFile) as f:
                labelsAll[labelsKey] = f.read().splitlines()
                for index in range(0, len(labelsAll[labelsKey])):
                    labelsAll[labelsKey][index] = labelsAll[labelsKey][index].split()
            if (labelsKey == 'gaze'):
                for index in range(0, len(labelsAll[labelsKey])):
                    labelsAll[labelsKey][index] = labelsAll[labelsKey][index][:2] + labelsAll[labelsKey][index][71].split(':') + labelsAll[labelsKey][index][72].split(':')
            if (labelsKey == 'pose'):
                for index in range(0, len(labelsAll[labelsKey])):
                    labelsAll[labelsKey][index] = labelsAll[labelsKey][index][:2] + labelsAll[labelsKey][index][73].split(':') + labelsAll[labelsKey][index][74].split(':')


    if args.render_bbox or args.render_landmarks:
        with open(args.landmark_file) as l:
            landmarks = l.read().splitlines()
        counterOfLandmarksFrames = 0

    frameId = 0

    while(True):

        ret, frame = cap.read()
        if(ret==False):
            break

        frameId += 1

        labels = {}

        if (args.render_bbox or args.render_landmarks):
            landmarksLabels = landmarks[counterOfLandmarksFrames].split()

        for labelsKey in labelsAll:
            labels[labelsKey] = countLabel(numberOfFrames, frameId, labelsAll[labelsKey])

        for labelKey in labels:
            if (labelKey == 'emotions'):
                frame = renderEmotionsLabelToFrame(frame, labels[labelKey], args.emoticons_dir)
            elif (labelKey == 'gender'):
                frame = renderGenderLabelToFrame(frame, labels[labelKey], args.emoticons_dir)
            elif (labelKey == 'age'):
                frame = renderAgeLabelToFrame(frame, labels[labelKey])
            elif (labelKey == 'gaze'):
                eye0 = [(int(point.split(':')[0]), int(point.split(':')[1])) for point in landmarksLabels[39:45]]
                eye1 = [(int(point.split(':')[0]), int(point.split(':')[1])) for point in landmarksLabels[45:51]]
                eye0Center = tuple([int(value/len(eye0)) for value in map(sum, itertools.izip(*eye0))])
                eye1Center = tuple([int(value/len(eye1)) for value in map(sum, itertools.izip(*eye1))])
                frame = renderGazeLabelToFrame(frame, labels[labelKey], cameraProperities, (eye0Center, eye1Center))
            elif (labelKey == 'pose'):
                frame = renderHeadOrientaion(frame, labels[labelKey][0:3], labels[labelKey][3:6], cameraProperities)


        if (landmarksLabels[0] == str(frameId)):
            if args.render_bbox:
                boxLand = landmarksLabels[2].split(':')
                boxLand = [int(point) for point in boxLand]
                boxPoints = []
                for index in range(0,2):
                    boxPoints.append((boxLand[index*2], boxLand[index*2 + 1]))
                frame = renderBoxToFrame(frame, boxPoints)

            if args.render_landmarks:
                landmarksPoints = landmarksLabels[3:71]
                landmarksPoints = [(int(point.split(':')[0]), int(point.split(':')[1])) for point in landmarksPoints]
                frame = renderLandmarksToFrame(frame, landmarksPoints)
                counterOfLandmarksFrames += 1
            try:
                while (landmarks[counterOfLandmarksFrames].split()[0] == str(frameId)):
                    counterOfLandmarksFrames += 1
            except Exception:
                    counterOfLandmarksFrames -= 1

        writer.write(frame)

        if frameId % 100 == 0:
            print('{} images processed'.format(frameId))

    cap.release()
    writer.release()

if __name__ == "__main__":
    main()

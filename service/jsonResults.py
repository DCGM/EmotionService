#!/usr/bin/env python

import sys
import numpy as np
import os
import json
from collections import defaultdict

def parse_args():
    print( ' '.join(sys.argv))
    import argparse
    parser = argparse.ArgumentParser(epilog="Convert result files to JSON.")

    parser.add_argument('-o', '--output',
                        required=True,
                        help='Output JSON file name.')
    parser.add_argument('-e', '--emotions-file',
                        help='File with emotions predictions.')
    parser.add_argument('-g', '--gender-file',
                        help='File with gender predictions.')
    parser.add_argument('-a', '--age-file',
                        help='File with age predictions.')
    parser.add_argument('-l', '--landmark-file',
                        help='Landmark file.')
    parser.add_argument('-L', '--openface-landmark-file',
                        help='Openface landmark file.')
    parser.add_argument('-A', '--openface-au-file',
                        help='Openface landmark file.')

    args = parser.parse_args()

    return args

def defDictOfDict():
    return defaultdict(dict)

def main():
    args = parse_args()

    allFrameData = defaultdict(defDictOfDict)
    emotion_names = ['anger', 'disgust', 'fear',
                'smile', 'sad', 'surprised', 'neutral']

    if args.emotions_file:
        with open(args.emotions_file, 'r') as f:
            for line in f:
                words = line.split()
                frameID = int(words[0])
                personID = int(words[1])
                data = [float(x) for x in words[2:]]
                allFrameData[frameID][personID]['emotions'] = data

                namedEmotions = {}
                for emotion, score in zip(emotion_names, data):
                    namedEmotions[emotion] = score
                allFrameData[frameID][personID]['named_emotions'] = namedEmotions


    if args.gender_file:
        with open(args.gender_file, 'r') as f:
            for line in f:
                words = line.split()
                frameID = int(words[0])
                personID = int(words[1])
                data = [float(x) for x in words[2:]]
                allFrameData[frameID][personID]['gender'] = data

    if args.age_file:
        with open(args.age_file, 'r') as f:
            for line in f:
                words = line.split()
                frameID = int(words[0])
                personID = int(words[1])
                data = float(words[-1])
                allFrameData[frameID][personID]['age'] = data

    if args.landmark_file:
        with open(args.landmark_file, 'r') as f:
            for line in f:
                words = line.split()
                frameID = int(words[0])
                personID = int(words[1])

                bb = [int(x) for x in words[2].split(':')]
                bb = {'lt': bb[0:2], 'rb': bb[2:4]}
                allFrameData[frameID][personID]['bounding_box'] = bb

                landmarks = [[int(y) for y in x.split(':')] for x in words[3:71]]
                allFrameData[frameID][personID]['landmarks'] = landmarks

    if args.openface_landmark_file:
        with open(args.openface_landmark_file, 'r') as f:
            for line in f:
                words = line.split()
                frameID = int(words[0])
                personID = int(words[1])
                leftGaze = [float(x) for x in words[71].split(':')]
                rightGaze = [float(x) for x in words[72].split(':')]
                headPosition = [float(x) for x in words[73].split(':')]
                headPose = [float(x) for x in words[74].split(':')]

                allFrameData[frameID][personID]['gaze_left'] = leftGaze
                allFrameData[frameID][personID]['gaze_right'] = rightGaze
                allFrameData[frameID][personID]['head_pose'] = headPose

    unitID = [1,2,4,5,6,7,9,10,12,14,15,17,20,23,25,26,45, 28]

    if args.openface_au_file:
        with open(args.openface_au_file, 'r') as f:
            for line in f:
                words = line.split()
                frameID = int(words[0])
                personID = int(words[1])
                AU = [x.split(':') for x in words[2:]]
                AU[-1] = [AU[-1][0], 0]
                AU = [(int(x[0]), float(x[1])) for x in AU]
                auDict = {}
                for auID, au in zip(unitID, AU):
                    auDict[auID] = au
                allFrameData[frameID][personID]['action_units'] = auDict

    with open(args.output, 'w') as f:
        json.dump( allFrameData, f, sort_keys=True, indent=1, separators=(',', ': '))


if __name__ == "__main__":
    main()

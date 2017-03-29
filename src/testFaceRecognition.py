#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import time
import re
import cv2
import os
import sys
from collections import defaultdict
from classifySoftmax import Net
from sklearn.metrics import pairwise_distances


def parse_args():
    print( ' '.join(sys.argv))
    import argparse
    parser = argparse.ArgumentParser(epilog="Extract features with Openface network.")

    parser.add_argument('-a', '--annotation',
                        required=True,
                        help='File with annotation. File name and person identifier on each line.')
    parser.add_argument( '-f', '--feature-file',
        help="Numpy matrix with features.", required=True)
    parser.add_argument('-i', '--iterations',
                        default=100,
                        type=int,
                        help='Number of random evaluation iterations.')

    args = parser.parse_args()

    return args

def readAnnotation(fileName):
    annotations = defaultdict(list)
    with open(fileName, 'r') as f:
        for lineID, line in enumerate(f):
            fileName, personID = line.split()
            annotations[personID].append({'line':lineID, 'file': fileName})
    # for person in annotations.keys():
    #     if len(annotations[person]) == 1:
    #         del annotations[person]
    #         print('Removing "{}" from evaluation as it has only one example.'.format(person))

    return annotations


def main():
    args = parse_args()

    features = np.load(args.feature_file)
    annotation = readAnnotation(args.annotation)
    print('Evaluating on {} identities.'.format(len(annotation)))

    sumCount = 0
    goodCount = 0

    for iteration in range(args.iterations):
        queryIDs = []
        databaseIDs = []
        for person in annotation:
            query, database = np.random.choice( annotation[person], 2, replace=False)
            queryIDs.append(query['line'])
            databaseIDs.append(database['line'])

        queryFeatures = features[ queryIDs, :]
        databaseFeatures = features[databaseIDs, :]

        distances = pairwise_distances(queryFeatures, databaseFeatures, metric='cosine')


        good0 = np.sum(np.argmin( distances, axis=0) == np.arange(len(annotation)))
        good1 = np.sum(np.argmin( distances, axis=1) == np.arange(len(annotation)))
        sumCount += 2*len(annotation)
        goodCount += good0 + good1
        print(len(annotation), good0, good1,  good0+good1/2.0/len(annotation), sumCount, goodCount/float(sumCount))

if __name__ == "__main__":
    main()

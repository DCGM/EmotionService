#!/usr/bin/env python
import numpy as np
import os

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(epilog="Count age")

    parser.add_argument('-cf', '--class-file',
                        required=True,
                        help='File with age predictions.')
    parser.add_argument('-o', '--output-file',
                        required=True)

    args = parser.parse_args()

    return args

def main():

    args = parse_args()

    with open(args.class_file) as f:
        lines = f.read().splitlines()

    c = open(os.path.join(os.getcwd(), args.output_file), 'w')

    valuesLine = []

    for index in range(0, len(lines)):
        valuesLine.append(lines[index].split())

    for index in range(0, len(lines)):
        meanAge = 0
        values = np.asarray([float(age) for age in valuesLine[index][2:]])
        for i, value in enumerate(values):
            meanAge += i*value
        c.write("{} {}\n".format(lines[index], meanAge))

    c.close()

if __name__ == "__main__":
    main()

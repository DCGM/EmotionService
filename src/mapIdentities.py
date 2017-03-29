#!/usr/bin/env python
from __future__ import print_function


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input',
                        required=True,
                        help='Input text file.')
    parser.add_argument('-m', '--map',
                        required=True,
                        help='Map file.')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    mapping = {}
    with open(args.map, 'r') as f:
        for line in f:
            words = line.split()
            mapping[words[0]] = words[1]

    with open(args.input, 'r') as f:
        for line in f:
            words = line.split()
            words[1] = mapping[words[1]]
            print('\t'.join(words))


if __name__ == "__main__":
    main()

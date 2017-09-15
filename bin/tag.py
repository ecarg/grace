#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
tagging named entity from trained model
__author__ = 'Hubert (voidtype@gmail.com)'
__copyright__ = 'No copyright. Just copyleft!'
"""


###########
# imports #
###########
import argparse
import logging
import sys
import tagger

GPU_NUM = 5

#############
# functions #
#############
def run(args):    # pylint: disable=too-many-locals,too-many-statements
    """
    run function which is the start point of program
    :param  args:  arguments
    """
    grace = tagger.GraceTagger(args.model, args.gpu_num)

    if args.eval:
        grace.evaluate(sys.stdin)
        return

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        tagged_line = grace.tagging(line)
        print(tagged_line)

########
# main #
########
def main():
    """
    main function processes only argument parsing
    """
    parser = argparse.ArgumentParser(description='train model from data')
    parser.add_argument('-m', '--model', help='model path', metavar='FILE', required=True)
    parser.add_argument('-i', '--input', help='corpus to tagging', metavar='FILE', required=True)
    parser.add_argument('-o', '--output', help='named entity tagged corpus', metavar='FILE')
    parser.add_argument('--gpu-num', help='GPU number to use <default: %d>' % GPU_NUM,\
            metavar='INT', type=int, default=GPU_NUM)
    parser.add_argument('--log', help='loss and accuracy log file', metavar='FILE',
                        type=argparse.FileType('wt'))
    parser.add_argument('--eval', help='check f-score', action='store_true')
    parser.add_argument('--debug', help='enable debug', action='store_true')
    args = parser.parse_args()

    if args.input:
        sys.stdin = open(args.input, 'rt')
    if args.output:
        sys.stdout = open(args.output, 'wt')
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    run(args)


if __name__ == '__main__':
    main()

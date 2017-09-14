#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
raw 코퍼스 포맷을 학습 가능한 형태로 변경합니다.

__author__ = 'Hubert (voidtype@gmail.com)'
__copyright__ = 'Copyright (C) 2017-, grace TEAM. All rights reserved.'
"""

###########
# imports #
###########
import argparse
import logging
import sys
import corpus_parser
import data

def run(args):
    """
    좌우, 특정 길이만큼 컨텍스트를 갖는 학습데이터를 생성합니다.
    :param  args:  arguments
    """
    voca = data.load_voca(args.rsc_dir, args.phonemes, args.cutoff)

    for sent in corpus_parser.sents(sys.stdin):
        cnn_corpus = sent.translate_cnn_corpus(voca, args.context_size, args.phonemes)
        for label, src_txt in cnn_corpus:
            print("%s\t%s" % (label, src_txt))
        print("")

########
# main #
########
def main():
    """
    make training set
    """
    parser = argparse.ArgumentParser(description='make training set for sequence to sequence model')
    parser.add_argument('--input', help='input file <default: stdin>', metavar='FILE')
    parser.add_argument('--output', help='output file <default: stdout>', metavar='FILE')
    parser.add_argument('--context-size', help='set context size]',\
            default=10, metavar='INT', type=int)
    parser.add_argument('-r', '--rsc-dir', help='resource directory', metavar='DIR', required=True)
    parser.add_argument('--debug', help='enable debug', action='store_true')
    parser.add_argument('--phonemes', help='expand phonemes context', action='store_true')
    parser.add_argument('--cutoff', help='cutoff', action='store',\
            type=int, metavar="int", default=1)
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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
raw corpus를 읽어들여서 vocabulary를 생성합니다.

__author__ = 'Hubert (voidtype@gmail.com)'
__copyright__ = 'Copyright (C) 2017-, grace TEAM. All rights reserved.'
"""

###########
# imports #
###########
import argparse
import logging
import sys
from collections import Counter
from operator import itemgetter
import unicodedata
import corpus_parser

#############
# functions #
#############

def build_syllable_voca_in(args):
    """
    음절 단위, 입력 사전을 만듭니다.
    """
    cnt = Counter()
    for sent in corpus_parser.sents(open(args.input)):
        for item in sent.named_entity:
            raw = item.ne_str.replace(" ", "")
            for syl in raw:
                cnt[syl] += 1
    sorted_cnt = sorted(cnt.items(), key=itemgetter(1), reverse=True)
    for syl, freq in sorted_cnt:
        if freq > args.cutoff:
            print('%s\t%d' % (syl, freq))

def build_voca_out(args):
    """
    출력 태그들의 사전을 만듭니다.
    """
    tags = set('O')
    for sent in corpus_parser.sents(open(args.input)):
        for item in sent.named_entity:
            if item.ne_tag != corpus_parser.OUTSIDE_TAG:
                tags.add('B-'+item.ne_tag)
                tags.add('I-'+item.ne_tag)

    for tag in sorted(list(tags)):
        print(tag)

def build_phonemes_voca_in(args):
    """
    음소단위 사전을 만듭니다.
    """
    cnt = Counter()
    for sent in corpus_parser.sents(open(args.input)):
        for item in sent.named_entity:
            raw = item.ne_str.replace(" ", "")
            for syl in raw:
                phonemes = unicodedata.normalize('NFKD', syl)
                for pho in phonemes:
                    cnt[pho] += 1
    sorted_cnt = sorted(cnt.items(), key=itemgetter(1), reverse=True)
    for syl, freq in sorted_cnt:
        if freq > args.cutoff:
            print('%s\t%d' % (syl, freq))

def run(args): # pylint: disable=too-many-locals
    """
    vocabulary를 생성합니다.
    :param  args:  arguments
    """
    if args.unit == 'syllable':
        if args.type == 'in':
            build_syllable_voca_in(args)
        elif args.type == 'out':
            build_voca_out(args)

    if args.unit == 'phonemes':
        if args.type == 'in':
            build_phonemes_voca_in(args)
        elif args.type == 'out':
            build_voca_out(args)

########
# main #
########
def main():
    """
    make training set
    """
    parser = argparse.ArgumentParser(description='make vocabulary')
    parser.add_argument('--input', help='input file <default: stdin>', metavar='FILE')
    parser.add_argument('--output', help='output file <default: stdout>', metavar='FILE')
    parser.add_argument('--unit', help='syllable or Consonants Vowels <default: SYL>',\
            metavar='[syllable|phonemes]', default='syllable')
    parser.add_argument('--type', help='label or source text <default: in>',\
            metavar='[in|out]', default='in')
    parser.add_argument('--cutoff', help='cutoff', action='store',\
            type=int, metavar="int", default=1)
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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CNN 모델로 태깅된 결과에 대해서 f-score를 계산하는 프로그램입니다.

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
import corpus_parser

#############
# functions #
#############

def read_corpus():
    """
    [(label, src_test)] 형식의 파일을 읽어서
    label만 리스트로 읽어서 리턴합니다.
    """
    corpus = []
    for line in sys.stdin:
        line = line.strip()
        if not line:
            yield corpus
            corpus = []
            continue
        label, _ = line.split('\t')
        corpus.append(label)

def run(args): # pylint: disable=too-many-locals
    """
    좌우, 특정 길이만큼 컨텍스트를 갖는 학습데이터를 생성합니다.
    :param  args:  arguments
    """
    cnt = Counter()
    for gold, pred in zip(open(args.gold), read_corpus()):
        gold_sent = corpus_parser.Sentence(gold.strip())
        cnt += gold_sent.compare_label(pred)

    if cnt['total_gold_ne'] == 0 or cnt['total_pred_ne'] == 0:
        print('f-score / (recall, precision): %.4f / (%.4f, %.4f)' % (0, 0, 0))
        return

    recall = cnt['match_ne'] / cnt['total_gold_ne']
    precision = cnt['match_ne'] / cnt['total_pred_ne']
    f_score = 2.0 * recall * precision / (recall + precision)
    print('f-score / (recall, precision): %.4f / (%.4f, %.4f)' % (f_score, recall, precision))


########
# main #
########
def main():
    """
    make training set
    """
    parser = argparse.ArgumentParser(description='make training set for sequence to sequence model')
    parser.add_argument('-g', '--gold', help='gold standard file', metavar='FILE', required=True)
    parser.add_argument('--input', help='input file <default: stdin>', metavar='FILE')
    parser.add_argument('--output', help='output file <default: stdout>', metavar='FILE')
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

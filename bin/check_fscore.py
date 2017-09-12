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
    하나의 문장단위로 리스트를 구성해서 리턴합니다.
    """
    corpus = []
    for line in sys.stdin:
        line = line.strip()
        if not line:
            yield corpus
            corpus = []
            continue
        label, src_txt = line.split('\t')
        corpus.append((label, src_txt))

def run(args): # pylint: disable=too-many-locals
    """
    좌우, 특정 길이만큼 컨텍스트를 갖는 학습데이터를 생성합니다.
    :param  args:  arguments
    """
    cnt = Counter()
    for gold, pred in zip(open(args.gold), read_corpus()):
        gold_sent = corpus_parser.Sentence(gold.strip())
        pred_sent = corpus_parser.Sentence.restore(pred)

        if args.boundary:
            gold_ne = set([x.get_ne_pos() for x in gold_sent.named_entity\
                if x.get_ne_pos() is not None])
            pred_ne = set([x.get_ne_pos() for x in pred_sent.named_entity\
                    if x.get_ne_pos() is not None])
        else:
            gold_ne = set([x.get_ne_pos_tag() for x in gold_sent.named_entity\
                    if x.get_ne_pos_tag() is not None])
            pred_ne = set([x.get_ne_pos_tag() for x in pred_sent.named_entity\
                    if x.get_ne_pos_tag() is not None])

        cnt['total_gold_ne'] += len(gold_ne)
        cnt['total_pred_ne'] += len(pred_ne)
        cnt['match_ne'] += len(gold_ne & pred_ne)

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
    parser.add_argument('--boundary', help='only named entity boundary', action='store_true')
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

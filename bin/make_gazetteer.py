#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
코퍼스에 포함된 개체명을 이용하여 gazetteer를 생성한다.
__author__ = 'krikit (krikit@naver.com)'
__copyright__ = 'No copyright. Just copyleft!'
"""


###########
# imports #
###########
import argparse
import codecs
from collections import Counter, defaultdict
import logging
import re
import sys

import corpus_parser


#############
# functions #
#############
def run():
    """
    actual function which is doing some task
    """
    gazetteer = defaultdict(Counter)
    for line in sys.stdin:
        line = line.rstrip('\r\n')
        if not line:
            continue
        sent = corpus_parser.Sentence(line)
        for ne_ in sent.named_entity:
            if ne_.ne_tag == corpus_parser.OUTSIDE_TAG or len(ne_.ne_str) < 2:
                continue
            ne_.ne_str = ne_.ne_str.lower()
            gazetteer[ne_.ne_str][ne_.ne_tag] += 1
            if ne_.ne_tag in ['DT', 'TI'] and not re.match(r'^[0-9 ]+$', ne_.ne_str):
                ne_str_norm = re.sub(r'[0-9]', '0', ne_.ne_str)
                if ne_str_norm != ne_.ne_str:
                    gazetteer[ne_str_norm]['%s0' % ne_.ne_tag] += 1
                    logging.debug('%s => %s', ne_.ne_str, ne_str_norm)
    for ne_str, cnt in sorted(gazetteer.items(), key=lambda x: x[0]):
        cates = sorted(cnt.items(), key=lambda x: x[1], reverse=True)
        print('%s\t%s' % (ne_str, ','.join([cate for cate, freq in cates])))


########
# main #
########
def main():
    """
    main function processes only argument parsing
    """
    parser = argparse.ArgumentParser(description='코퍼스에 포함된 개체명을 이용하여 gazetteer를 생성한다.')
    parser.add_argument('--input', help='input file <default: stdin>', metavar='FILE')
    parser.add_argument('--output', help='output file <default: stdout>', metavar='FILE')
    parser.add_argument('--debug', help='enable debug', action='store_true')
    args = parser.parse_args()

    if args.input:
        sys.stdin = codecs.open(args.input, 'r', encoding='UTF-8')
    if args.output:
        sys.stdout = codecs.open(args.output, 'w', encoding='UTF-8')
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    run()


if __name__ == '__main__':
    main()

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
import logging
import sys

import gazetteer


#############
# functions #
#############
def run():
    """
    actual function which is doing some task
    """
    gazetteer.build(sys.stdin, sys.stdout)


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

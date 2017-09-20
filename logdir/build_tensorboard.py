#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
학습 과정에서 출력한 텍스트 로그 파일을 기반으로 TensorBoard 로그를 만든다.
__author__ = 'Jamie (krikit@naver.com)'
__copyright__ = 'No copyright. Just copyleft!'
"""


###########
# imports #
###########
import argparse
import codecs
import logging
import os
import shutil
import sys
import time

from tensorboardX import SummaryWriter


#############
# functions #
#############
def run(args):
    """
    run function which is the start point of program
    :param  args:  arguments
    """
    file_name = os.path.basename(args.input)
    if not file_name.endswith('.tsv'):
        raise ValueError('input file is not *.tsv file: %s', file_name)

    header = sys.stdin.readline().rstrip('\r\n')
    if header != 'iter\tloss\taccuracy\tf-score':
        raise ValueError('invalid log header: %s', header)

    file_dir = os.path.dirname(args.input)
    if not file_dir:
        file_dir = '.'
    model_id = file_name[:-len('.tsv')]
    model_dir = '%s/%s' % (file_dir, model_id)
    if os.path.exists(model_dir):
        logging.info('rebuilding log: %s', model_dir)
        shutil.rmtree(model_dir)
        time.sleep(1)
    sum_wrt = SummaryWriter(model_dir)

    for line in sys.stdin:
        line = line.rstrip('\r\n')
        if not line:
            continue
        iter_, loss, accuracy, f_score = line.split('\t')
        iter_ = int(iter_)
        loss = float(loss)
        accuracy = float(accuracy)
        if accuracy > 1.0:
            accuracy /= 100.0
        f_score = float(f_score)
        if f_score > 1.0:
            f_score /= 100.0
        sum_wrt.add_scalar('loss', loss, iter_)
        sum_wrt.add_scalar('accuracy', accuracy, iter_)
        sum_wrt.add_scalar('f-score', f_score, iter_)


########
# main #
########
def main():
    """
    main function processes only argument parsing
    """
    parser = argparse.ArgumentParser(description='학습 과정에서 출력한 텍스트 로그 파일을 기반으로'
                                                 ' TensorBoard 로그를 만든다.')
    parser.add_argument('-i', '--input', help='input file', metavar='FILE', required=True)
    parser.add_argument('--debug', help='enable debug', action='store_true')
    args = parser.parse_args()

    if args.input:
        sys.stdin = codecs.open(args.input, 'r', encoding='UTF-8')
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    run(args)


if __name__ == '__main__':
    main()

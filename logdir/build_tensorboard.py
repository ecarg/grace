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
import glob
import logging
import os
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
    sys.stdin = open('%s/log.tsv' % args.in_dir, 'rt')
    header = sys.stdin.readline().rstrip('\r\n')
    if header != 'iter\tloss\taccuracy\tf-score':
        raise ValueError('invalid log header: %s', header)

    events_files = glob.glob('%s/events.out.tfevents.*' % args.in_dir)
    if events_files:
        logging.info('rebuilding log: %s', args.in_dir)
        for events_file in events_files:
            os.remove(events_file)
        time.sleep(1)
    sum_wrt = SummaryWriter(args.in_dir)

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
    parser.add_argument('-i', '--in-dir', help='input directory', metavar='DIR', required=True)
    parser.add_argument('--debug', help='enable debug', action='store_true')
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    run(args)


if __name__ == '__main__':
    main()

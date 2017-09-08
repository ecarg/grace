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


OPEN_PADDING = '<p>'
CLOSE_PADDING = '</p>'
OPEN_WORD = '<w>'
CLOSE_WORD = '</w>'


#############
# functions #
#############
def get_context(cur_idx, marked_text, context_size):
    """
    현재 위치를 기준으로 좌우 컨텍스트를 가져옵니다.
    :param cur_idx: '<w>,</w>'를 제외한 원래 현재 위치
    :param marked_text: '<w>,</w>를 포함한 리스트
    :param context_size: 추출할 컨텍스트 길이
    """
    # marked_text = ['어', '제', '<w>', '철', '수', '가', '</w>', '밥', '을', '먹', '고'
    #                 0    1            2    3         <-- cur_idx('<w>,</w>' 제외된 인덱스)
    #                 0    1      2     3    4     5   <-- ('<w>,</w>' 포함된 인덱스)
    #
    # 윈쪽 컨텍스트의 시작위치 : '철'의 경우 : 2 - 10 + 1 = -7 : 7개의 패딩 필요
    left_begin = max(cur_idx-context_size+1, 0)
    # 왼쪽 컨텍스트의 끝 위치 : '철'의 경우 : 2+1 = 3
    left_end = cur_idx+1
    left_context = marked_text[left_begin:left_end]
    # 오른쪽 컨텍스트의 시작위치 : '철'의 경우 : 2 + 2 = 4 / 현재 다음+1, <w>건너뜀+1 = +2
    right_begin = cur_idx+2
    # 오른쪽 컨텍스트의 끝위치 : '철'의 경우 2 + 10 + 2 / <w>, </w>의 개수만큼 2개 보정
    right_end = cur_idx+context_size+2
    right_context = marked_text[right_begin:right_end]
    left_padding = [OPEN_PADDING] * (context_size - len(left_context))
    right_padding = [CLOSE_PADDING] * (context_size - len(right_context))
    return ' '.join(left_padding+left_context), ' '.join(right_context+right_padding)

def run(args): # pylint: disable=too-many-locals
    """
    좌우, 특정 길이만큼 컨텍스트를 갖는 학습데이터를 생성합니다.
    :param  args:  arguments
    """
    sys.path.append('%s/src' % args.ner_dir)
    import parser # pylint: disable=import-error

    for sent in parser.sents(sys.stdin):
        org_sen = sent.raw_str().split(" ")
        words = [list(word) for word in org_sen]

        cur_idx = 0
        for idx_1, word in enumerate(words):
            marked_list = words[:]
            marked_list[idx_1] = [OPEN_WORD] + word + [CLOSE_WORD]
            marked_text = [syl for word in marked_list for syl in word]
            for syl in word:
                left, right = get_context(cur_idx, marked_text, args.context_size)
                bio, tag = sent.syl2tag[cur_idx+idx_1]
                if tag == parser.OUTSIDE_TAG:
                    label = bio
                else:
                    label = '%s-%s' % (bio, tag)
                print("%s\t%s %s %s" % (label, left, syl, right))
                cur_idx += 1
        print("")

########
# main #
########
def main():
    """
    make training set
    """
    parser = argparse.ArgumentParser(description='make training set for sequence to sequence model')
    parser.add_argument('-m', '--ner-dir', help='ner repository dir', metavar='DIR',
                        required=True)
    parser.add_argument('--input', help='input file <default: stdin>', metavar='FILE')
    parser.add_argument('--output', help='output file <default: stdout>', metavar='FILE')
    parser.add_argument('--level', help='named entity level. [main|sc2015]',\
            default='main', metavar='STRING')
    parser.add_argument('--context-size', help='set context size]',\
            default=10, metavar='INT', type=int)
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

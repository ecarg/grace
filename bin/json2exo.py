#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
JSON 포맷의 코퍼스를 엑소브레인 코퍼스 포맷으로 변환하는 스크립트
__author__ = 'krikit (krikit@naver.com)'
__copyright__ = 'No copyright. Just copyleft!'
"""


###########
# imports #
###########
import argparse
import codecs
import json
import logging
import sys


#############
# functions #
#############
def run():
    """
    actual function which is doing some task
    """
    json_obj = json.load(sys.stdin)
    for sent in json_obj['sentence']:
        sent_text = sent['text']
        sent_byte = sent_text.encode('UTF-8')
        morps = sent['morp']
        for ne_ in reversed(sent['NE']):
            ne_text = ne_['text']
            ne_byte = ne_text.encode('UTF-8')
            position = morps[ne_['begin']]['position']
            if not sent_byte[position:].startswith(ne_byte):
                # 하나의 형태소가 여러 NE로 분할된 경우. <겨울:DT><밤:TI>
                word = sent_byte[position:]
                if b' ' in word:
                    word = word[:word.index(b' ')]
                if ne_byte in word:
                    position_in_word = word.index(ne_byte)
                    position += position_in_word
                else:
                    # train.json 파일의 203417 줄에 position 값이 252로 오류가 있어 249로 수정
                    raise RuntimeError('문장에서 개체명 문자열을 발견할 수 없습니다: %s' % ne_text)
            sent_byte = (sent_byte[:position] + '<'.encode('UTF-8') + ne_byte +
                         ':'.encode('UTF-8') + ne_['type'].encode('UTF-8') +
                         '>'.encode('UTF-8') + sent_byte[position + len(ne_byte):])
        print(str(sent_byte, 'UTF-8'))


########
# main #
########
def main():
    """
    main function processes only argument parsing
    """
    parser = argparse.ArgumentParser(description='JSON 포맷의 코퍼스를 엑소브레인 코퍼스 포맷으로 변환하는'
                                                 ' 스크립트')
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

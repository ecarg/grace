#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
각종 포맷의 코퍼스를 엑소브레인 코퍼스 포맷으로 변환하는 스크립트
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
def _print_line(line):
    """
    한 줄(문장)을 출력한다. 정규화도 동시에 수행한다.
    :param  line:  줄
    """
    line = line.strip()
    if not line:
        return
    lines = line.split()
    if len(lines) > 1 and lines[-1] == '.':
        # 마지막 구두점만 따로 떨어지고 나머지가 모두 동일한 경우가 많아 앞으로 붙이는 정규화를 수행
        lines[-2] += '.'
        del lines[-1]
    print(' '.join(lines))


def _json2exo():
    """
    작년(2016년) 대회 코퍼스 형식인 JSON 포맷으로부터 변환
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
        _print_line(str(sent_byte, 'UTF-8'))


def _normalize():
    """
    엑소브레인 코퍼스는 정규화만 수행하여 출력한다.
    """
    for line in sys.stdin:
        _print_line(line)


def _train2exo():
    """
    올해(2017년) 대회 코퍼스 중 가운데 엑소브레인 코퍼스만 출력한다.
    """
    for line in sys.stdin:
        line = line.strip()
        if not line or line[0] != '$':
            continue
        _print_line(line[1:])


def run(args):
    """
    actual function which is doing some task
    :param  args:  arguments
    """
    if args.format.lower() == 'json':
        _json2exo()
    elif args.format.lower() == 'exo':
        _normalize()
    elif args.format.lower() == 'train':
        _train2exo()
    else:
        raise RuntimeError('invalid format: %s' % args.format)


########
# main #
########
def main():
    """
    main function processes only argument parsing
    """
    parser = argparse.ArgumentParser(description='각종 포맷의 코퍼스를 엑소브레인 코퍼스 포맷으로 변환하는'
                                                 ' 스크립트')
    parser.add_argument('-f', '--format', help='input file format <json, exo, train>',
                        metavar='NAME', required=True)
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

    run(args)


if __name__ == '__main__':
    main()

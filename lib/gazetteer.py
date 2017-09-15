#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
gazetteer 관련 핸들링을 수행하는 라이브러리
__author__ = 'Jamie (krikit@naver.com)'
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

import ahocorasick

import corpus_parser


#############
# constants #
#############
CATE_DELIM = ','    # delimiter between categories in gazetteer


#############
# functions #
#############
def build(fin, fout):
    """
    개체명 사전(gazetteer)을 만든다.
    :param  fin:  input file (corpus)
    :param  fout:  output file
    """
    gazet = defaultdict(Counter)
    for line in fin:
        line = line.rstrip('\r\n')
        if not line:
            continue
        sent = corpus_parser.Sentence(line)
        for ne_ in sent.named_entity:
            if ne_.ne_tag == corpus_parser.OUTSIDE_TAG or len(ne_.ne_str) < 2:
                # NE가 아니거나 2음절 아래는 버린다.
                continue
            ne_.ne_str = ne_.ne_str.lower()
            gazet[ne_.ne_str][ne_.ne_tag] += 1
            if ne_.ne_tag in ['DT', 'TI'] and not re.match(r'^[0-9 ]+$', ne_.ne_str):
                # DT, TI 카테고리에 대해 숫자를 패턴화한 엔트리를 추가한다. 단, 숫자로만 이뤄진 엔트리는 제외한다.
                ne_str_norm = re.sub(r'[0-9]', '0', ne_.ne_str)
                if ne_str_norm != ne_.ne_str:
                    # 패턴의 경우 DT0, TI0으로 가상의 카테고리를 부여한다.
                    gazet[ne_str_norm]['%s0' % ne_.ne_tag] += 1
                    logging.debug('%s => %s', ne_.ne_str, ne_str_norm)
    for ne_str, cnt in sorted(gazet.items(), key=lambda x: x[0]):
        # 빈도가 높은 순으로 카테고리를 정렬한다.
        cates = sorted(cnt.items(), key=lambda x: x[1], reverse=True)
        print('%s\t%s' % (ne_str, CATE_DELIM.join([cate for cate, _ in cates])), file=fout)


def load(fin):
    """
    개체명 사전(gazetteer)을 읽어들인다.
    :param  fin:  input file (gazetteer)
    :return:  gazetteer dictionary
    """
    gazet = {'exact': ahocorasick.Automaton(),    # pylint: disable=no-member
             'pattern': ahocorasick.Automaton()}    # pylint: disable=no-member
    for line in fin:
        line = line.rstrip('\r\n')
        if not line:
            continue
        entity, cate_str = line.split('\t')
        cates = cate_str.split(CATE_DELIM)
        exact_cates = [cate for cate in cates if not cate.endswith('0')]
        if exact_cates:
            gazet['exact'].add_word(entity, (len(entity), exact_cates))
        pattern_cates = [cate for cate in cates if cate in ['DT0', 'TI0']]
        if pattern_cates:
            gazet['pattern'].add_word(entity, (len(entity), pattern_cates))
    for aho in gazet.values():
        aho.make_automaton()
    return gazet


def match(gazet, sent):
    """
    gazetteer와 문장을 이용해 문장 내 음절 별로 매칭 태그를 부착한다.
    :param  gazet:  gazetteer dictionary
    :param  sent:  corpus_parser.Sentence object
    :return:  matched tags for characters without spaces in sentence
    """
    def _match_exact(sent_tags):
        """
        gazetteer를 이용하여 완전일치 매칭을 수행한다.
        :param  sent_tags:  음절 별 매칭 태그
        """
        for end, (length, exact_cates) in gazet['exact'].iter(sent.org.lower()):
            if length == 1:    # 1음절 매칭은 버린다.
                continue
            end += 1
            begin = end - length
            for idx in range(begin, end):
                match_tags = ['%s-%s' % ('B' if idx == begin else 'I', cate)
                              for cate in exact_cates]
                # 'B-'로 시작하는 태그의 경우 현재 문자에 같은 카테고리로 'I-'로 시작하는 태가가 있으면 추가하지 않는다.
                # 즉, 'B-DT' 태그는 현재 글자에 'I-DT' 태그가 있으면 추가하지 않는다.
                # 왼쪽에 이미 'B-' 태그로 열렸으므로 포함관계 방지를 위해
                new_tags = [tag for tag in match_tags
                            if tag[0] != 'B' or 'I-%s' % tag[2:] not in sent_tags[idx]]
                sent_tags[idx].update(new_tags)
            logging.debug('%s[%d:%d] => %s', sent.org[begin:end], begin, end,
                          CATE_DELIM.join(exact_cates))

    def _match_pattern(sent_tags):
        """
        gazetteer를 DT0, TI0 태그에 대해 패턴 매칭을 수행한다.
        :param  sent_tags:  음절 별 매칭 태그
        """
        org_0 = re.sub(r'[0-9]', '0', sent.org.lower())
        for end, (length, pattern_cates) in gazet['pattern'].iter(org_0):
            if length == 1:    # 1음절 매칭은 버린다.
                continue
            end += 1
            begin = end - length
            for idx in range(begin, end):
                # DT0, TI0 카테고리의 경우 현재 문자에 DT, TI가 이미 있으면 넣지 않는다.
                # exact로 이미 매칭이 되었으므로 pattern에 의해 중복해서 매칭할 필요가 없으므로
                match_cates = set([tag[2:] for tag in sent_tags[idx]])
                new_cates = [cate for cate in pattern_cates if cate[:-1] not in match_cates]
                match_tags = ['%s-%s' % ('B' if idx == begin else 'I', cate) for cate in new_cates]
                # 'B-'로 시작하는 태그의 경우 현재 문자에 같은 카테고리로 'I-'로 시작하는 태가가 있으면 추가하지 않는다.
                # 즉, 'B-DT' 태그는 현재 글자에 'I-DT' 태그가 있으면 추가하지 않는다.
                # 왼쪽에 이미 'B-' 태그로 열렸으므로 포함관계 방지를 위해
                new_tags = [tag for tag in match_tags
                            if tag[0] != 'B' or 'I-%s' % tag[2:] not in sent_tags[idx]]
                if new_tags:
                    sent_tags[idx].update(new_tags)
            logging.debug('%s[%d:%d] => %s', sent.org[begin:end], begin, end,
                          CATE_DELIM.join(pattern_cates))

    logging.debug(sent.org)
    sent_tags = [set() for _ in sent.org.lower()]    # include spaces
    _match_exact(sent_tags)
    _match_pattern(sent_tags)
    for idx, char in enumerate(sent.org):
        if char == ' ':
            logging.debug('[%d]', idx)
            continue
        logging.debug('[%d] %s: %s', idx, char, CATE_DELIM.join(sorted(list(sent_tags[idx]))))
    return sent_tags
    #return [sent_tags[idx] for idx, char in enumerate(sent.org) if char != ' ']    # remove spaces


def run(args):
    """
    actual function which is doing some task
    :param  args:  arguments
    """
    gazet = load(codecs.open('%s/gazetteer.dic' % args.rsc_dir, 'r', encoding='UTF-8'))
    for line in sys.stdin:
        line = line.rstrip('\r\n')
        if not line:
            continue
        sent = corpus_parser.Sentence(line)
        tags = match(gazet, sent)
        print(tags)


########
# main #
########
def main():
    """
    main function processes only argument parsing
    """
    parser = argparse.ArgumentParser(description='gazetteer와 코퍼스를 이용해 매칭을 수행한다.')
    parser.add_argument('-r', '--rsc-dir', help='resource directory', metavar='DIR', required=True)
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

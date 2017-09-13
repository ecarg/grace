# -*- coding: utf-8 -*-


"""
gazetteer 관련 핸들링을 수행하는 라이브러리
__author__ = 'Jamie (krikit@naver.com)'
__copyright__ = 'No copyright. Just copyleft!'
"""


###########
# imports #
###########
import codecs
import logging
import re

import ahocorasick


#############
# functions #
#############
def match(gazet, sent):
    """
    match in sentence
    :param  gazet:  gazetteer dictionary
    :param  sent:  corpus_parser.Sentence object
    :return:  matched tags for characters without spaces in sentence
    """
    logging.debug(sent.org)
    org_low = sent.org.lower()
    sent_tags = [set() for _ in org_low]    # include spaces
    for end, (length, exact_cates) in gazet['exact'].iter(org_low):
        if length == 1:
            continue
        end += 1
        begin = end - length
        for idx in range(begin, end):
            match_tags = ['%s-%s' % ('B' if idx == begin else 'I', cate) for cate in exact_cates]
            # 'B-'로 시작하는 태그의 경우 현재 문자에 같은 카테고리로 'I-'로 시작하는 태가가 있으면 추가하지 않는다.
            # 즉, 'B-DT' 태그는 현재 글자에 'I-DT' 태그를 추가하지 않는다. (왼쪽에 이미 열렸으므로 포함관계 방지를 위해)
            new_tags = [tag for tag in match_tags
                        if tag[0] != 'B' or 'I-%s' % tag[2:] not in sent_tags[idx]]
            sent_tags[idx].update(new_tags)
        logging.debug('%s[%d:%d] => %s', sent.org[begin:end], begin, end, ','.join(exact_cates))
    org_0 = re.sub(r'[0-9]', '0', org_low)
    for end, (length, pattern_cates) in gazet['pattern'].iter(org_0):
        if length == 1:
            continue
        end += 1
        begin = end - length
        for idx in range(begin, end):
            # DT0, TI0 카테고리의 경우 현재 문자에 DT, TI가 있으면 넣지 않는다.
            match_cates = set([tag[2:] for tag in sent_tags[idx]])
            new_cates = [cate for cate in pattern_cates if cate[:-1] not in match_cates]
            match_tags = ['%s-%s' % ('B' if idx == begin else 'I', cate) for cate in new_cates]
            # 'B-'로 시작하는 태그의 경우 현재 문자에 같은 카테고리로 'I-'로 시작하는 태가가 있으면 추가하지 않는다.
            # 즉, 'B-DT' 태그는 현재 글자에 'I-DT' 태그를 추가하지 않는다. (왼쪽에 이미 열렸으므로 포함관계 방지를 위해)
            new_tags = [tag for tag in match_tags
                        if tag[0] != 'B' or 'I-%s' % tag[2:] not in sent_tags[idx]]
            if new_tags:
                sent_tags[idx].update(new_tags)
        logging.debug('%s[%d:%d] => %s', sent.org[begin:end], begin, end, ','.join(pattern_cates))
    for idx, char in enumerate(sent.org):
        if char == ' ':
            logging.debug('[%d]', idx)
            continue
        logging.debug('[%d] %s: %s', idx, char, ','.join(sorted(list(sent_tags[idx]))))
    return [sent_tags[idx] for idx, char in enumerate(sent.org) if char != ' ']


def load(path):
    """
    load gazetteer
    :param  path:  file path
    :return:  gazetteer dictionary
    """
    gazet = {'exact': ahocorasick.Automaton(),    # pylint: disable=no-member
             'pattern': ahocorasick.Automaton()}    # pylint: disable=no-member
    for line in codecs.open(path, 'r', encoding='UTF-8'):
        line = line.rstrip('\r\n')
        if not line:
            continue
        entity, cate_str = line.split('\t')
        cates = cate_str.split(',')
        exact_cates = [cate for cate in cates if not cate.endswith('0')]
        if exact_cates:
            gazet['exact'].add_word(entity, (len(entity), exact_cates))
        pattern_cates = [cate for cate in cates if cate in ['DT0', 'TI0']]
        if pattern_cates:
            gazet['pattern'].add_word(entity, (len(entity), pattern_cates))
    for aho in gazet.values():
        aho.make_automaton()
    return gazet

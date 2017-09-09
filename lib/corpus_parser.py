#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NER 코퍼스를 파싱합니다.

__author__ = 'hubert (voidtype@gmail.com)'
__copyright__ = 'Copyright (C) 2017-, Grace TEAM. All rights reserved.'
"""

###########
# imports #
###########
import argparse
import logging
import sys
import os
import hashlib

##########
# global #
##########
OPEN_NE = '<'
CLOSE_NE = '>'
DIM = ':'
OUTSIDE_TAG = "OUTSIDE"
GUESS_TAG = "GUESS"
SPACE = ' '

#########
# types #
#########
class Statistic(object): # pylint: disable=too-few-public-methods
    """
    통계정보를 표현
    """
    def __init__(self):
        # 전체 문장 수
        self.total_sentence_count = 0
        # 개체명이 1개 이상 포함된 문장 수
        self.total_include_ne_sentence_count = 0
        # 개체명이 하나도 포함되지 않은 문장 수
        self.total_nothing_ne_sentence_count = 0
        # 공백 단위 전체 어절 수
        self.total_word_count = 0
        # 전체 개체명 개수
        self.total_ne_count = 0
        # 전체 문장 길이(charactor 기준)
        self.total_sentence_length = 0

    def print_statics(self):
        """
        통계정보를 출력합니다.
        """
        avg_word_count_per_sentence = self.total_word_count / self.total_sentence_count
        avg_sentence_length = self.total_sentence_length / self.total_sentence_count
        avg_ne_count_per_sentence = self.total_ne_count / self.total_sentence_count
        ratio_include = self.total_include_ne_sentence_count / self.total_sentence_count * 100
        ratio_nothing = self.total_nothing_ne_sentence_count / self.total_sentence_count * 100
        logging.info("############################################")
        logging.info("# TOTAL_SENTENCE_COUNT = %d", self.total_sentence_count)
        logging.info("#     include ne = %5.2f%% (%d)",\
                ratio_include, self.total_include_ne_sentence_count)
        logging.info("#     nothing ne = %5.2f%% (%d)",\
                ratio_nothing, self.total_nothing_ne_sentence_count)
        logging.info("# AVG_WORD_COUNT_PER_SENTENCE = %5.2f", avg_word_count_per_sentence)
        logging.info("# AVG_NE_COUNT_PER_SENTENCE = %5.2f", avg_ne_count_per_sentence)
        logging.info("# AVG_SENTENCE_LENGTH(char) = %5.2f", avg_sentence_length)
        logging.info("############################################")

class ParseError(Exception):
    """
    error occurred while parsing corpus
    """
    pass

class Sentence(object):
    """
    sentence
    """
    def __init__(self, sent):
        self.raw = sent
        self.named_entity = []
        self.syl2tag = {}
        self.md5 = hashlib.sha224(sent.encode("UTF-8")).hexdigest()

        for item in NamedEntity.parse(sent):
            self.named_entity.append(item)
            self.syl2tag.update(item.syl2tag)
        self.org = ''.join([item.ne_str for item in self.named_entity])

    def __str__(self):
        return ''.join([str(ne) for ne in self.named_entity])

    def __len__(self):
        return len(self.org)

    def get_word_count(self):
        """
        공백단위로 분리한 어절을 구합니다.
        """
        return len(self.org.split(" "))

    def get_original_sentence(self):
        """
        개체명 태깅이 없는 오리지날 원문 문장을 리턴합니다.
        """
        return self.org

    def get_ne_count(self):
        """
        문장 내 포함된 개체명 개수를 리턴합니다.
        """
        return sum([x.ne_tag != OUTSIDE_TAG for x in self.named_entity])

    def raw_str(self):
        """
        개체명 문자열을 리턴합니다.
        """
        return "".join([x.ne_str for x in self.named_entity])

class NamedEntity(object): # pylint: disable=too-few-public-methods
    """
    어절안에 포함된 개체명
    """
    def __init__(self, ne_str, ne_tag, ne_beg, ne_end):
        self.ne_str = ne_str
        self.ne_tag = ne_tag
        self.ne_beg = ne_beg
        self.ne_end = ne_end
        self.syl2tag = {}
        # [pos] -> PS_NAME char_pos에 해당 음절의 NE 태그를 갖는 사전을 만들어둔다.
        for idx in range(len(ne_str)):
            if idx+ne_beg in self.syl2tag:
                logging.error('something wrong!!')
                return
            if ne_tag == OUTSIDE_TAG:
                self.syl2tag[idx+ne_beg] = ('O', ne_tag)
            else:
                if idx == 0:
                    self.syl2tag[idx+ne_beg] = ('B', ne_tag)
                else:
                    self.syl2tag[idx+ne_beg] = ('I', ne_tag)

    def __str__(self):
        if self.ne_tag == OUTSIDE_TAG:
            return self.ne_str
        return OPEN_NE + ":".join([self.ne_str, self.ne_tag]) + CLOSE_NE

    def str_pair(self):
        """
        str/tag 형태로 출력합니다.
        """
        named_entity = "/".join([self.ne_str, self.ne_tag])
        position_info = "[%d~%d:(%d)]"%(self.ne_beg, self.ne_end, self.ne_end - self.ne_beg + 1)
        return named_entity + position_info

    def get_ne_pos(self):
        """
        start-end
        """
        if self.ne_tag != OUTSIDE_TAG:
            return (self.ne_beg, self.ne_end)
        return None

    def get_ne_pos_tag(self):
        """
        start-end-tag
        """
        if self.ne_tag != OUTSIDE_TAG:
            return (self.ne_beg, self.ne_end, self.ne_tag)
        return None

    def get_ne_pos_tag_word(self):
        """
        start-end-tag-word
        """
        if self.ne_tag != OUTSIDE_TAG:
            return (self.ne_beg, self.ne_end, self.ne_tag, self.ne_str)
        return None



    @classmethod
    def parse(cls, sent): # pylint: disable=too-many-branches
        """
        어절을 파싱합니다.
        :param word: 파싱할 어절
        :return yield: 파싱이 완료된 NamedEntity 객체
        """
        outside_str = '' # 개체명이 아닌 어휘
        inside_str = '' # 개체명
        is_inside = False
        start = 0
        for syl in sent:
            if syl == OPEN_NE:
                if is_inside:
                    #
                    # "<문자열1 <개체명:태그>"와 같이 "<"이 나오고 "<"가 또 열리는 경우
                    # "<문자열1"까지를 outside 태그로 처리한다.
                    #
                    outside_str = OPEN_NE + inside_str
                    inside_str = ''
                else:
                    is_inside = True
                if outside_str:
                    yield NamedEntity(outside_str, OUTSIDE_TAG, start, start+len(outside_str)-1)
                    start += len(outside_str)
                    outside_str = ''

            elif syl == CLOSE_NE:
                if is_inside:
                    try:
                        ne_str, ne_tag = inside_str.rsplit(DIM, 1)
                    except ValueError:
                        #raise ParseError('%s(%d) %s: %s' % (file_name, line_num, val_err, sent))
                        # "<표 3>","<후기>"와 같이 "<>"로 둘러싸여있지만,  개체명 정보가 아닌 경우
                        # OUTSIDE_TAG로 처리하도록 한다.
                        #
                        ne_str = OPEN_NE+inside_str+CLOSE_NE
                        ne_tag = OUTSIDE_TAG

                    yield NamedEntity(ne_str, ne_tag, start, start+len(ne_str)-1)
                    is_inside = False
                    inside_str = ''
                    start += len(ne_str)
                else:
                    #
                    # "<이효리:PS_NAME> 그리고>"와 같이 닫힌 상태에서
                    # 또 닫는경우, outside로 처리한다.
                    #
                    outside_str += CLOSE_NE
            else:
                if is_inside:
                    inside_str += syl
                else:
                    outside_str += syl
        # 입니다. <- 그래서
        # "<- 그래서" 가 처리되지 않고 남아 있는 경우 처리
        if inside_str:
            yield NamedEntity(OPEN_NE+inside_str, OUTSIDE_TAG,\
                    start, start+len(OPEN_NE+inside_str)-1)

        if outside_str:
            yield NamedEntity(outside_str, OUTSIDE_TAG, start, start+len(outside_str)-1)

#############
# functions #
#############
def sents(fin, stat=None):
    """
    코퍼스를 파싱합니다.
    :param fin: input file object
    :param stat: 통계정보를 저장할 Statistic 객체
    :return yield: sentence object
    """
    file_name = os.path.basename(fin.name)
    sent = None
    for line_num, line in enumerate(fin, start=1):
        line = line.strip()
        try:
            sent = Sentence(line)
        except ParseError as msg:
            logging.error('%s(%d) %s: %s', file_name, line_num, msg, line)

        #logging.debug(sent.get_original_sentence())
        ne_count = sent.get_ne_count()
        if stat:
            stat.total_sentence_count += 1
            if ne_count:
                stat.total_include_ne_sentence_count += 1
            else:
                stat.total_nothing_ne_sentence_count += 1
            stat.total_ne_count += ne_count
            stat.total_word_count += sent.get_word_count()
            stat.total_sentence_length += len(sent)
        yield sent

    logging.info("%s is done...", file_name)

def run(args):
    """
    문장을 읽어들이고, 출력합니다.
    """
    stat = Statistic()
    for sent in sents(sys.stdin, stat):
        print(sent)
        continue
    if args.report:
        stat.print_statics()

########
# main #
########
def main():
    """
    main function processes only argument parsing
    """
    parser = argparse.ArgumentParser(description='etri ner corpus parser')
    parser.add_argument('--input', help='input file <default: stdin>', metavar='FILE')
    parser.add_argument('--output', help='output file <default: stdout>', metavar='FILE')
    parser.add_argument('--report', help='print statistic info', action='store_true')
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

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
import collections
import math

import torch
import torch.utils.data


##########
# global #
##########
OPEN_NE = '<'
CLOSE_NE = '>'
DIM = ':'
OUTSIDE_TAG = "OUTSIDE"
GUESS_TAG = "GUESS"
SPACE = ' '
OPEN_PADDING = '<p>'
CLOSE_PADDING = '</p>'
OPEN_WORD = '<w>'
CLOSE_WORD = '</w>'

# 학습 코퍼스의, 아이템 하나를 표현하는 자료구조
# DEBUG:root:Lable(bio='B', tag='PS', cur_syl='최', next_syl='희')
TrainEntry = collections.namedtuple('TrainEntry', 'bio tag cur_syl nxt_syl')


ACTION_TABLE = {\
    'B':{'B':'CLOSE_OPEN', 'I':'NOTHING', 'O':'CLOSE', 'EOS':'ONLY_CLOSE'},\
    'I':{'B':'CLOSE_OPEN', 'I':'NOTHING', 'O':'CLOSE', 'EOS':'ONLY_CLOSE'},\
    'O':{'B':'OPEN', 'I':'ERROR', 'O':'NOTHING', 'EOS':'QUIT'},\
    'BOS':{'B':'OPEN', 'I':'ERROR', 'O':'NOTHING', 'EOS':'NOTHING'}\
}

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
        self.tagcnt = {}
        self.md5 = hashlib.sha224(sent.encode("UTF-8")).hexdigest()
        self.label_nums = []
        self.context_nums = []
        self.label_tensors = None
        self.context_tensors = None

        for item in NamedEntity.parse(sent):
            self.named_entity.append(item)
            self.syl2tag.update(item.syl2tag)
            self.tagcnt.update(item.tagcnt)
        self.org = ''.join([item.ne_str for item in self.named_entity])

    def __str__(self):
        return ''.join([str(ne) for ne in self.named_entity])

    def __len__(self):
        return len(self.org)

    def to_num_arr(self, voca):
        """
        문장을 문장에 포함된 문자 갯수 만큼의 배치 크기의 숫자 배열을 생성하여 리턴한다.
        :param  voca:  in/out vocabulary
        :return:  문자 갯수만큼의 숫자 배열
        """
        if self.label_nums and self.context_nums:
            return self.label_nums, self.context_nums
        for label, context in self.translate_cnn_corpus(10):
            self.label_nums.append(voca['out'][label])
            self.context_nums.append([voca['in'][char] for char in context.split()])
        return self.label_nums, self.context_nums

    def to_tensor(self, voca):
        """
        문장을 문장에 포함된 문자 갯수 만큼의 배치 크기의 텐서를 생성하여 리턴한다.
        :param  voca:  in/out vocabulary
        :return:  문자 갯수만큼의 텐서
        """
        if self.label_tensors is not None and self.context_tensors is not None:
            return self.label_tensors, self.context_tensors
        self.to_num_arr(voca)
        self.label_tensors = torch.LongTensor(self.label_nums)
        self.context_tensors = torch.LongTensor(self.context_nums)
        return self.label_tensors, self.context_tensors

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

    @staticmethod
    def _get_context(cur_idx, marked_text, context_size):
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

    def translate_cnn_corpus(self, context_size=10):
        """
        context_size 크기를 갖는 ngram 형태로 학습 코퍼스를 생성합니다.
        :param context_size: context size
        """
        words = [list(word) for word in self.raw_str().split(" ")]
        cnn_corpus = []

        cur_idx = 0
        for idx_1, word in enumerate(words):
            marked_list = words[:]
            marked_list[idx_1] = [OPEN_WORD] + word + [CLOSE_WORD]
            marked_text = [syl for word in marked_list for syl in word]
            for syl in word:
                left, right = self._get_context(cur_idx, marked_text, context_size)
                bio, tag = self.syl2tag[cur_idx+idx_1]
                if tag == OUTSIDE_TAG:
                    label = bio
                else:
                    label = '%s-%s' % (bio, tag)
                #print("%s\t%s %s %s" % (label, left, syl, right))
                cnn_corpus.append((label, "%s %s %s" % (left, syl, right)))
                cur_idx += 1
        return cnn_corpus

    @staticmethod
    def _get_train_entry(src_lab, src_str):
        """
        학습 코퍼스의 엔트리 하나를, namedtuple로 만들어서 리턴합니다.
        :param src_lab: 학습코퍼스에서 label (ex, B-PS, O)
        :param src_str: 학습코퍼스에서 음절 단위 컨텍스트("<p> <p> 이 대 호")
        :return TrainEntry: 파싱된 결과를 namedtuple로 리턴
        """
        src_list = src_str.split(" ")
        mid = math.ceil(len(src_list) / 2)
        bio = tag = 'O'
        try:
            bio, tag = src_lab.split('-', 1)
        except ValueError:
            pass
        return TrainEntry(bio, tag, src_list[mid-1], src_list[mid])

    @staticmethod
    def _do_action(prev_entry, entry, sent):
        """
        현재 태그, 다음태그를 고려해서 액선을 수행한다.
        """
        action = ACTION_TABLE[prev_entry.bio][entry.bio]
        if action == 'OPEN':
            if prev_entry.nxt_syl == CLOSE_WORD:
                sent.append(' <')
            else:
                sent.append('<')
        elif action == 'CLOSE_OPEN':
            if prev_entry.nxt_syl == CLOSE_WORD:
                sent.append(':%s> <' % prev_entry.tag)
            else:
                sent.append(':%s><' % prev_entry.tag)
        elif action == 'CLOSE':
            if prev_entry.nxt_syl == CLOSE_WORD:
                sent.append(':%s> ' % prev_entry.tag)
            else:
                sent.append(':%s>' % prev_entry.tag)
        elif action == 'ONLY_CLOSE':
            sent.append(':%s>' % prev_entry.tag)
        elif action == 'NOTHING':
            if prev_entry.nxt_syl == CLOSE_WORD:
                sent.append(' ')

    @classmethod
    def restore(cls, cnn_corpus):
        """
        [(label, src_text)] 형식으로 된 학습 데이터를 raw corpus 형태로 복원합니다.
        :param cnn_corpus: 학습 데이터
        :return string: raw sentence
        """
        sent = []
        end_of_sent = TrainEntry('EOS', '', '', '')
        beg_of_sent = TrainEntry('BOS', '', '', '')
        prev_entry = beg_of_sent
        for item in cnn_corpus:
            label, src_str = item
            entry = cls._get_train_entry(label, src_str)
            logging.debug("%s-%s\t%s %s", entry.bio, entry.tag, entry.cur_syl, entry.nxt_syl)
            cls._do_action(prev_entry, entry, sent)
            sent.append(entry.cur_syl)
            prev_entry = entry
        cls._do_action(prev_entry, end_of_sent, sent)
        return cls(''.join(sent))

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
        self.tagcnt = {}
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
        if ne_tag != OUTSIDE_TAG:
            self.tagcnt[ne_tag] = 1

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

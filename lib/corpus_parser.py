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
import copy
import hashlib
import collections
import unicodedata
import string

import torch
from torch.autograd import Variable
import torch.utils.data

import gazetteer

##########
# global #
##########
OPEN_NE = '<'
CLOSE_NE = '>'
DIM = ':'
OUTSIDE_TAG = "OUTSIDE"
GUESS_TAG = "GUESS"
SPACE = ' '
PADDING = collections.OrderedDict([
    ('unk', '<UNK/>'),
    # 음절단위 학습셋에서 사용하는 패딩
    ('pre', '<p>'),
    ('suf', '</p>'),
    ('op_wrd', '<w>'),
    ('cl_wrd', '</w>'),
    # 자소단위 학습셋에서 사용하는 패딩
    # 한글 초/중/종
    ('cho', '<c>'),
    ('u_cho', '<uc>'),
    ('jung', '<j>'),
    ('u_jung', '<uj>'),
    ('jong', '<o>'),
    ('u_jong', '<uo>'),
    # dight, alphabet
    ('dig', '<d>'),
    ('u_dig', '<ud>'),
    ('eng', '<e>'),
    ('u_eng', '<ue>'),
    # hanja
    ('hanja', '<h>'),
    ('u_hanja', '<uh>'),
    # symbol
    ('symbol', '<y>'),
    ('u_symbol', '<uy>'),
    # unknown
    ('etc', '<t>')
])

# 얘측한 아이템을 표현
PredItem = collections.namedtuple('PredItem', 'beg end tag')

# CLOSE_OPEN : prev 음절을 종료처리하고, curr음절을 새로 생성한다.
# CHECK_TAG : prev와 curr의 태그가 동일한지 체크해서, 동일하지 않은 경우
#             prev를 종료처리하고, curr을 새로 생성한다.
# OPEN : curr음절을 대상으로 새로 생성한다.
# CLOSE : prev 음절을 종료처리한다.
ACTION_TABLE = {\
    'B':{'B':'CLOSE_OPEN', 'I':'CHECK_TAG', 'O':'CLOSE', 'EOS':'CLOSE'},\
    'I':{'B':'CLOSE_OPEN', 'I':'CHECK_TAG', 'O':'CLOSE', 'EOS':'CLOSE'},\
    'O':{'B':'OPEN', 'I':'OPEN', 'O':'NOTHING', 'EOS':'QUIT'},\
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


class TrainContext():
    """
    음절을 자소로 확장하는 기능을 담는 클래스 입니다.
    """
    def __init__(self, is_phonemes=False):
        self.is_phonemes = is_phonemes
    @classmethod
    def ext_hangul(cls, syl):
        """
        한글 1음절을 3개의 자소로 확장합니다.
        :param syl:  확장할 음절
        :return list: 고정길이 3을 갖는 확장된 리스트
        """
        ext = []

        char = unicodedata.normalize('NFKD', syl)
        if len(char) == 1: # 초성만 있는 경우
            ext += [char, PADDING['jung'], PADDING['jong']]
            return ext

        if len(char) == 2: # 초/중성만 있는 경우
            ch1 = char[0]
            ch2 = char[1]
            ext += [ch1, ch2, PADDING['jong']]
            return ext

        if len(char) == 3: # 초/중성만 있는 경우
            ch1 = char[0]
            ch2 = char[1]
            ch3 = char[2]
            ext += [ch1, ch2, ch3]
            return ext

        return [PADDING['etc'], syl, PADDING['etc']]

    @classmethod
    def check_char_type(cls, char):
        """
        문자에 대한 타입을 리턴합니다.
        """
        if char.isdigit():
            return PADDING['u_dig']
        elif char.isalpha():
            return PADDING['u_eng']
        elif char in string.punctuation:
            return PADDING['u_symbol']
        name = unicodedata.name(char)
        if 'HANGUL CHOSEONG' in name:
            return PADDING['u_cho']
        elif 'HANGUL JUNGSEONG' in name:
            return PADDING['u_jung']
        elif 'HANGUL JONGSEONG' in name:
            return PADDING['u_jong']
        elif 'CJK UNIFIED' in name:
            return PADDING['u_hanja']
        return PADDING['unk']

    def expand_context(self, context):
        """
        컨텍스트 문자열을 자소단위로 확장한다.
        :param  context: 확장할 문자열
        :param  is_right: 오른쪽 패딩인지 여부(open/close 구분용)
        :return: 자소단위로 확장한 문자열
        """
        ext = []
        for syl in context.split(' '):
            try:
                name = unicodedata.name(syl)
            except TypeError:
                # 1음절이 아닌 경우, 패딩이다.
                ext += [syl] * 3
            else:
                if 'HANGUL SYLLABLE' in name:
                    ext += self.ext_hangul(syl)
                elif 'CJK UNIFIED' in name:
                    ext += [PADDING['hanja'], syl, PADDING['hanja']]
                elif syl in string.punctuation:
                    ext += [PADDING['symbol'], syl, PADDING['symbol']]
                elif syl.isdigit():
                    ext += [PADDING['dig'], syl, PADDING['dig']]
                elif syl.isalpha():
                    ext += [PADDING['eng'], syl, PADDING['eng']]
                else:
                    ext += [PADDING['etc'], syl, PADDING['etc']]

        # 3배로 확장되므로, 항상 3으로 나눈 나머지는 없어야 한다.
        assert len(ext) % 3 == 0
        return ' '.join(ext)

    def get_context(self, cur_idx, marked_text, context_size):
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
        cur_syl = marked_text[cur_idx+1]
        left_begin = max(cur_idx-context_size+1, 0)
        # 왼쪽 컨텍스트의 끝 위치 : '철'의 경우 : 2+1 = 3
        left_end = cur_idx+1
        left_context = marked_text[left_begin:left_end]
        # 오른쪽 컨텍스트의 시작위치 : '철'의 경우 : 2 + 2 = 4 / 현재 다음+1, <w>건너뜀+1 = +2
        right_begin = cur_idx+2
        # 오른쪽 컨텍스트의 끝위치 : '철'의 경우 2 + 10 + 2 / <w>, </w>의 개수만큼 2개 보정
        right_end = cur_idx+context_size+2
        right_context = marked_text[right_begin:right_end]
        left_padding = [PADDING['pre']] * (context_size - len(left_context))
        right_padding = [PADDING['suf']] * (context_size - len(right_context))

        if self.is_phonemes:
            ext_syl = self.expand_context(cur_syl)
            ext_left_context = self.expand_context(' '.join(left_padding+left_context))
            ext_right_context = self.expand_context(' '.join(right_context+right_padding))
            return "%s %s %s" % (ext_left_context, ext_syl, ext_right_context)
        return ' '.join(left_padding+left_context+[cur_syl]+right_context+right_padding)

class Sentence(object):
    """
    sentence
    """
    gazet_voca = {}
    def __init__(self, sent):
        self.raw = sent
        self.named_entity = []
        self.syl2tag = {}
        self.tagcnt = {}
        self.md5 = hashlib.sha224(sent.encode("UTF-8")).hexdigest()
        self.label_nums = []
        self.context_nums = []
        self.context_strs = []    # 형태소 분석기에 입력으로 사용할 데이터
        self.gazet_matches = []
        self.label_tensors = None
        self.context_tensors = None
        self.gazet_tensors = None
        self.pos_tensors = None
        self.word_tensors = None
        self.pos_outputs = None

        for item in NamedEntity.parse(sent):
            self.named_entity.append(item)
            self.syl2tag.update(item.syl2tag)
            self.tagcnt.update(item.tagcnt)
        self.org = ''.join([item.ne_str for item in self.named_entity])

    def del_all_except_tensor(self):
        """
        메모리 사용량을 줄이기 위해 학습 시 사용하는 텐서들을 제외한 나머지는 삭제합니다.
        """
        # self.raw = ''
        # self.named_entity = []
        # self.syl2tag = {}
        self.tagcnt = {}
        # self.md5 = ''
        self.label_nums = []
        self.context_nums = []
        self.context_strs = []
        self.gazet_matches = []
        # self.label_tensors = None
        # self.context_tensors = None
        # self.gazet_tensors = None
        # self.pos_tensors = None
        # self.word_tensors = None
        self.pos_outputs = None
        self.org = ''

    def del_tensor(self):
        """
        캐시를 사용하는 경우 텐서도 메모리에서 삭제한다.
        """
        self.label_tensors = None
        self.context_tensors = None
        self.gazet_tensors = None
        self.pos_tensors = None
        self.word_tensors = None

    def __str__(self):
        return ''.join([str(ne) for ne in self.named_entity])

    def __len__(self):
        return len(self.org)

    @classmethod
    def in_to_num(cls, voca, char):
        """
        음절 또는 자소 하나를 숫자로 바꿉니다.
        :param voca: 사전
        :param char: 대상 음절 또는 자소
        """
        num = voca['in'][char]
        if num:
            return num
        return voca['in'][TrainContext.check_char_type(char)]

    def to_num_arr(self, voca, context_size, is_phonemes=False):
        """
        문장을 문장에 포함된 문자 갯수 만큼의 배치 크기의 숫자 배열을 생성하여 리턴한다.
        :param  voca:  in/out vocabulary
        :param  is_phonemes: 자소단위로 확장할 경우
        :param  context_size: context size
        :return:  문자 갯수만큼의 숫자 배열
        """
        if self.label_nums and self.context_nums:
            return self.label_nums, self.context_nums
        for label, context in self.translate_cnn_corpus(context_size, is_phonemes):
            self.label_nums.append(voca['out'][label])
            self.context_nums.append([self.in_to_num(voca, char) for char in context.split()])
        return self.label_nums, self.context_nums

    def to_str_arr(self, context_size):
        """
        문장을 문장에 포함된 문자 갯수 만큼의 배치 크기의 문자 배열을 생성하여 리턴한다.
        :param  context_size: context size
        :return:  문자 갯수만큼의 문자 배열
        """
        if self.context_strs:
            return self.context_strs
        self.context_strs = [context.split() for _, context
                             in self.translate_cnn_corpus(context_size, False)]
        return self.context_strs

    @classmethod
    def onehot2number(cls, gazet):
        """
        onehot으로 인코딩된 gazet을 1차원 LongTensor로 변경
        :param gazet: gazet matrix
        """
        gazet_list = []
        for batch in gazet:
            scala = []
            for row in batch:
                val = 0
                for idx, col in enumerate(row):
                    if int(col) == 1:
                        val += (1 << idx)
                scala.append(val)
            gazet_list.append(scala)
        return gazet_list

    def set_word_feature(self, pos_model, word_model, window):
        """
        음절단위 형태소 분석결과로부터, 단어단위 형태소 분석
        결과를 복원합니다.
        :param pos_model: 형태소 분석기
        :param pos_outpus: 형태소 분석결과(음절)
        """
        if self.word_tensors is not None:
            return

        pos_outputs = self.run_pos_tagger(pos_model, window)
        tags = [pos_model.cfg.voca['tuo'][x] for x in pos_outputs.data]
        raw_str_spc = self.raw_str()
        raw_str = raw_str_spc.replace(" ", "")
        words = []
        for idx, (char, tag) in enumerate(zip(raw_str, tags)):
            bio_i, tag_i = tag.split('-', 1)
            if tag_i[0] == 'N':
                tag_i = 'N'
            if bio_i == 'B' or bio_i == 'I' and idx == 0:
                word = [raw_str[idx]]
                for jdx in range(idx+1, len(raw_str)):
                    bio_j, tag_j = tags[jdx].split('-', 1)
                    if tag_j[0] == 'N':
                        tag_j = 'N'
                    if bio_j == 'B':
                        break
                    if tag_i == tag_j:
                        word.append(raw_str[jdx])
                    else:
                        tags[jdx] = "B-%s" % (tag_j)
                        break
                words.append("%s/%s" % (''.join(word), tag_i[0]))
            elif bio_i == 'I':
                words.append(words[idx-1])

        assert len(words) == len(raw_str)

        # 공백에 해당하는 자리에 [0 x PoS out dim] 텐서를 삽입
        out_idx = 0
        words_spc = []
        for char in raw_str_spc:
            if char == ' ':
                words_spc.append([0.0]*100)
            else:
                word_tensor = word_model.word2vec(words[out_idx])
                words_spc.append(word_tensor)
                out_idx += 1

        # BS * 1
        # BS * 21 * 100 사이즈로 만들어야 한다.
        words_contexts = []
        for idx, (char, word) in enumerate(zip(raw_str_spc, words_spc)):
            if char == ' ':
                continue
            has_boe = False
            left_context = []
            for jdx in range(idx-1, idx-(window*2), -1):
                if jdx < 0 or len(left_context) >= window:
                    break
                if raw_str_spc[jdx] == ' ':
                    if has_boe:
                        continue
                    else:
                        left_context.append(words_spc[jdx])
                        has_boe = True
                else:
                    left_context.append(words_spc[jdx])
            if not has_boe and len(left_context) < window:
                left_context.append([0.0]*100)
            left_context.extend([[0.0]* 100] * (window - len(left_context)))
            assert len(left_context) == window
            has_eoe = False
            right_context = []
            for jdx in range(idx+1, idx+(window*2)+1):
                if jdx >= len(raw_str_spc) or len(right_context) >= window:
                    break
                if raw_str_spc[jdx] == ' ':
                    if has_eoe:
                        continue
                    else:
                        right_context.append(words_spc[jdx])
                        has_eoe = True
                else:
                    right_context.append(words_spc[jdx])
            if not has_eoe and len(right_context) < window:
                right_context.append([0.0]*100)
            right_context.extend([[0.0]* 100] * (window - len(right_context)))
            assert len(right_context) == window
            words_contexts.append(\
                    list(reversed(left_context))+[word, ]+right_context)
        assert len(words_contexts) == self.get_syllable_count()
        self.word_tensors = torch.FloatTensor(words_contexts)

    def run_pos_tagger(self, pos_model, window):
        """
        run tagger
        """
        if self.pos_outputs is not None:
            return self.pos_outputs

        str_contexts = self.to_str_arr(window)
        pos_contexts = []
        for str_context in str_contexts:
            pos_contexts.append([pos_model.cfg.voca['in'][char] for char in str_context])
        pos_model.eval()
        pos_contexts_var = Variable(torch.LongTensor(pos_contexts), volatile=True)
        if torch.cuda.is_available():
            pos_contexts_var = pos_contexts_var.cuda()
        _, pos_outputs = pos_model(pos_contexts_var).max(1) # BS * 1
        self.pos_outputs = pos_outputs
        return self.pos_outputs

    def set_pos_feature(self, pos_model, window):
        """
        형태소 분석기를 실행하고 그 출력으로 임베딩을 만든다.
        :param  pos_model:  형태소 분석기 모델
        :param  window:  좌/우 문맥 길이
        """
        if self.pos_tensors is not None:
            return

       # BS * PoS tagger output vocabulary size
        pos_outputs_0 = self.run_pos_tagger(pos_model, window)
        pos_outputs = pos_outputs_0.data + 1

        # 공백에 해당하는 자리에 [0 x PoS out dim] 텐서를 삽입
        out_idx = 0
        pos_outputs_spc = []
        raw_str = self.raw_str()
        for char in raw_str:
            if char == ' ':
                pos_outputs_spc.append(0)
            else:
                pos_outputs_spc.append(pos_outputs[out_idx])
                out_idx += 1

        # BS * 1
        # BS * 21 * 1 사이즈로 만들어야 한다.
        pos_outputs_contexts = []
        for idx, (char, pos_output) in enumerate(zip(raw_str, pos_outputs_spc)):
            if char == ' ':
                continue
            has_boe = False
            left_context = []
            for jdx in range(idx-1, idx-(window*2), -1):
                if jdx < 0 or len(left_context) >= window:
                    break
                if raw_str[jdx] == ' ':
                    if has_boe:
                        continue
                    else:
                        left_context.append(pos_outputs_spc[jdx])
                        has_boe = True
                else:
                    left_context.append(pos_outputs_spc[jdx])
            if not has_boe and len(left_context) < window:
                left_context.append(0)
            left_context.extend([0] * (window - len(left_context)))
            assert len(left_context) == window
            has_eoe = False
            right_context = []
            for jdx in range(idx+1, idx+(window*2)+1):
                if jdx >= len(raw_str) or len(right_context) >= window:
                    break
                if raw_str[jdx] == ' ':
                    if has_eoe:
                        continue
                    else:
                        right_context.append(pos_outputs_spc[jdx])
                        has_eoe = True
                else:
                    right_context.append(pos_outputs_spc[jdx])
            if not has_eoe and len(right_context) < window:
                right_context.append(0)
            right_context.extend([0, ] * (window - len(right_context)))
            assert len(right_context) == window

            pos_outputs_contexts.append(\
                    list(reversed(left_context))+[pos_output, ]+right_context)
        assert len(pos_outputs_contexts) == self.get_syllable_count()
        self.pos_tensors = torch.LongTensor((pos_outputs_contexts))

    def match_gazet(self, gazet, voca, context_size, gazet_embed=False):
        """
        gazetteer에서 매핑된 태그를 바탕으로 1-hot 벡터를 만듭니다.
        """

        if not self.gazet_voca:
            self.gazet_voca.update(voca['out'])
            for tag in ['DT0', 'TI0']:
                for bio in ['B', 'I']:
                    self.gazet_voca['%s-%s' % (bio, tag)] = len(self.gazet_voca)

        empty_gazet = [0] * len(self.gazet_voca)
        for tags in gazetteer.match(gazet, self):
            self.gazet_matches.append(copy.deepcopy(empty_gazet))
            for tag in tags:
                self.gazet_matches[-1][self.gazet_voca[tag]] = 1

        # BS * 15
        # BS * 21 * 15 사이즈로 만들어야 한다.
        raw_str = self.raw_str()
        gazet_context = []
        for idx, (char, tag) in enumerate(zip(raw_str, self.gazet_matches)):
            if char == ' ':
                continue
            has_boe = False
            left_context = []
            for jdx in range(idx-1, idx-(context_size*2), -1):
                if jdx < 0 or len(left_context) >= context_size:
                    break
                if raw_str[jdx] == ' ':
                    if has_boe:
                        continue
                    else:
                        left_context.append(self.gazet_matches[jdx])
                        has_boe = True
                else:
                    left_context.append(self.gazet_matches[jdx])
            if not has_boe and len(left_context) < context_size:
                left_context.append(copy.deepcopy(empty_gazet))
            left_context.extend(copy.deepcopy([empty_gazet] * (context_size - len(left_context))))
            assert len(left_context) == context_size
            has_eoe = False
            right_context = []
            for jdx in range(idx+1, idx+(context_size*2)+1):
                if jdx >= len(raw_str) or len(right_context) >= context_size:
                    break
                if raw_str[jdx] == ' ':
                    if has_eoe:
                        continue
                    else:
                        right_context.append(self.gazet_matches[jdx])
                        has_eoe = True
                else:
                    right_context.append(self.gazet_matches[jdx])
            if not has_eoe and len(right_context) < context_size:
                right_context.append(copy.deepcopy(empty_gazet))
            right_context.extend(copy.deepcopy([empty_gazet] * (context_size - len(right_context))))
            assert len(right_context) == context_size

            gazet_context.append(\
                    list(reversed(left_context))+[self.gazet_matches[idx]]+right_context)
        assert len(gazet_context) == self.get_syllable_count()
        if gazet_embed:
            self.gazet_tensors = torch.LongTensor(self.onehot2number(gazet_context))
        else:
            self.gazet_tensors = torch.FloatTensor(gazet_context)

    def to_tensor(self, voca, gazet, context_size, is_phonemes=False, gazet_embed=False):
        """
        문장을 문장에 포함된 문자 갯수 만큼의 배치 크기의 텐서를 생성하여 리턴한다.
        :param  voca:  in/out vocabulary
        :param  gazet: gazztter dictionary
        :param  is_phonemes: 자소단위로 확장할 경우
        :param context_size: context size
        :return:  문자 갯수만큼의 텐서
        """
        if (self.label_tensors is not None and self.context_tensors is not None and
                self.gazet_tensors is not None and self.pos_tensors is not None and
                self.word_tensors is not None):
            return (self.label_tensors, self.context_tensors,
                    self.gazet_tensors, self.pos_tensors, self.word_tensors)

        if not self.gazet_matches:
            self.match_gazet(gazet, voca, context_size, gazet_embed)
        self.to_num_arr(voca, context_size, is_phonemes)
        self.label_tensors = torch.LongTensor(self.label_nums)
        self.context_tensors = torch.LongTensor(self.context_nums)
        self.del_all_except_tensor()
        return (self.label_tensors, self.context_tensors, self.gazet_tensors,
                self.pos_tensors, self.word_tensors)

    def get_word_count(self):
        """
        공백단위로 분리한 어절을 구합니다.
        """
        return len(self.org.split(" "))

    def get_syllable_count(self):
        """
        전체 음절 개수를 리턴합니다.
        """
        return sum([len(x.ne_str.replace(" ", "")) for x in self.named_entity])

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

    def translate_cnn_corpus(self, context_size=10, is_phonemes=False):
        """
        context_size 크기를 갖는 ngram 형태로 학습 코퍼스를 생성합니다.
        :param context_size: context size
        """
        words = [list(word) for word in self.raw_str().split(" ")]
        cnn_corpus = []
        context_tool = TrainContext(is_phonemes)

        cur_idx = 0
        for idx_1, word in enumerate(words):
            marked_list = words[:]
            marked_list[idx_1] = [PADDING['op_wrd']] + word + [PADDING['cl_wrd']]
            marked_text = [syl for word in marked_list for syl in word]
            for _ in word:
                context = context_tool.get_context(cur_idx, marked_text, context_size)
                bio, tag = self.syl2tag[cur_idx+idx_1]
                if tag == OUTSIDE_TAG:
                    label = bio
                else:
                    label = '%s-%s' % (bio, tag)
                #print("%s\t%s %s %s" % (label, left, syl, right))
                cnn_corpus.append((label, context))
                cur_idx += 1
        return cnn_corpus

    @staticmethod
    def _do_action(nes, pred, prev_idx, cur_idx, prev_wrd_idx, cur_wrd_idx):
        """
        현재 태그, 다음태그를 고려해서 액선을 수행한다.
        :param nes:  발견된 NE를 담는 리스트
        :param pred:  모델에 의해 예측된 bio-tag 리스트
        :param prev_idx:  이전 음절 위치
        :param cur_idx:  현재 음절 위치
        :param prev_wrd_idx: 이전음절까지 발생한 단어의 개수
        :param cur_wrd_idx: 현재 음절까지 발생한 단어의 개수
        """
        if isinstance(prev_idx, int):
            if pred[prev_idx] == 'O':
                prev_bio = prev_tag = pred[prev_idx]
            else:
                prev_bio, prev_tag = pred[prev_idx].split('-', 2)
        else:
            prev_bio = 'BOS'

        if isinstance(cur_idx, int):
            if pred[cur_idx] == 'O':
                cur_bio = cur_tag = pred[cur_idx]
            else:
                cur_bio, cur_tag = pred[cur_idx].split('-', 2)
        else:
            cur_bio = 'EOS'

        action = ACTION_TABLE[prev_bio][cur_bio]
        if action == 'OPEN':
            nes.append(NamedEntity('', cur_tag, cur_idx+cur_wrd_idx, -1))
        elif action == 'CLOSE_OPEN':
            if nes and nes[-1].ne_end == -1:
                nes[-1].ne_tag = prev_tag
                nes[-1].ne_end = prev_idx+prev_wrd_idx
            nes.append(NamedEntity('', cur_tag, cur_idx+cur_wrd_idx, -1))
        elif action == 'CLOSE':
            if nes and nes[-1].ne_end == -1:
                nes[-1].ne_end = prev_idx+prev_wrd_idx
        elif action == 'CHECK_TAG':
            # I - I 일때는 태그가 같아야만 의미가 있다.
            # B-PS I-PS I-TI I TI
            #      ~~~~ ~~~~
            #      태그가 달라지면, I-PS에서 닫고, I-TI를 B-TI로 취급한다.
            if prev_tag != cur_tag:
                if nes and nes[-1].ne_end == -1:
                    nes[-1].ne_tag = prev_tag
                    nes[-1].ne_end = prev_idx+prev_wrd_idx
                nes.append(NamedEntity('', cur_tag, cur_idx+cur_wrd_idx, -1))

    def get_named_entity_list(self, predicts, voca=None, measure=None):
        """
        개체명 코퍼스에서 등장하는 NE에 대해서 (시작,끝,태그)의 리스트를 구합니다.
        :param predicts:  모델에서 예측된 클래스 배열
        :param voca:  예측된 클래스를 문자(B-PS)로 변경하기 위한 사전
        :param cnt: Counter()이며, accuracy계산을 위해 필요한 경우 카운트한다.
        """
        words = [list(word) for word in self.raw_str().split(" ")]
        cur_idx = 0
        prev_idx = 'BOS'
        prev_wrd_idx = 0
        nes = []

        assert len(predicts) == self.get_syllable_count()
        if voca:
            pred = [voca['tuo'][x.data[0]] for x in  predicts]
        else:
            pred = predicts # voca가 없으면 ['O', 'B-PS', ...]

        for wrd_idx, word in enumerate(words):
            for syl in word:
                bio, tag = self.syl2tag[cur_idx+wrd_idx]
                label = ("%s-%s") % (bio, tag) if tag != OUTSIDE_TAG else 'O'
                logging.debug("[%d] %s %s : %s", cur_idx+wrd_idx, label, syl, pred[cur_idx])
                if measure:
                    measure.update_accuracy(label, pred[cur_idx])
                self._do_action(nes, pred, prev_idx, cur_idx, prev_wrd_idx, wrd_idx)
                prev_idx = cur_idx
                prev_wrd_idx = wrd_idx
                cur_idx += 1
        self._do_action(nes, pred, prev_idx, 'EOS', prev_wrd_idx, len(words)-1)
        return nes

    def compare_label(self, predicts, voca=None, measure=None):
        """
        예측한 레이블과 현재 문장의 레이블을 비교해서 맞춘 카운트를 계산한다.
        :param predicts:  모델에서 예측된 클래스 배열
        :param voca:  예측된 클래스를 문자(B-PS)로 변경하기 위한 사전
        :param measure:  성능평가를 위한 PerformanceMeasure 클래스
        """
        logging.debug("====== compare label ======")
        logging.debug(self.raw_str())
        if not measure:
            return
        nes = self.get_named_entity_list(predicts, voca, measure)
        measure.update_fscore(self.named_entity, nes)


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

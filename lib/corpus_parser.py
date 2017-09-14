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
import unicodedata
import string
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

    def to_num_arr(self, voca, is_phonemes=False):
        """
        문장을 문장에 포함된 문자 갯수 만큼의 배치 크기의 숫자 배열을 생성하여 리턴한다.
        :param  voca:  in/out vocabulary
        :param  is_phonemes: 자소단위로 확장할 경우
        :return:  문자 갯수만큼의 숫자 배열
        """
        if self.label_nums and self.context_nums:
            return self.label_nums, self.context_nums
        for label, context in self.translate_cnn_corpus(10, is_phonemes):
            self.label_nums.append(voca['out'][label])
            self.context_nums.append([self.in_to_num(voca, char) for char in context.split()])
        return self.label_nums, self.context_nums

    def to_tensor(self, voca, is_phonemes=False):
        """
        문장을 문장에 포함된 문자 갯수 만큼의 배치 크기의 텐서를 생성하여 리턴한다.
        :param  voca:  in/out vocabulary
        :param  is_phonemes: 자소단위로 확장할 경우
        :return:  문자 갯수만큼의 텐서
        """
        if self.label_tensors is not None and self.context_tensors is not None:
            return self.label_tensors, self.context_tensors
        self.to_num_arr(voca, is_phonemes)
        self.label_tensors = torch.LongTensor(self.label_nums)
        self.context_tensors = torch.LongTensor(self.context_nums)
        return self.label_tensors, self.context_tensors

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

    def compare_label(self, predicts, voca=None):
        """
        예측한 레이블과 현재 문장의 레이블을 비교해서 맞춘 카운트를 계산한다.
        :param predicts:  모델에서 예측된 클래스 배열
        :param voca:  예측된 클래스를 문자(B-PS)로 변경하기 위한 사전
        """
        logging.debug(self.raw_str())
        words = [list(word) for word in self.raw_str().split(" ")]
        cur_idx = 0
        prev_idx = 'BOS'
        prev_wrd_idx = 0
        nes = []
        cnt = collections.Counter()

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
                if label == pred[cur_idx]:
                    cnt['correct_char'] += 1
                self._do_action(nes, pred, prev_idx, cur_idx, prev_wrd_idx, wrd_idx)
                prev_idx = cur_idx
                prev_wrd_idx = wrd_idx
                cur_idx += 1
        self._do_action(nes, pred, prev_idx, 'EOS', prev_wrd_idx, len(words)-1)

        gold_ne = set([x.get_ne_pos_tag() for x in self.named_entity\
                    if x.get_ne_pos_tag() is not None])
        pred_ne = set([x.get_ne_pos_tag() for x in nes\
                    if x.get_ne_pos_tag() is not None])
        cnt['total_gold_ne'] += len(gold_ne)
        cnt['total_pred_ne'] += len(pred_ne)
        cnt['match_ne'] += len(gold_ne & pred_ne)
        cnt['total_char'] += len(predicts)

        logging.debug("GOLD = %s\nPRED = %s\nMATCH = %s", gold_ne, pred_ne, cnt)
        return cnt

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

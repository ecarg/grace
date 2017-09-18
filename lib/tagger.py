# -*- coding: utf-8 -*-


"""
Named Entity Tagger using Pytorch model
__author__ = 'Hubert (voidtype@gmail.com)
__copyright__ = 'No copyright. Just copyleft!'
"""

###########
# imports #
###########
import collections
import logging
import torch
import torch.autograd as autograd

import corpus_parser as cp
import models

#########
# types #
#########


class GraceTagger(object):
    """
    학습된 모델을 이용하여 원문을 태깅하는 클리스입니다.
    """
    def __init__(self, model_path):
        self.model = models.Ner.load(model_path)
        self.model.eval()
        self.voca = self.model.voca
        self.gazet = self.model.gazet
        self.window = self.model.window
        self.is_phoneme = self.model.is_phoneme

    def get_predicts(self, sent):
        """
        Sentence 클래스에 저장된 문장에 대해서 태깅을 수행한 결과를 받아옵니다.
        :param sent: Sentnece class
        :return predicts: 태깅된 레이블 시퀀스
        """
        _, contexts, gazet = sent.to_tensor(self.voca, self.gazet, self.window, self.is_phoneme)
        if torch.cuda.is_available():
            contexts = contexts.cuda()
            gazet = gazet.cuda()
        outputs = self.model((autograd.Variable(contexts), autograd.Variable(gazet)))
        _, predicts = outputs.max(1)
        return predicts

    def get_tagged_result(self, sent, predicts):
        """
        Sentance 클래스로 부터 태깅된 원문을 리턴합니다.
        :param sent: Sentance 클래스
        :param predicts: 태깅된 레이블 시퀀스
        """
        raw_line = sent.raw_str()
        nes = sent.get_named_entity_list(predicts, self.voca)
        beg_list = []
        end_dict = {}
        tagged_line = ''
        for item in nes:
            beg, end, tag = item.get_ne_pos_tag()
            beg_list.append(beg)
            end_dict[end] = tag
        for idx, char in enumerate(raw_line):
            if idx in beg_list:
                tagged_line += '<'
            tagged_line += char
            if idx in end_dict:
                tagged_line += ':%s>' % (end_dict[idx])

        logging.debug([x.get_ne_pos_tag() for x in nes])
        return tagged_line

    def tagging(self, line):
        """
        입력 문장을 대상으로 개체명 태깅을 수행합니다.
        :param line: 태깅할 문자열
        :return string: 태깅된 문자열
        """
        tagged_line = ''
        sent = cp.Sentence(line)
        predicts = self.get_predicts(sent)
        tagged_line = self.get_tagged_result(sent, predicts)
        logging.debug("%s\tORIGINAL : %s", self.__class__.__name__, line)
        logging.debug("%s\tTAGGED   : %s", self.__class__.__name__, tagged_line)
        return tagged_line

    @classmethod
    def _calc_f_score(cls, gold_ne, pred_ne, match_ne):
        """
        calculate f-score
        :param  gold_ne:  number of NEs in gold standard
        :param  pred_ne:  number of NEs in predicted
        :param  match_ne:  number of matching NEs
        :return:  f_score
        """
        precision = (match_ne / pred_ne) if pred_ne > 0 else 0.0
        recall = (match_ne / gold_ne) if gold_ne > 0 else 0.0
        if (recall + precision) == 0.0:
            return 0.0
        f_score = 2.0 * recall * precision / (recall + precision)
        return precision, recall, f_score

    def evaluate(self, corpus_path):
        """
        입력문장이 개체명 tagged corpus인 경우 f-score를 계산하여 출력합니다.
        :param corpus_path:  개체명 코퍼스 파일 경로
        :return: accuracy, precision, recall, f_score
        """
        cnt = collections.Counter()
        for sent in cp.sents(corpus_path):
            predicts = self.get_predicts(sent)
            cnt += sent.compare_label(predicts, self.voca)
            tagged = self.get_tagged_result(sent, predicts)
            print(tagged)
        accuracy_char = cnt['correct_char'] / cnt['total_char']
        precision, recall, f_score = self._calc_f_score(cnt['total_gold_ne'],\
                cnt['total_pred_ne'], cnt['match_ne'])
        logging.info('accuracy: %f, f-score: %f (recall = %f, precision = %f)',\
                accuracy_char, f_score, recall, precision)
        return accuracy_char, precision, recall, f_score

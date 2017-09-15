# -*- coding: utf-8 -*-


"""
Named Entity Tagger using Pytorch model
__author__ = 'Hubert (voidtype@gmail.com)
__copyright__ = 'No copyright. Just copyleft!'
"""

###########
# imports #
###########
import os
import collections
import logging
import torch
import torch.autograd as autograd
import corpus_parser as cp

#########
# types #
#########


class GraceTagger(object):
    """
    학습된 모델을 이용하여 원문을 태깅하는 클리스입니다.
    """
    def __init__(self, model_path, gpu_num=None):
        if gpu_num:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_num)

        self.model = torch.load(model_path)
        if torch.cuda.is_available():
            self.model.cuda()

    def tagging(self, line):
        """
        입력 문장을 대상으로 개체명 태깅을 수행합니다.
        :param line: 태깅할 문자열
        :return string: 태깅된 문자열
        """
        sent = cp.Sentence(line)
        _, contexts = sent.to_tensor(self.model.voca, self.model.is_phoneme)
        if torch.cuda.is_available():
            contexts = contexts.cuda()
        outputs = self.model(autograd.Variable(contexts))
        _, predicts = outputs.max(1)
        nes = sent.get_named_entity_list(predicts, self.model.voca)
        beg_list = []
        end_dict = {}
        for item in nes:
            beg, end, tag = item.get_ne_pos_tag()
            beg_list.append(beg)
            end_dict[end] = tag
        tagged_line = ''
        for idx, char in enumerate(line):
            if idx in beg_list:
                tagged_line += '<'
            tagged_line += char
            if idx in end_dict:
                tagged_line += ':%s>' % (end_dict[idx])
        logging.debug(line)
        logging.debug(tagged_line)
        logging.debug([x.get_ne_pos_tag() for x in nes])
        logging.debug(beg_list)
        logging.debug(end_dict)
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
            _, contexts = sent.to_tensor(self.model.voca, self.model.is_phoneme)
            if torch.cuda.is_available():
                contexts = contexts.cuda()
            outputs = self.model(autograd.Variable(contexts))
            _, predicts = outputs.max(1)
            cnt += sent.compare_label(predicts, self.model.voca)
        accuracy_char = cnt['correct_char'] / cnt['total_char']
        precision, recall, f_score = self._calc_f_score(cnt['total_gold_ne'],\
                cnt['total_pred_ne'], cnt['match_ne'])
        print('accuracy: %f, f-score: %f (recall = %f, precision = %f' %\
                (accuracy_char, f_score, recall, precision))
        return accuracy_char, precision, recall, f_score

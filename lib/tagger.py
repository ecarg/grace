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
from pathlib import Path
import torch
from torch.autograd import Variable

import corpus_parser as cp
import models
import data

#########
# types #
#########

class PerformanceMeasure(object):
    """
    성능 측정 관련 기능을 수행하는 클래스 입니다.
    """
    _acc_prefix = 'acc_'
    def __init__(self):
        self.gold = collections.Counter()
        self.pred = collections.Counter()
        self.match = collections.Counter()

        self.accuracy = 0
        self.f_score = 0
        self.precision = 0
        self.recall = 0

    def update_accuracy(self, gold, pred):
        """
        정답 / 예측 레이블을 받아서 카운트 합니다.
        :param gold:  정답 레이블
        :param pred:  예측한 레이블
        """
        if gold == pred:
            self.match['%s%s' % (self._acc_prefix, gold)] += 1
        self.gold['%s%s' % (self._acc_prefix, gold)] += 1

    def update_fscore(self, gold, pred):
        """
        NamedEntity 클래스의 리스트를 받아서 카테고리별로
        잘 예측한 결과를 카운팅 합니다.
        :param gold: 정답 튜플의 리스트
        :param pred: 예측한 튜플의 리스트
        """
        def _to_tuple(ne_class):
            """
            NamedEntity클래스를 tuple로 변경합니다.
            """
            return set([x.get_ne_pos_tag() for x in ne_class\
                    if x.get_ne_pos_tag() is not None])

        gold_ne = _to_tuple(gold)
        pred_ne = _to_tuple(pred)

        for _, _, tag in gold_ne:
            self.gold[tag] += 1

        for _, _, tag in pred_ne:
            self.pred[tag] += 1

        for _, _, tag in gold_ne & pred_ne:
            self.match[tag] += 1

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
            return 0.0, recall, precision
        f_score = 2.0 * recall * precision / (recall + precision)
        return f_score, recall, precision

    def calculate_score_detail(self, display=False):
        """
        카테고리별로 지표를 계산합니다.
        """
        if display:
            logging.info("========== CATEGORY SCORE(Accuracy) ==========")
        for key, val in self.gold.items():
            if key.startswith(self._acc_prefix):
                correct = self.match[key]
                total = val
                if display:
                    logging.info("%s : %f : (correct = %d, total = %d)",
                                 key.replace(self._acc_prefix, ""),
                                 correct / total if total > 0 else 0.0, correct, total)

        if display:
            logging.info("========== CATEGORY SCORE(f-score) ==========")
        for key, val in self.gold.items():
            if not key.startswith(self._acc_prefix):
                gold_ne = val
                pred_ne = self.pred[key]
                match_ne = self.match[key]
                f_score, recall, precision = self._calc_f_score(gold_ne, pred_ne, match_ne)
                if display:
                    logging.info('%s : f-score: %f (recall = %f, precision = %f)'\
                                 ':(gold = %d, pred = %d, match = %d)',\
                                 key, f_score, recall, precision, gold_ne, pred_ne, match_ne)

    def calculate_score(self, display=False):
        """
        지표를 계산합니다.
        """
        def _get_ne_sum(cnt):
            """
            특정 조건을 만족하는 엔트리의 합을 구합니다.
            """
            return sum([v for k, v in cnt.items() if not k.startswith(self._acc_prefix)])

        def _get_acc_sum(cnt):
            """
            accuracy와 관련된 엔트리 합을 구합니다.
            """
            return sum([v for k, v in cnt.items() if k.startswith(self._acc_prefix)])

        total = _get_acc_sum(self.gold)
        correct = _get_acc_sum(self.match)
        self.accuracy = correct / total if total else 0

        gold_ne = _get_ne_sum(self.gold)
        pred_ne = _get_ne_sum(self.pred)
        match_ne = _get_ne_sum(self.match)

        self.f_score, self.recall, self.precision = self._calc_f_score(gold_ne, pred_ne, match_ne)
        if display:
            logging.info("========== TOTAL SCORE ==========")
            logging.info('accuracy: %f, f-score: %f (recall = %f, precision = %f)',\
                    self.accuracy, self.f_score, self.recall, self.precision)

    def get_score(self, display=False):
        """
        지표를 계산해서 출력합니다.
        """
        self.calculate_score_detail(display)
        self.calculate_score(display)
        return self.accuracy, self.f_score

    def display(self):
        """
        지표를 info level로 출력합니다.
        """
        self.calculate_score_detail(True)
        self.calculate_score(True)

class GraceTagger(object):
    """
    학습된 모델을 이용하여 원문을 태깅하는 클리스입니다.
    """
    def __init__(self, model_path, rsc_path):
        self.cfg = torch.load(model_path+"/cfg")
        setattr(self.cfg, 'rsc_dir', Path(rsc_path))
        logging.info(self.cfg)
        rsc = data.load_text(self.cfg, True)
        self.voca = rsc[0]
        self.gazet = rsc[1]
        self.pos_model = rsc[3]
        self.word_model = rsc[4]
        self.model = models.Ner.load(model_path+"/model_param")

    def get_predicts(self, sent):
        """
        Sentence 클래스에 저장된 문장에 대해서 태깅을 수행한 결과를 받아옵니다.
        :param sent: Sentnece class
        :return predicts: 태깅된 레이블 시퀀스
        """
        sent.set_word_feature(self.pos_model, self.word_model, self.cfg.window)
        sent.set_pos_feature(self.pos_model, self.cfg.window)
        _, dev_contexts, dev_gazet, dev_pos, dev_words = \
            sent.to_tensor(self.voca, self.gazet, self.cfg.window,
                           self.cfg.phoneme, self.cfg.gazet_embed)
        dev_contexts = Variable(dev_contexts, volatile=True)
        dev_gazet = Variable(dev_gazet, volatile=True)
        dev_pos = Variable(dev_pos, volatile=True)
        dev_words = Variable(dev_words, volatile=True)
        if torch.cuda.is_available():
            dev_contexts = dev_contexts.cuda()
            dev_gazet = dev_gazet.cuda()
            dev_pos = dev_pos.cuda()
            dev_words = dev_words.cuda()

        outputs = self.model(dev_contexts, dev_gazet, dev_pos, dev_words)
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
        ne_list = []
        is_inside = False
        tagged_line = ''
        named_entity = ''
        for item in nes:
            beg, end, tag = item.get_ne_pos_tag()
            beg_list.append(beg)
            end_dict[end] = tag
        for idx, char in enumerate(raw_line):
            if idx in beg_list:
                tagged_line += '<'
                is_inside = True
            tagged_line += char
            if is_inside:
                named_entity += char
            if idx in end_dict:
                tagged_line += ':%s>' % (end_dict[idx])
                ne_list.append((named_entity, end_dict[idx]))
                named_entity = ''
                is_inside = False

        logging.debug([x.get_ne_pos_tag() for x in nes])
        return tagged_line, ne_list

    def tagging(self, line, fmt_competition=False):
        """
        입력 문장을 대상으로 개체명 태깅을 수행합니다.
        :param line: 태깅할 문자열
        :return string: 태깅된 문자열
        """
        tagged_line = ''
        sent = cp.Sentence(line)
        predicts = self.get_predicts(sent)
        tagged_line, ne_list = self.get_tagged_result(sent, predicts)
        if fmt_competition:
            tagged_line = "; %s\n$%s\n%s\n" % (line, tagged_line,\
                          "\n".join("%s\t%s" % (x[0], x[1]) for x in ne_list))
        logging.debug("%s\tORIGINAL : %s", self.__class__.__name__, line)
        logging.debug("%s\tTAGGED   : %s", self.__class__.__name__, tagged_line)
        logging.debug("%s\tNE LIST : %s", self.__class__.__name__, ne_list)
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
        measure = PerformanceMeasure()
        for sent in cp.sents(corpus_path):
            predicts = self.get_predicts(sent)
            tagged = self.get_tagged_result(sent, predicts)
            sent.compare_label(predicts, self.voca, measure)
            print(tagged)
        measure.display()

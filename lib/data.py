# -*- coding: utf-8 -*-


"""
data processing library
__author__ = 'Jamie (krikit@naver.com)'
__copyright__ = 'No copyright. Just copyleft!'
"""


###########
# imports #
###########
import codecs
import random
from collections import defaultdict
from pathlib import Path
import corpus_parser
import fasttext
import torch


#########
# types #
#########
class NerDataset(object):
    """
    part-of-speech tag dataset
    """
    def __init__(self, fin, voca):
        """
        :param  fin:  input file
        :param  voca:  in/out vocabulary
        """
        self.fin = fin
        self.voca = voca
        self.sents = []
        self.sent_idx = -1
        self._load()

    def _load(self):
        """
        load data file
        """
        for line in self.fin:
            line = line.rstrip('\r\n')
            if not line:
                continue
            self.sents.append(corpus_parser.Sentence(line))

    def __iter__(self):
        self.sent_idx = -1
        random.shuffle(self.sents)
        return self

    def __next__(self):
        self.sent_idx += 1
        if self.sent_idx >= len(self.sents):
            raise StopIteration()
        return self.sents[self.sent_idx]

    def get_syllable_count(self):
        """
        전체 음절의 갯수
        :return:  전체 음절의 갯수
        """
        return sum([sent.get_syllable_count() for sent in self.sents])


#############
# functions #
#############
def load_voca(dir_, is_phonemes=False, cutoff=1):
    """
    load input/output vocabularies
    :param  dir_:  where vocabulary files are
    :param  is_phonemes:  자소단위 사전인 경우
    :param  cutoff: cutoff
    :return:  (in, out) vocabulary pair
    """
    def _load_voca_inner(path, is_in):
        """
        load vocabulary from file
        :param  path:  file path
        :param  is_in: is input voa
        :return:  (vocabulary, inverted vocabulary) pair
        """
        voca = defaultdict(int)    # string to number
        acov = []    # number to string
        if is_in:
            for special in corpus_parser.PADDING.values():
                voca[special] = len(voca)
                acov.append(special)

        for line in codecs.open(path, 'r', encoding='UTF-8'):
            line = line.rstrip('\r\n')
            if not line:
                continue
            # in voca인 경우, 빈도가 있고, cutoff를 적용한다.
            if is_in:
                key, freq = line.split("\t", 1)
                if int(freq) <= cutoff:
                    continue
                line = key
            if line in voca:
                continue
            voca[line] = len(voca)
            acov.append(line)
        return voca, acov

    voca_dic = {}
    for name in ['in', 'out']:
        if is_phonemes:
            voca_path = dir_.joinpath('voca.pho.%s' % name)
        else:
            voca_path = dir_.joinpath('voca.syl.%s' % name)
        voca, acov = _load_voca_inner(voca_path, name == 'in')
        voca_dic[name] = voca
        voca_dic[name[::-1]] = acov
    return voca_dic


def load_data(data_dir, path_pfx, voca):
    """
    load training/dev/test data
    :param data_dir: data directory
    :param  path_pfx:  path prefix
    :param  voca:  vocabulary
    :return:  (dev, test, train) dataset triple
    """
    fins = [(name, codecs.open(data_dir.joinpath(\
        '{}.{}'.format(path_pfx, name)), 'r', encoding='UTF-8')) \
            for name in ['dev', 'test', 'train']]
    return {name: NerDataset(fin, voca) for name, fin in fins}


class Vocabulary(object):
    """Word Vocaublary class"""
    def __init__(self, model_path):
        """
        model_path: pretrained embedding model trained with fasttext
        """
        print('Loading Word2Vec Embedding..', end=' ')

        model_path = Path(model_path)
        self.model = fasttext.load_model(str(model_path))
        self.dim = self.model.dim
        print('Done!')

    def word2vec(self, word, to_tensor=False):
        """word (str) => word vector (list of int)"""
        vec = self.model[word]
        if to_tensor:
            vec = torch.FloatTensor(vec)
        return vec

    def sent2vec(self, sent, to_tensor=True):
        """sent (list of str) => list of word vector [n_words, word dim]"""
        if isinstance(sent, str):
            sent = sent.split()
        sent_tensor = [self.word2vec(word) for word in sent]
        if to_tensor:
            sent_tensor = torch.FloatTensor(sent_tensor)
        return sent_tensor

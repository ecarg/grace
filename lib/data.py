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
import corpus_parser


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
        syl_meta = ['pre', 'suf', 'op_wrd', 'cl_wrd', 'unk']
        pho_meta = ['cho', 'u_cho', 'jung', 'u_jung', 'jong', 'u_jong', 'dig', 'u_dig',
                    'eng', 'u_eng', 'hanja', 'u_hanja', 'symbol', 'u_symbol', 'etc', 'u_etc']
        voca = {}    # string to number
        acov = []    # number to string
        if is_in:
            for meta in pho_meta if is_phonemes else syl_meta:
                val = corpus_parser.PADDING[meta]
                voca[val] = len(voca)
                acov.append(val)

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
            voca_path = '%s/voca.pho.%s' % (dir_, name)
        else:
            voca_path = '%s/voca.syl.%s' % (dir_, name)
        voca, acov = _load_voca_inner(voca_path, name == 'in')
        voca_dic[name] = voca
        voca_dic[name[::-1]] = acov
    return voca_dic


def load_data(path_pfx, voca):
    """
    load training/dev/test data
    :param  path_pfx:  path prefix
    :param  voca:  vocabulary
    :return:  (dev, test, train) dataset triple
    """
    fins = [(name, codecs.open('%s.%s' % (path_pfx, name), 'r', encoding='UTF-8')) \
            for name in ['dev', 'test', 'train']]
    return {name: NerDataset(fin, voca) for name, fin in fins}

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
import gazetteer
import models
import pos_models
from embedder import Embedder


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
        infix = '' if name == 'out' else ('.pho' if is_phonemes else '.syl')
        voca_path = dir_.joinpath('voca%s.%s' % (infix, name))
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

def load_text(cfg, except_data=False):
    """
    load text data
    :param  cfg: config
    :param  except_data: 코퍼스로딩 여부를 지정
    """
    # Load Vocabulary
    # voca['in'] = char
    # voca['out'] = word
    voca = load_voca(cfg.rsc_dir, cfg.phoneme, cfg.cutoff)

    # gazet
    gazet_path = cfg.rsc_dir.joinpath('gazetteer.dic')
    gazet = gazetteer.load(codecs.open(gazet_path, 'r', encoding='UTF-8'))
    pos_path = cfg.rsc_dir.joinpath('pos_tagger.model')
    pos_model = pos_models.PosTagger.load(str(pos_path))
    word_path = cfg.rsc_dir.joinpath('wiki_ko.model.bin')
    word_model = Vocabulary(word_path)
    pos_model.eval()

    # Load Data
    if except_data:
        data_ = None
    else:
        data_ = load_data(cfg.data_dir, cfg.in_pfx, voca)
    return voca, gazet, data_, pos_model, word_model


def build_model(cfg, char_voca, word_voca=None, gazet=None, pos_voca=None):
    """Build Neural Network based Ner model (Embedder + Classifier)"""

    # Build Embedder
    embedder = Embedder(
        window=cfg.window,
        char_voca=char_voca,
        word_voca=word_voca,
        jaso_dim=cfg.jaso_dim,
        char_dim=cfg.char_dim,
        word_dim=cfg.word_dim,
        gazet=gazet,
        gazet_embed=True,
        pos_enc=True,
        phoneme=True,
        pos_voca_size=len(pos_voca),
        pos_dim=cfg.pos_dim)

    print('Total Embedding_size: ', embedder.embed_dim)


    encoder_name, decoder_name = cfg.model_name.lower().split('-')

    # Build Encoder
    if encoder_name == 'fnn5':
        encoder = models.Fnn5(context_len=cfg.context_len,
                              in_dim=embedder.embed_dim,
                              hidden_dim=cfg.hidden_dim)
    elif encoder_name == 'cnn7':
        encoder = models.Cnn7(in_dim=embedder.embed_dim,
                              hidden_dim=cfg.hidden_dim)
    elif encoder_name == 'cnn8':
        encoder = models.Cnn8(context_len=cfg.context_len,
                              in_dim=embedder.embed_dim,
                              hidden_dim=cfg.hidden_dim)
    elif encoder_name in ['gru', 'lstm', 'sru']:
        encoder = models.RnnEncoder(context_len=cfg.context_len,
                                    in_dim=embedder.embed_dim,
                                    out_dim=cfg.hidden_dim,
                                    cell=encoder_name)
    else:
        raise ValueError('unknown model name: %s' % cfg.model_name)

    # Build Decoder
    if decoder_name.lower() == 'fc':
        decoder = models.FCDecoder(in_dim=encoder.out_dim,
                                   hidden_dim=cfg.hidden_dim,
                                   n_tags=cfg.n_tags)
    elif decoder_name in ['gru', 'lstm', 'sru']:
        decoder = models.RnnDecoder(in_dim=encoder.out_dim,
                                    hidden_dim=cfg.hidden_dim,
                                    n_tags=cfg.n_tags,
                                    num_layers=cfg.num_layers,
                                    cell=decoder_name)

    model = models.Ner(embedder, encoder, decoder)

    return model

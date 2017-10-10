#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
train model from data
__author__ = 'Jamie (krikit@naver.com)'
__copyright__ = 'No copyright. Just copyleft!'
"""


#pylint: disable=no-member


###########
# imports #
###########
import base64
import copy
import datetime
import dbm
import logging
import os
import pickle
import shutil
import sys
import time
import zlib

from tensorboardX import SummaryWriter
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.utils.data

from configs import get_config
import data
import tagger


#########
# types #
#########
class CheckPoint(object):
    """
    check point to revert model and optimizer
    """
    def __init__(self, optimizer, model, state_dict):
        self.state_dict = state_dict
        self.model_dump = copy.deepcopy(state_dict)
        self.model_dump['model'] = model.state_dict()
        self.model_dump['optim'] = optimizer.state_dict()
        self.optimizer = optimizer


    def __str__(self):
        lrs = [str(param_group['lr']) for param_group in self.optimizer.param_groups]
        return '{state_dict: %s, lrs: %s}' % (self.state_dict, ', '.join(lrs))

    def save(self, path):
        """
        save current state
        :param  path:  path
        """
        torch.save(self.model_dump, str(path))

    @classmethod
    def load(cls, path):
        """
        load previous state
        :param  path:  path
        :return:  dumped model
        """
        return torch.load(str(path))


class Cache(object):
    """
    disk cache
    """
    def __init__(self, path, is_clean=False):
        """
        :param  path:  file path
        :param  is_clean:  remove previous cache data if True
        """
        self.dbf = dbm.open(path, 'n' if is_clean else 'c')

    def __del__(self):
        if self.dbf:
            self.dbf.close()

    def put(self, key, val):
        """
        put data into db
        :param  key:  key (must be a string)
        :param  val:  value
        """
        self.dbf[key] = base64.b85encode(zlib.compress(pickle.dumps(val)))

    def get(self, key):
        """
        get data from db
        :param  key:  key (must be a string)
        :return:  value
        """
        return pickle.loads(zlib.decompress(base64.b85decode(self.dbf[key])))

    def contains(self, key):
        """
        whether db contains key or not
        :param  key:  key (must be a string)
        :return:  contains key
        """
        return key in self.dbf


#############
# functions #
#############
def run(cfg):    # pylint: disable=too-many-locals,too-many-statements
    """
    run function which is the start point of program
    :param  cfg:  arguments
    """
    # load_text
    voca, gazet, data_, pos_model, word_model = data.load_text(cfg)

    char_voca = voca['in']

    # Build Ner model
    model = data.build_model(cfg, char_voca=char_voca, word_voca=None,
                             gazet=gazet, pos_voca=pos_model.cfg.voca['out'])

    epoch_syl_cnt = data_['train'].get_syllable_count()
    iter_per_epoch = epoch_syl_cnt // cfg.batch_size
    iter_to_rvt = iter_per_epoch * cfg.rvt_epoch

    # Load GPU
    if torch.cuda.is_available():
        model.cuda()

    # Loss / Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = cfg.optimizer(model.parameters())

    losses = []
    accuracies = []
    f_scores = []

    iter_ = 1
    best_iter = 0

    # Remove existing log directory
    if cfg.clean:
        logging.info('==== removing log: %s ====', cfg.model_dir)
        shutil.rmtree(cfg.model_dir)
        time.sleep(3)
    else:
        if cfg.ckpt_path.exists():
            logging.info('==== reverting from check point ====')
            model_dump = CheckPoint.load(cfg.ckpt_path)
            model.load_state_dict(model_dump['model'])
            optimizer.load_state_dict(model_dump['optim'])
            best_iter = model_dump['iter']
            iter_ = best_iter + 1
            losses.append(model_dump['loss'])
            accuracies.append(model_dump['accuracy'])
            f_scores.append(model_dump['f-score'])
            logging.info('---- iter: %dk, loss: %f, accuracy: %f, f-score: %f ----',
                         iter_ // 1000, losses[-1], accuracies[-1], f_scores[-1])
            lrs = [param_group['lr'] for param_group in optimizer.param_groups]
            logging.info('learning rates: %s', ', '.join([str(_) for _ in lrs]))

    # Tensorboard Summary Writer
    sum_wrt = SummaryWriter(cfg.model_dir)

    # loss / accuracy / f-score logging (.tsv)
    log_path = cfg.model_dir.joinpath('log.tsv')
    logf = open(log_path, 'at' if cfg.ckpt_path.exists() else 'wt')
    if os.path.getsize(log_path) == 0:
        print('iter\tloss\taccuracy\tf-score', file=logf)

    train_cache = None
    if cfg.train_cache:
        train_cache_path = cfg.model_dir.joinpath('train_cache.dbm')
        train_cache = Cache(str(train_cache_path), cfg.clean)

    # Main Training Loop
    revert = 0
    one_more_thing = True    # one more change to increase learning rate into 10 times
    batches = []
    torch.save(cfg, str(cfg.cfg_path))
    time1 = datetime.datetime.now()
    while revert <= cfg.rvt_term or one_more_thing:
        for train_sent in data_['train']:
            # Convert to Tensor
            # labels [sentence_len]
            # contexts [sentence_len, 21]
            # gazet [sentence_len, 21, 15]
            train_labels, train_contexts, train_gazet, train_pos, train_words = [None, ] * 5
            if train_cache is None or not train_cache.contains(train_sent.md5):
                train_sent.set_word_feature(pos_model, word_model, cfg.window)
                train_sent.set_pos_feature(pos_model, cfg.window)
                train_labels, train_contexts, train_gazet, train_pos, train_words = \
                    train_sent.to_tensor(voca, gazet, cfg.window, cfg.phoneme, cfg.gazet_embed)
                if train_cache is not None:
                    train_cache.put(train_sent.md5, (train_labels, train_contexts, train_gazet,
                                                     train_pos, train_words))
                    train_sent.del_tensor()
            else:
                train_labels, train_contexts, train_gazet, train_pos, train_words = \
                    train_cache.get(train_sent.md5)
                train_sent.del_all_except_tensor()
                train_sent.del_tensor()

            # Convert to Variable
            train_labels = Variable(train_labels)
            train_contexts = Variable(train_contexts)
            train_gazet = Variable(train_gazet)
            train_pos = Variable(train_pos, requires_grad=False)
            train_words = Variable(train_words, requires_grad=False)

            # Load on GPU
            if torch.cuda.is_available():
                train_labels = train_labels.cuda()
                train_contexts = train_contexts.cuda()
                train_gazet = train_gazet.cuda()
                train_pos = train_pos.cuda()
                train_words = train_words.cuda()

            # Reset Gradient
            optimizer.zero_grad()

            # Training mode (updates/dropout/batchnorm)
            model.train()

            # import ipdb; ipdb.set_trace()

            # Forward Prop
            outputs = model(train_contexts, train_gazet, train_pos, train_words)

            batches.append((train_labels, outputs))
            if sum([batch[0].size(0) for batch in batches]) < cfg.batch_size:
                continue
            batch_label = torch.cat([x[0] for x in batches], 0)
            batch_output = torch.cat([x[1] for x in batches], 0)
            batches = []

            # Backprop
            loss = criterion(batch_output, batch_label)
            loss.backward()
            optimizer.step()

            # Validation
            if iter_ % 1000 == 0:
                time2 = datetime.datetime.now()
                elapsed = time2 - time1
                time1 = time2
                measure = tagger.PerformanceMeasure()
                # Freeze parameters
                model.eval()

                # Calculate loss
                losses.append(loss.data[0])
                for dev_sent in data_['dev']:
                    # Convert to CUDA Variable
                    dev_sent.set_word_feature(pos_model, word_model, cfg.window)
                    dev_sent.set_pos_feature(pos_model, cfg.window)
                    _, dev_contexts, dev_gazet, dev_pos, dev_words = \
                        dev_sent.to_tensor(voca, gazet, cfg.window, cfg.phoneme, cfg.gazet_embed)
                    dev_contexts = Variable(dev_contexts, volatile=True)
                    dev_gazet = Variable(dev_gazet, volatile=True)
                    dev_pos = Variable(dev_pos, volatile=True)
                    dev_words = Variable(dev_words, volatile=True)
                    if torch.cuda.is_available():
                        dev_contexts = dev_contexts.cuda()
                        dev_gazet = dev_gazet.cuda()
                        dev_pos = dev_pos.cuda()
                        dev_words = dev_words.cuda()

                    outputs = model(dev_contexts, dev_gazet, dev_pos, dev_words)

                    _, predicts = outputs.max(1)
                    dev_sent.compare_label(predicts, voca, measure)

                accuracy, f_score = measure.get_score()
                print(file=sys.stderr)
                sys.stderr.flush()
                if not f_scores or f_score > max(f_scores):
                    logging.info('==== writing best model: %f ====', f_score)
                    model.save(cfg.model_param_path)
                    check_point = CheckPoint(optimizer, model,
                                             {'iter': iter_, 'loss': loss.data[0],
                                              'accuracy': accuracy, 'f-score': f_score})
                    check_point.save(cfg.ckpt_path)
                    logging.info('check point: %s', check_point)
                    best_iter = iter_
                    revert = 0
                    one_more_thing = True
                accuracies.append(accuracy)
                f_scores.append(f_score)
                logging.info('---- elapsed: %s, iter: %dk, loss: %f, accuracy: %f, f-score: %f'
                             ' (max: %f) ----', elapsed, iter_ // 1000, losses[-1], accuracy,
                             f_score, max(f_scores))

                if cfg.model_dir.exists():
                    sum_wrt.add_scalar('loss', losses[-1], iter_ // 1000)
                    sum_wrt.add_scalar('accuracy', accuracy, iter_ // 1000)
                    sum_wrt.add_scalar('f-score', f_score, iter_ // 1000)
                    print('{}\t{}\t{}\t{}'.format(iter_ // 1000, losses[-1], accuracy,
                                                  f_score), file=logf)
                    logf.flush()

                # revert policy
                if (iter_ - best_iter) > iter_to_rvt:
                    revert += 1
                    logging.info('==== revert to iter: %dk, revert count: %d ====',
                                 best_iter // 1000, revert)
                    model_dump = CheckPoint.load(cfg.ckpt_path)
                    model.load_state_dict(model_dump['model'])
                    optimizer.load_state_dict(model_dump['optim'])
                    lrs = []
                    for param_group in optimizer.param_groups:
                        param_group['lr'] *= (0.9 if one_more_thing else 0.8) ** revert
                        lrs.append(param_group['lr'])
                    best_iter = iter_
                    logging.info('learning rates: %s', ', '.join([str(_) for _ in lrs]))
            elif iter_ % 100 == 0:
                print('.', end='', file=sys.stderr)
                sys.stderr.flush()

            iter_ += 1
        if revert > cfg.rvt_term and one_more_thing:
            logging.info('==== one more thing, revert to iter: %dk ====', best_iter // 1000)
            model_dump = CheckPoint.load(cfg.ckpt_path)
            model.load_state_dict(model_dump['model'])
            optimizer.load_state_dict(model_dump['optim'])
            lrs = []
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 10.0
                lrs.append(param_group['lr'])
            best_iter = iter_
            revert = 0
            one_more_thing = False
            logging.info('learning rates: %s', ', '.join([str(_) for _ in lrs]))


########
# main #
########
def main():
    """
    Main training function
    """
    cfg = get_config(is_train=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.gpu_num)
    print(cfg)
    run(cfg)

if __name__ == '__main__':
    main()

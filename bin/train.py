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
import argparse
import codecs
import copy
import logging
import os
import shutil
import sys
import time

from tensorboardX import SummaryWriter
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.utils.data

import data
import models
import gazetteer
import pos_models
import tagger


###########
# options #
###########
GPU_NUM = 0
WINDOW = 10
EMBED_DIM = 50
RVT_EPOCH = 2
RVT_TERM = 10
BATCH_SIZE = 100


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
        torch.save(self.model_dump, path)

    @classmethod
    def load(cls, path):
        """
        load previous state
        :param  path:  path
        :return:  dumped model
        """
        return torch.load(path)


#############
# functions #
#############
def _make_model_id(args):
    """
    make model ID string
    :return:  model ID
    """
    model_ids = [args.model_name, ]
    model_ids.append('cut%d' % args.cutoff)
    model_ids.append('pho' if args.phoneme else 'chr')
    model_ids.append('w%d' % args.window)
    model_ids.append('e%d' % args.embed_dim)
    model_ids.append('gzte' if args.gazet_embed else 'gzt1')
    model_ids.append('pe%d' % (1 if args.pos_enc else 0))
    model_ids.append('re%d' % args.rvt_epoch)
    model_ids.append('rt%d' % args.rvt_term)
    model_ids.append('bs%d' % args.batch_size)
    return '.'.join(model_ids)


def _init(args):
    """
    initialize dictionaries and model
    :param  args:  arguments
    :return:  (vocabulary, gazetteer, data_, model) tuple
    """
    voca = data.load_voca(args.rsc_dir, args.phoneme, args.cutoff)
    gazet = gazetteer.load(codecs.open("%s/gazetteer.dic" % args.rsc_dir, 'r', encoding='UTF-8'))
    pos_model = pos_models.PosTagger.load('%s/pos_tagger.model' % args.rsc_dir)
    pos_model.eval()

    # Load Data
    data_ = data.load_data(args.in_pfx, voca)

    # Build Model
    if args.model_name.lower() == 'fnn5':
        hidden_dim = ((2 * args.window + 1) *\
                (args.embed_dim + 15 + 20)+ len(voca['out'])) // 2
        model = models.Fnn5(args.window, voca, gazet,
                            args.embed_dim, hidden_dim,
                            args.phoneme, args.gazet_embed, args.pos_enc)
    elif args.model_name.lower() == 'cnn7':
        concat_dim = args.embed_dim + 20 + len(voca['out']) + 4
        hidden_dim = (concat_dim * 4 + len(voca['out'])) // 2
        model = models.Cnn7(args.window, voca, gazet,
                            args.embed_dim, hidden_dim,
                            args.phoneme, args.gazet_embed, args.pos_enc)
    elif args.model_name.lower() == 'rnn1':
        rnn_dim = 100
        hidden_dim = rnn_dim * 2 + len(voca['out']) // 2
        model = models.Rnn1(args.window, voca, gazet,
                            args.embed_dim, rnn_dim, hidden_dim,
                            args.phoneme, args.gazet_embed, args.pos_enc)
    elif args.model_name.lower() == 'rnn2':
        rnn_dim = 100
        hidden_dim = rnn_dim * 2 + len(voca['out']) // 2
        model = models.Rnn2(args.window, voca, gazet,
                            args.embed_dim, rnn_dim, hidden_dim,
                            args.phoneme, args.gazet_embed, args.pos_enc)
    else:
        raise ValueError('unknown model name: %s' % args.model_name)
    return voca, gazet, pos_model, data_, model


def run(args):    # pylint: disable=too-many-locals,too-many-statements
    """
    run function which is the start point of program
    :param  args:  arguments
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_num)

    voca, gazet, pos_model, data_, model = _init(args)

    epoch_syl_cnt = data_['train'].get_syllable_count()
    iter_per_epoch = epoch_syl_cnt // args.batch_size
    iter_to_rvt = iter_per_epoch * args.rvt_epoch

    # Load GPU
    if torch.cuda.is_available():
        model.cuda()

    # Loss / Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    losses = []
    accuracies = []
    f_scores = []

    iter_ = 1
    best_iter = 0
    has_check_point = os.path.exists(args.output) and os.path.exists('%s.chk' % args.output)
    if has_check_point:
        logging.info('==== reverting from check point ====')
        model_dump = CheckPoint.load('%s.chk' % args.output)
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

    model_id = _make_model_id(args)
    logf = None
    if args.logdir:
        model_dir = '%s/%s' % (args.logdir, model_id)
        if os.path.exists(model_dir) and not has_check_point:
            logging.info('==== removing log: %s ====', model_dir)
            shutil.rmtree(model_dir)
            time.sleep(3)
        sum_wrt = SummaryWriter(model_dir)
        log_path = '%s/%s.tsv' % (args.logdir, model_id)
        logf = open(log_path, 'at' if has_check_point else 'wt')
        if os.path.getsize(log_path) == 0:
            print('iter\tloss\taccuracy\tf-score', file=logf)

    revert = 0
    one_more_thing = True    # one more change to increase learning rate into 10 times
    batches = []
    while revert <= args.rvt_term or one_more_thing:
        for train_sent in data_['train']:
            # Convert to CUDA Variable
            train_sent.set_pos_feature(pos_model, args.window)
            train_labels, train_contexts, train_gazet, train_pos = \
                train_sent.to_tensor(voca, gazet, args.window, args.phoneme, args.gazet_embed)
            train_labels = Variable(train_labels)
            train_contexts = Variable(train_contexts)
            train_gazet = Variable(train_gazet)
            train_pos = Variable(train_pos)
            if torch.cuda.is_available():
                train_labels = train_labels.cuda()
                train_contexts = train_contexts.cuda()
                train_gazet = train_gazet.cuda()
                train_pos = train_pos.cuda()

            # Reset Gradient
            optimizer.zero_grad()

            # Forwardprop / Backprop
            model.train()

            outputs = model((train_contexts, train_gazet, train_pos))
            batches.append((train_labels, outputs))
            batch_size = sum([batch[0].size(0) for batch in batches])
            if batch_size < args.batch_size:
                continue
            batch_label = torch.cat([x[0] for x in batches], 0)
            batch_output = torch.cat([x[1] for x in batches], 0)
            batches = []

            loss = criterion(batch_output, batch_label)
            loss.backward()
            optimizer.step()

            # Validation
            if iter_ % 1000 == 0:
                measure = tagger.PerformanceMeasure()
                # Freeze parameters
                model.eval()

                # Calculate loss
                losses.append(loss.data[0])
                for dev_sent in data_['dev']:
                    # Convert to CUDA Variable
                    dev_sent.set_pos_feature(pos_model, args.window)
                    _, dev_contexts, dev_gazet, dev_pos = \
                        dev_sent.to_tensor(voca, gazet, args.window, args.phoneme, args.gazet_embed)
                    dev_contexts = Variable(dev_contexts, volatile=True)
                    dev_gazet = Variable(dev_gazet, volatile=True)
                    dev_pos = Variable(dev_pos, volatile=True)
                    if torch.cuda.is_available():
                        dev_contexts = dev_contexts.cuda()
                        dev_gazet = dev_gazet.cuda()
                        dev_pos = dev_pos.cuda()

                    outputs = model((dev_contexts, dev_gazet, dev_pos))
                    _, predicts = outputs.max(1)
                    dev_sent.compare_label(predicts, voca, measure)

                accuracy, f_score = measure.get_score()
                print(file=sys.stderr)
                sys.stderr.flush()
                if not f_scores or f_score > max(f_scores):
                    logging.info('==== writing best model: %f ====', f_score)
                    model.save(args.output)
                    check_point = CheckPoint(optimizer, model,
                                             {'iter': iter_, 'loss': loss.data[0],
                                              'accuracy': accuracy, 'f-score': f_score})
                    check_point.save('%s.chk' % args.output)
                    logging.info('check point: %s', check_point)
                    best_iter = iter_
                    revert = 0
                    one_more_thing = True
                accuracies.append(accuracy)
                f_scores.append(f_score)
                logging.info('---- iter: %dk, loss: %f, accuracy: %f, f-score: %f (max: %r) ----',
                             iter_ // 1000, losses[-1], accuracy, f_score, max(f_scores))
                if args.logdir:
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
                    model_dump = CheckPoint.load('%s.chk' % args.output)
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
        if revert > args.rvt_term and one_more_thing:
            logging.info('==== one more thing, revert to iter: %dk ====', best_iter // 1000)
            model_dump = CheckPoint.load('%s.chk' % args.output)
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
    main function processes only argument parsing
    """
    parser = argparse.ArgumentParser(description='train model from data')
    parser.add_argument('-r', '--rsc-dir', help='resource directory', metavar='DIR', required=True)
    parser.add_argument('-p', '--in-pfx', help='input data prefix', metavar='NAME', required=True)
    parser.add_argument('-m', '--model-name', help='model name', metavar='NAME', required=True)
    parser.add_argument('-o', '--output', help='model output file', metavar='FILE', required=True)
    parser.add_argument('--logdir', help='tensorboard log dir <default: ./logdir>',
                        metavar='DIR', default='./logdir')
    parser.add_argument('--window', help='left/right character window length <default: %d>' % \
                                          WINDOW, metavar='INT', type=int, default=WINDOW)
    parser.add_argument('--embed-dim', help='embedding dimension <default: %d>' % EMBED_DIM,
                        metavar='INT', type=int, default=EMBED_DIM)
    parser.add_argument('--gpu-num', help='GPU number to use <default: %d>' % GPU_NUM,
                        metavar='INT', type=int, default=GPU_NUM)
    parser.add_argument('--phoneme', help='expand phonemes context', action='store_true')
    parser.add_argument('--pos-enc', help='add positional encoding',
                        action='store_true', default=False)
    parser.add_argument('--gazet-embed', help='gazetteer type', action='store_true',
                        default=False)
    parser.add_argument('--cutoff', help='cutoff', action='store',\
                        type=int, metavar='NUM', default=5)
    parser.add_argument('--rvt-epoch', help='최대치 파라미터로 되돌아갈 epoch 횟수 <default: %d>' % \
                                            RVT_EPOCH, type=int, metavar='NUM', default=RVT_EPOCH)
    parser.add_argument('--rvt-term', help='파라미터로 되돌아갈 최대 횟수 <default: %d>' % \
                                           RVT_TERM, type=int, metavar='NUM', default=RVT_TERM)
    parser.add_argument('--batch-size', help='batch size <default: %d>' % BATCH_SIZE, metavar='INT',
                        type=int, default=BATCH_SIZE)
    parser.add_argument('--debug', help='enable debug', action='store_true')
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    run(args)


if __name__ == '__main__':
    main()

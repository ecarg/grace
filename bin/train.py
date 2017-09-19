#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
train model from data
__author__ = 'Jamie (krikit@naver.com)'
__copyright__ = 'No copyright. Just copyleft!'
"""


###########
# imports #
###########
import argparse
import codecs
from collections import Counter
import logging
import os
import sys

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.utils.data

import data
import models
import gazetteer


###########
# options #
###########
WINDOW = 10
EMBED_DIM = 50
GPU_NUM = 0
BATCH_SIZE = 100
EPOCH_NUM = 100


#############
# functions #
#############
def _calc_f_score(gold_ne, pred_ne, match_ne):
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
    return 2.0 * recall * precision / (recall + precision)


def run(args):    # pylint: disable=too-many-locals,too-many-statements
    """
    run function which is the start point of program
    :param  args:  arguments
    """
    voca = data.load_voca(args.rsc_dir, args.phoneme, args.cutoff)
    gazet = gazetteer.load(codecs.open("%s/gazetteer.dic" % args.rsc_dir, 'r', encoding='UTF-8'))

    # Build Model
    if args.model_name.lower() == 'fnn3':
        hidden_dim = (2 * args.window + 1) * args.embed_dim + len(voca['out'])
        model = models.Fnn3(args.window, voca, gazet, args.embed_dim, hidden_dim, args.phoneme)
    elif args.model_name.lower() == 'fnn4':
        hidden_dim = (2 * args.window + 1) * (args.embed_dim + len(voca['out'])+4)+ len(voca['out'])
        model = models.Fnn4(args.window, voca, gazet, args.embed_dim, hidden_dim, args.phoneme)
    elif args.model_name.lower() == 'cnn3':
        hidden_dim = 1000
        model = models.Cnn6(args.window, voca, gazet, args.embed_dim, hidden_dim, args.phoneme)
    elif args.model_name.lower() == 'cnn4':
        hidden_dim = 2000
        model = models.Cnn4(args.window, voca, gazet, args.embed_dim, hidden_dim, args.phoneme)
    elif args.model_name.lower() == 'cnn5':
        hidden_dim = (args.embed_dim + len(voca['out'])) // 2
        model = models.Cnn5(args.window, voca, gazet, args.embed_dim, hidden_dim, args.phoneme)
    elif args.model_name.lower() == 'cnn6':
        hidden_dim = (args.embed_dim * 4 + len(voca['out'])) // 2
        model = models.Cnn6(args.window, voca, gazet, args.embed_dim, hidden_dim, args.phoneme)
    elif args.model_name.lower() == 'cnn7':
        concat_dim = args.embed_dim + len(voca['out']) + 4
        hidden_dim = (concat_dim * 4 + len(voca['out'])) // 2
        model = models.Cnn7(args.window, voca, gazet, args.embed_dim, hidden_dim, args.phoneme)

    # Load Data
    data_ = data.load_data(args.in_pfx, voca)

    # Load GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_num)
    if torch.cuda.is_available():
        model.cuda()

    # Loss / Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    if args.log:
        print('iter\tloss\taccuracy\tf-score', file=args.log)
    losses = []
    accuracies = []
    f_scores = []

    iter_ = 0
    for epoch in range(args.epoch_num):
        for train_sent in data_['train']:
            # Convert to CUDA Variable
            train_labels, train_contexts, train_gazet = \
                train_sent.to_tensor(voca, gazet, args.window, args.phoneme)
            train_labels = Variable(train_labels)
            train_contexts = Variable(train_contexts)
            train_gazet = Variable(train_gazet)
            if torch.cuda.is_available():
                train_labels = train_labels.cuda()
                train_contexts = train_contexts.cuda()
                train_gazet = train_gazet.cuda()

            # Reset Gradient
            optimizer.zero_grad()

            # Forwardprop / Backprop
            model.train()
            outputs = model((train_contexts, train_gazet))
            loss = criterion(outputs, train_labels)
            loss.backward()
            optimizer.step()
            iter_ += 1

            # Validation
            if iter_ % 1000 == 0:
                # Freeze parameters
                model.eval()

                # Calculate loss
                losses.append(loss.data[0])
                cnt = Counter()
                for dev_sent in data_['dev']:
                    # Convert to CUDA Variable
                    _, dev_contexts, dev_gazet =\
                            dev_sent.to_tensor(voca, gazet, args.window, args.phoneme)
                    dev_contexts = Variable(dev_contexts, volatile=True)
                    dev_gazet = Variable(dev_gazet, volatile=True)

                    if torch.cuda.is_available():
                        dev_contexts = dev_contexts.cuda()
                        dev_gazet = dev_gazet.cuda()

                    outputs = model((dev_contexts, dev_gazet))
                    _, predicts = outputs.max(1)
                    cnt += dev_sent.compare_label(predicts, voca)

                # Calculate accuracy / f-score
                accuracy_char = cnt['correct_char'] / cnt['total_char']
                f_score = _calc_f_score(cnt['total_gold_ne'], cnt['total_pred_ne'], cnt['match_ne'])
                print(file=sys.stderr)
                sys.stderr.flush()
                if not f_scores or f_score > max(f_scores):
                    logging.info('writing best model..')
                    model.save(args.output)
                accuracies.append(accuracy_char)
                f_scores.append(f_score)
                logging.info('epoch: %d, iter: %dk, loss: %f, accuracy: %f, f-score: %f (max: %r)',
                             epoch, iter_ // 1000, losses[-1], accuracy_char, f_score,
                             max(f_scores))
                if args.log:
                    print('{}\t{}\t{}\t{}'.format(iter_ // 1000, losses[-1], accuracy_char,
                                                  f_score), file=args.log)
                    args.log.flush()
            elif iter_ % 100 == 0:
                print('.', end='', file=sys.stderr)
                sys.stderr.flush()


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
    parser.add_argument('--log', help='loss and accuracy log file', metavar='FILE',
                        type=argparse.FileType('wt'))
    parser.add_argument('--window', help='left/right character window length <default: %d>' % \
                                          WINDOW, metavar='INT', type=int, default=WINDOW)
    parser.add_argument('--embed-dim', help='embedding dimension <default: %d>' % EMBED_DIM,
                        metavar='INT', type=int, default=EMBED_DIM)
    parser.add_argument('--gpu-num', help='GPU number to use <default: %d>' % GPU_NUM,
                        metavar='INT', type=int, default=GPU_NUM)
    parser.add_argument('--batch-size', help='batch size <default: %d>' % BATCH_SIZE, metavar='INT',
                        type=int, default=BATCH_SIZE)
    parser.add_argument('--epoch-num', help='epoch number <default: %d>' % EPOCH_NUM, metavar='INT',
                        type=int, default=EPOCH_NUM)
    parser.add_argument('--phoneme', help='expand phonemes context', action='store_true')
    parser.add_argument('--cutoff', help='cutoff', action='store',\
            type=int, metavar="int", default=5)
    parser.add_argument('--debug', help='enable debug', action='store_true')
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    run(args)


if __name__ == '__main__':
    main()

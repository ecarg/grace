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
from collections import Counter
import logging
import os
import sys

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.utils.data

import corpus_parser
import data
import model


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
def _count_match(cnt, gold_sent, predicts, voca):
    """
    f-score(recall/precision) 측정을 위한 개체명 단위 카운팅을 수행
    :param  cnt:  Counter 객체
    :param  gold_sent:  정답 corpus_parser.Sentence 객체
    :param  predicts:  예측 값 (문장 내 문자의 갯수 만큼의 출력 레이블의 숫자)
    :param  voca:  in/out vocabulary
    :return:
    """
    pred_pairs = []
    for pred, (_, context) in zip(predicts, gold_sent.translate_cnn_corpus(10)):
        pred_pairs.append((voca['tuo'][pred.data[0]], context))

    pred_sent = corpus_parser.Sentence.restore(pred_pairs)
    gold_ne = set([x.get_ne_pos_tag() for x in gold_sent.named_entity \
                   if x.get_ne_pos_tag() is not None])
    pred_ne = set([x.get_ne_pos_tag() for x in pred_sent.named_entity \
                   if x.get_ne_pos_tag() is not None])

    cnt['total_gold_ne'] += len(gold_ne)
    cnt['total_pred_ne'] += len(pred_ne)
    cnt['match_ne'] += len(gold_ne & pred_ne)


def run(args):    # pylint: disable=too-many-locals,too-many-statements
    """
    run function which is the start point of program
    :param  args:  arguments
    """
    voca = data.load_voca(args.rsc_dir)
    if args.model_name.lower() == 'fnn':
        hidden_dim = (2 * args.window + 1) * args.embed_dim + len(voca['out'])
        model_ = model.Fnn(args.window, voca, args.embed_dim, hidden_dim)
    elif args.model_name.lower() == 'cnn':
        hidden_dim = ((2 + 3 + 3 + 4 + 1) * args.embed_dim * 4 + len(voca['out'])) // 2
        model_ = model.Cnn(args.window, voca, args.embed_dim, hidden_dim)

    data_ = data.load_data(args.in_pfx, voca)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_num)
    if torch.cuda.is_available():
        model_.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_.parameters())

    if args.log:
        print('iter\tloss\taccuracy', file=args.log)
    losses = []
    accuracies = []
    f_scores = []

    iter_ = 0
    for epoch in range(args.epoch_num):
        for train_sent in data_['train']:
            train_labels, train_contexts = train_sent.to_tensor(voca)
            if torch.cuda.is_available():
                train_labels = train_labels.cuda()
                train_contexts = train_contexts.cuda()
            optimizer.zero_grad()
            model_.is_training = True
            outputs = model_(autograd.Variable(train_contexts))
            loss = criterion(outputs, autograd.Variable(train_labels))
            loss.backward()
            optimizer.step()
            iter_ += 1
            if iter_ % 1000 == 0:
                # loss and accuracy
                losses.append(loss.data[0])
                cnt = Counter()
                for dev_sent in data_['dev']:
                    dev_labels, dev_contexts = dev_sent.to_tensor(voca)
                    if torch.cuda.is_available():
                        dev_labels = dev_labels.cuda()
                        dev_contexts = dev_contexts.cuda()
                    model_.is_training = False
                    outputs = model_(autograd.Variable(dev_contexts))
                    _, predicts = outputs.max(1)
                    subtotal = dev_labels.size(0)
                    subcorrect = (predicts.data == dev_labels).sum()
                    cnt['correct_char'] += subcorrect
                    cnt['total_char'] += subtotal
                    _count_match(cnt, dev_sent, predicts, voca)
                accuracy_char = 100.0 * cnt['correct_char'] / cnt['total_char']
                recall = cnt['match_ne'] / cnt['total_gold_ne']
                precision = cnt['match_ne'] / cnt['total_pred_ne']
                f_score = 2.0 * recall * precision / (recall + precision)
                print(file=sys.stderr)
                sys.stderr.flush()
                if not f_scores or f_score > max(f_scores):
                    logging.info('writing best model..')
                    torch.save(model_.state_dict(), args.output)
                accuracies.append(accuracy_char)
                f_scores.append(f_score)
                logging.info('epoch: %d, iter: %dk, loss: %f, accuracy: %f, f-score: %f (max: %r)',
                             epoch, iter_ // 1000, losses[-1], accuracy_char, f_score,
                             max(f_scores))
                if args.log:
                    print('{}\t{}\t{}\t{}'.format(iter_ // 1000, loss, accuracy_char, f_score),
                          file=args.log)
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
    parser.add_argument('--debug', help='enable debug', action='store_true')
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    run(args)


if __name__ == '__main__':
    main()

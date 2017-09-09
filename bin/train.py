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
import logging
import os
import pickle
import sys

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

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
def _accuracy(model_, labels, contexts):
    """
    evaluate prediction accuracy
    :param  model_:  model
    :param  labels:  labels
    :param  contexts:  contexts
    :return:  accuracy
    """
    if torch.cuda.is_available():
        labels = labels.cuda()
        contexts = contexts.cuda()
    model.is_training = False
    outputs = model_(autograd.Variable(contexts))
    _, predicts = F.softmax(outputs).max(1)
    total = labels.size(0)
    correct = (predicts.data == labels).sum()
    return 100.0 * correct / total


def run(args):    # pylint: disable=too-many-locals,too-many-statements
    """
    run function which is the start point of program
    :param  args:  arguments
    """
    voca = data.load_voca(args.rsc_dir)
    if args.model.lower() == 'fnn':
        hidden_dim = (2 * args.window + 1) * args.embed_dim + len(voca['out'])
        model_ = model.Fnn(args.window, voca, args.embed_dim, hidden_dim)
    elif args.model.lower() == 'cnn':
        hidden_dim = ((2 + 3 + 3 + 4 + 1) * args.embed_dim * 4 + len(voca['out'])) // 2
        model_ = model.Cnn(args.window, voca, args.embed_dim, hidden_dim)

    data_ = data.load_data('%s/%s' % (args.in_dir, args.in_pfx), voca)
    train_data = torch.utils.data.DataLoader(dataset=data_['train'], batch_size=args.batch_size,
                                             shuffle=True)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_num)
    if torch.cuda.is_available():
        model_.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_.parameters())

    if args.log:
        print('loss\taccuracy', file=args.log)
    losses = []
    accuracies = []

    iter_ = 0
    for epoch in range(args.epoch_num):
        for labels, contexts in train_data:
            if torch.cuda.is_available():
                labels = labels.cuda()
                contexts = contexts.cuda()
            optimizer.zero_grad()
            model.is_training = True
            outputs = model_(autograd.Variable(contexts))
            loss = criterion(outputs, autograd.Variable(labels))
            loss.backward()
            optimizer.step()
            iter_ += 1
            if iter_ % 1000 == 0:
                # loss and accuracy
                losses.append(loss.data[0])
                labels, contexts = data_['dev'].all()
                if torch.cuda.is_available():
                    labels = labels.cuda()
                    contexts = contexts.cuda()
                accuracy = _accuracy(model_, labels, contexts)
                accuracies.append(accuracy)
                print(file=sys.stderr)
                sys.stderr.flush()
                logging.info('epoch: %d, iter: %d, loss: %f, accuracy: %f (max: %f)',
                             epoch, iter_, losses[-1], accuracies[-1], max(accuracies))
                if iter_ > 10000 and accuracy == max(accuracies):
                    logging.info('writing best model..')
                    with open(args.output, 'wb') as fout:
                        pickle.dump(model_, fout, protocol=pickle.HIGHEST_PROTOCOL)
                if args.log:
                    print('{}\t{}'.format(losses[-1], accuracies[-1]), file=args.log)
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
    parser.add_argument('-i', '--in-dir', help='input directory', metavar='DIR', required=True)
    parser.add_argument('-p', '--in-pfx', help='input data prefix', metavar='NAME', required=True)
    parser.add_argument('-m', '--model', help='model name', metavar='NAME', required=True)
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

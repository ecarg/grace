# -*- coding: utf-8 -*-


"""
Pytorch models
__author__ = 'Jamie (krikit@naver.com)'
__copyright__ = 'No copyright. Just copyleft!'
"""


###########
# imports #
###########
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F


#########
# types #
#########
class Ner(nn.Module):
    """
    part-of-speech tagger pytorch model
    """
    def __init__(self, window, voca):
        """
        :param  window:  left/right window size from current character
        :param  voca:  vocabulary
        """
        super().__init__()
        self.is_training = True    # is training phase or not (use drop-out at training)
        self.window = window
        self.voca = voca

    def forward(self, *inputs):
        raise NotImplementedError


class Fnn(Ner):
    """
    feed-forward neural network based part-of-speech tagger
    """
    def __init__(self, window, voca, embed_dim, hidden_dim):
        """
        :param  window:  left/right window size from current character
        :param  voca:  vocabulary
        :param  embed_dim:  character embedding dimension
        :param  hidden_dim:  hidden layer dimension
        """
        super().__init__(window, voca)
        context_len = 2 * window + 1
        self.embedding = nn.Embedding(len(voca['in']), embed_dim)
        self.hidden = autograd.Variable(torch.zeros(1, 1, hidden_dim))
        self.relu = nn.ReLU()
        self.embeds2hidden = nn.Linear(context_len * embed_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, len(voca['out']))

    def forward(self, contexts):    # pylint: disable=arguments-differ
        """
        forward path
        :param  contexts:  batch size list of character and context
        :return:  output score
        """
        embeds = self.embedding(contexts)
        hidden_out = self.embeds2hidden(embeds.view(len(contexts), -1))
        hidden_relu = self.relu(hidden_out)
        hidden_drop = F.dropout(hidden_relu, training=self.is_training)
        tag_out = self.hidden2tag(hidden_drop)
        return tag_out


class Cnn(Ner):    # pylint: disable=too-many-instance-attributes
    """
    convolutional neural network based part-of-speech tagger
    """
    def __init__(self, window, voca, embed_dim, hidden_dim):
        """
        :param  window:  left/right window size from current character
        :param  voca:  vocabulary
        :param  embed_dim:  character embedding dimension
        :param  hidden_dim:  hidden layer dimension
        """
        super().__init__(window, voca)
        self.is_training = True
        self.context_len = 2 * window + 1
        self.embedding = nn.Embedding(len(voca['in']), embed_dim)
        self.conv5_1 = nn.Conv1d(embed_dim, embed_dim * 2, 5)    # 21 - 4 => 17
        self.relu51 = nn.ReLU()
        self.pool5_1 = nn.MaxPool1d(2)    # 17 // 2 => 8
        self.conv5_2 = nn.Conv1d(embed_dim * 2, embed_dim * 4, 5)    # 8 - 4 => 4
        self.relu52 = nn.ReLU()
        self.pool5_2 = nn.MaxPool1d(2)    # 4 // 2 => 2
        self.conv4_1 = nn.Conv1d(embed_dim, embed_dim * 2, 4)    # 21 - 3 => 18
        self.relu42 = nn.ReLU()
        self.pool4_1 = nn.MaxPool1d(2)    # 18 // 2 => 9
        self.conv4_2 = nn.Conv1d(embed_dim * 2, embed_dim * 4, 4)    # 9 - 3 => 6
        self.relu41 = nn.ReLU()
        self.pool4_2 = nn.MaxPool1d(2)    # 6 // 2 => 3
        self.conv3_1 = nn.Conv1d(embed_dim, embed_dim * 2, 3)    # 21 - 2 => 19
        self.relu31 = nn.ReLU()
        self.pool3_1 = nn.MaxPool1d(2)    # 19 // 2 => 9
        self.conv3_2 = nn.Conv1d(embed_dim * 2, embed_dim * 4, 3)    # 9 - 2 => 7
        self.relu32 = nn.ReLU()
        self.pool3_2 = nn.MaxPool1d(2)    # 7 // 2 => 3
        self.conv2_1 = nn.Conv1d(embed_dim, embed_dim * 2, 2)    # 21 - 1 => 20
        self.relu21 = nn.ReLU()
        self.pool2_1 = nn.MaxPool1d(2)    # 20 // 2 => 10
        self.conv2_2 = nn.Conv1d(embed_dim * 2, embed_dim * 4, 2)    # 10 - 1 => 9
        self.relu22 = nn.ReLU()
        self.pool2_2 = nn.MaxPool1d(2)    # 9 // 2 => 4
        self.conv_tri = nn.Conv1d(embed_dim, embed_dim * 4, 3)
        self.relu_tri = nn.ReLU()
        self.hidden = autograd.Variable(torch.zeros(1, 1, hidden_dim))
        self.relu_h = nn.ReLU()
        self.conv2hidden = nn.Linear((2 + 3 + 3 + 4 + 1) * embed_dim * 4, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, len(voca['out']))

    def forward(self, contexts):    # pylint: disable=arguments-differ,too-many-locals
        embeds = self.embedding(contexts)
        # batch_size x context_len x embedding_dim => batch_size x embedding_dim x context_len
        embeds_t = torch.transpose(embeds, 1, 2)
        conv5_1 = self.conv5_1(embeds_t)
        pool5_1 = self.pool5_1(self.relu51(conv5_1))
        conv5_2 = self.conv5_2(pool5_1)
        pool5_2 = self.pool5_1(self.relu52(conv5_2))
        conv5_t = torch.transpose(pool5_2, 1, 2)    # transpose to original dimension
        conv4_1 = self.conv4_1(embeds_t)
        pool4_1 = self.pool4_1(self.relu41(conv4_1))
        conv4_2 = self.conv4_2(pool4_1)
        pool4_2 = self.pool4_1(self.relu42(conv4_2))
        conv4_t = torch.transpose(pool4_2, 1, 2)    # transpose to original dimension
        conv3_1 = self.conv3_1(embeds_t)
        pool3_1 = self.pool3_1(self.relu31(conv3_1))
        conv3_2 = self.conv3_2(pool3_1)
        pool3_2 = self.pool3_1(self.relu32(conv3_2))
        conv3_t = torch.transpose(pool3_2, 1, 2)    # transpose to original dimension
        conv2_1 = self.conv2_1(embeds_t)
        pool2_1 = self.pool2_1(self.relu21(conv2_1))
        conv2_2 = self.conv2_2(pool2_1)
        pool2_2 = self.pool2_1(self.relu22(conv2_2))
        conv2_t = torch.transpose(pool2_2, 1, 2)    # transpose to original dimension
        trigram = embeds.narrow(1, self.context_len // 2, 3)    # cut middle trigram
        conv_tri = self.conv_tri(torch.transpose(trigram, 1, 2))
        # transpose to original dimension
        conv_tri_t = torch.transpose(self.relu_tri(conv_tri), 1, 2)
        features = torch.cat([conv5_t, conv4_t, conv3_t, conv2_t, conv_tri_t], dim=1)
        hidden_out = self.conv2hidden(features.view(len(contexts), -1))
        hidden_relu = self.relu_h(hidden_out)
        hidden_drop = F.dropout(hidden_relu, training=self.is_training)
        tag_out = self.hidden2tag(hidden_drop)
        return tag_out

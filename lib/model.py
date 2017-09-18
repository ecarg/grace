# -*- coding: utf-8 -*-


"""
Pytorch models
__author__ = 'Jamie (krikit@naver.com)'
__copyright__ = 'No copyright. Just copyleft!'
"""


# pylint: disable=no-member


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
    def __init__(self, window, voca, gazet, is_phoneme):
        """
        :param  window:  left/right window size from current character
        :param  voca:  vocabulary
        """
        super().__init__()
        self.is_training = True    # is training phase or not (use drop-out at training)
        self.window = window
        self.voca = voca
        self.gazet = gazet
        self.is_phoneme = is_phoneme

    def forward(self, *inputs):
        raise NotImplementedError


class Fnn3(Ner):
    """
    feed-forward neural network based part-of-speech tagger
    """
    def __init__(self, window, voca, gazet, embed_dim, hidden_dim, is_phoneme):
        """
        :param  window:  left/right window size from current character
        :param  voca:  vocabulary
        :param  embed_dim:  character embedding dimension
        :param  hidden_dim:  hidden layer dimension
        """
        super().__init__(window, voca, gazet, is_phoneme)
        context_len = 2 * window + 1
        self.embedding = nn.Embedding(len(voca['in']), embed_dim)
        if self.is_phoneme:
            self.pho2syl = nn.Conv2d(1, embed_dim, (3, embed_dim), 3)
        self.hidden = autograd.Variable(torch.zeros(1, 1, hidden_dim))
        self.relu = nn.ReLU()
        self.embeds2hidden = nn.Linear(context_len * embed_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, len(voca['out']))

    def forward(self, contexts_gazet):    # pylint: disable=arguments-differ
        """
        forward path
        :param  contexts:  batch size list of character and context
        :return:  output score
        """
        contexts, _ = contexts_gazet
        embeds = self.embedding(contexts)
        if self.is_phoneme:
            embeds = self.pho2syl(embeds.unsqueeze(1)).squeeze().transpose(1, 2)
            embeds.contiguous()
        hidden_out = self.embeds2hidden(embeds.view(len(contexts), -1))
        hidden_relu = self.relu(hidden_out)
        hidden_drop = F.dropout(hidden_relu, training=self.is_training)
        tag_out = self.hidden2tag(hidden_drop)
        return tag_out


class Fnn4(Ner):
    """
    feed-forward neural network based part-of-speech tagger
    """
    def __init__(self, window, voca, gazet, embed_dim, hidden_dim, is_phoneme):
        """
        :param  window:  left/right window size from current character
        :param  voca:  vocabulary
        :param  embed_dim:  character embedding dimension
        :param  hidden_dim:  hidden layer dimension
        """
        super().__init__(window, voca, gazet, is_phoneme)
        context_len = 2 * window + 1
        self.embedding = nn.Embedding(len(voca['in']), embed_dim)
        if self.is_phoneme:
            self.pho2syl = nn.Conv2d(1, embed_dim, (3, embed_dim), 3)

        self.hidden = autograd.Variable(torch.zeros(1, 1, hidden_dim))
        self.relu = nn.ReLU()
        self.embeds2hidden = nn.Linear(context_len * (embed_dim + len(voca['out'])+4), hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, len(voca['out']))

    def forward(self, contexts_gazet):    # pylint: disable=arguments-differ
        """
        forward path
        :param  contexts:  batch size list of character and context
        :return:  output score
        """
        contexts, gazet = contexts_gazet
        embeds = self.embedding(contexts)
        if self.is_phoneme:
            embeds = self.pho2syl(embeds.unsqueeze(1)).squeeze().transpose(1, 2)

        embeds = torch.cat([embeds, gazet], 2)
        hidden_out = self.embeds2hidden(embeds.view(len(contexts), -1))
        hidden_relu = self.relu(hidden_out)
        hidden_drop = F.dropout(hidden_relu, training=self.is_training)
        tag_out = self.hidden2tag(hidden_drop)
        return tag_out


class Cnn3(Ner):    # pylint: disable=too-many-instance-attributes
    """
    convolutional neural network based part-of-speech tagger
    """
    def __init__(self, window, voca, gazet, embed_dim, hidden_dim, is_phoneme):
        """
        :param  window:  left/right window size from current character
        :param  voca:  vocabulary
        :param  embed_dim:  character embedding dimension
        :param  hidden_dim:  hidden layer dimension
        """
        super().__init__(window, voca, gazet, is_phoneme)
        self.is_training = True
        self.context_len = 2 * window + 1

        self.embedding = nn.Embedding(len(voca['in']), embed_dim)
        if self.is_phoneme:
            self.pho2syl = nn.Conv2d(1, embed_dim, (3, embed_dim), 3)

        # conv3_1
        self.conv3_1 = nn.Conv2d(1, embed_dim * 2, kernel_size=(3, embed_dim), stride=1,
                                 padding=(1, 0))    # 21 x 50  =>  21 x 100
        self.pool3_1 = nn.MaxPool2d(kernel_size=(2, 1), ceil_mode=True)    # 21 x 100  =>  11 x 100

        # conv3_2
        self.conv3_2 = nn.Conv2d(1, embed_dim * 4, kernel_size=(3, embed_dim * 2), stride=1,
                                 padding=(1, 0))    # 11 x 100  =>  11 x 200
        self.pool3_2 = nn.MaxPool2d(kernel_size=(2, 1), ceil_mode=True)    # 11 x 200  =>  6 x 200

        # conv3_3
        self.conv3_3 = nn.Conv2d(1, embed_dim * 8, kernel_size=(3, embed_dim * 4), stride=1,
                                 padding=(1, 0))    # 6 x 400  =>  6 x 400
        self.pool3_3 = nn.MaxPool2d(kernel_size=(2, 1), ceil_mode=True)    # 6 x 400  =>  3 x 400


        # conv3_4: 3 x 400  =>  1 x 800
        self.conv3_4 = nn.Conv2d(1, embed_dim * 16, kernel_size=(3, embed_dim * 8), stride=1)

        # conv => hidden
        self.hidden = autograd.Variable(torch.zeros(1, 1, hidden_dim))
        self.conv2hidden = nn.Linear(800, hidden_dim)

        # hidden => tag
        self.hidden2tag = nn.Linear(hidden_dim, len(voca['out']))

    def forward(self, contexts_gazet):    # pylint: disable=arguments-differ,too-many-locals
        contexts, _ = contexts_gazet
        embeds = self.embedding(contexts)
        # batch_size x context_len x embedding_dim => batch_size x embedding_dim x context_len
        if self.is_phoneme:
            pho2syl_out = self.pho2syl(embeds.unsqueeze(1))
            embeds = pho2syl_out.squeeze().transpose(1, 2)

        # conv3_1
        conv3_1 = self.conv3_1(embeds.unsqueeze(1))
        pool3_1 = self.pool3_1(F.relu(conv3_1))
        pool3_1 = pool3_1.squeeze().transpose(1, 2)

        # conv3_2
        conv3_2 = self.conv3_2(pool3_1.unsqueeze(1))
        pool3_2 = self.pool3_2(F.relu(conv3_2))
        pool3_2 = pool3_2.squeeze().transpose(1, 2)

        # conv3_3
        conv3_3 = self.conv3_3(pool3_2.unsqueeze(1))
        pool3_3 = self.pool3_3(F.relu(conv3_3))
        pool3_3 = pool3_3.squeeze().transpose(1, 2)

        # conv3_4
        conv3_4 = self.conv3_4(pool3_3.unsqueeze(1))
        conv3_4_relu = F.relu(conv3_4)

        # conv => hidden
        features = conv3_4_relu.view(len(contexts), -1)
        features_drop = F.dropout(features, training=self.is_training)
        hidden_out = self.conv2hidden(features_drop)
        hidden_out_relu = F.relu(hidden_out)

        # hidden => tag
        hidden_out_drop = F.dropout(hidden_out_relu, training=self.is_training)
        tag_out = self.hidden2tag(hidden_out_drop)

        return tag_out


class Cnn4(Ner):    # pylint: disable=too-many-instance-attributes
    """
    convolutional neural network based part-of-speech tagger
    """
    def __init__(self, window, voca, gazet, embed_dim, hidden_dim, is_phoneme):
        """
        :param  window:  left/right window size from current character
        :param  voca:  vocabulary
        :param  embed_dim:  character embedding dimension
        :param  hidden_dim:  hidden layer dimension
        """
        super().__init__(window, voca, gazet, is_phoneme)
        self.is_training = True
        self.context_len = 2 * window + 1

        self.embedding = nn.Embedding(len(voca['in']), embed_dim)
        if self.is_phoneme:
            self.pho2syl = nn.Conv2d(1, embed_dim, (3, embed_dim), 3)

        feature_dim = embed_dim + len(voca['out']) + 4
        # conv3_1
        self.conv3_1 = nn.Conv2d(1, feature_dim * 2, kernel_size=(3, feature_dim), stride=1,
                                 padding=(1, 0))    # 21 x 65  =>  21 x 130
        self.pool3_1 = nn.MaxPool2d(kernel_size=(2, 1), ceil_mode=True)    # 21 x 130  =>  11 x 130

        # conv3_2
        self.conv3_2 = nn.Conv2d(1, feature_dim * 4, kernel_size=(3, feature_dim * 2), stride=1,
                                 padding=(1, 0))    # 11 x 130  =>  11 x 260
        self.pool3_2 = nn.MaxPool2d(kernel_size=(2, 1), ceil_mode=True)    # 11 x 260  =>  6 x 260

        # conv3_3
        self.conv3_3 = nn.Conv2d(1, feature_dim * 8, kernel_size=(3, feature_dim * 4), stride=1,
                                 padding=(1, 0))    # 6 x 260  =>  6 x 520
        self.pool3_3 = nn.MaxPool2d(kernel_size=(2, 1), ceil_mode=True)    # 6 x 520  =>  3 x 520

        # conv3_4: 3 x 520  =>  1 x 1040
        self.conv3_4 = nn.Conv2d(1, feature_dim * 16, kernel_size=(3, feature_dim * 8), stride=1)

        # conv => hidden
        self.hidden = autograd.Variable(torch.zeros(1, 1, hidden_dim))
        self.conv2hidden = nn.Linear(feature_dim * 16, hidden_dim)

        # hidden => tag
        self.hidden2tag = nn.Linear(hidden_dim, len(voca['out']))

    def forward(self, contexts_gazet):    # pylint: disable=arguments-differ,too-many-locals
        contexts, gazet = contexts_gazet
        embeds = self.embedding(contexts)
        # batch_size x context_len x embedding_dim => batch_size x embedding_dim x context_len
        if self.is_phoneme:
            pho2syl_out = self.pho2syl(embeds.unsqueeze(1))
            embeds = pho2syl_out.squeeze().transpose(1, 2)
        # concat gazet feature
        embeds = torch.cat([embeds, gazet], 2)
        # conv3_1
        conv3_1 = self.conv3_1(embeds.unsqueeze(1))
        pool3_1 = self.pool3_1(F.relu(conv3_1))
        pool3_1 = pool3_1.squeeze().transpose(1, 2)

        # conv3_2
        conv3_2 = self.conv3_2(pool3_1.unsqueeze(1))
        pool3_2 = self.pool3_2(F.relu(conv3_2))
        pool3_2 = pool3_2.squeeze().transpose(1, 2)

        # conv3_3
        conv3_3 = self.conv3_3(pool3_2.unsqueeze(1))
        pool3_3 = self.pool3_3(F.relu(conv3_3))
        pool3_3 = pool3_3.squeeze().transpose(1, 2)

        # conv3_4
        conv3_4 = self.conv3_4(pool3_3.unsqueeze(1))
        conv3_4_relu = F.relu(conv3_4)

        # conv => hidden
        features = conv3_4_relu.view(len(contexts), -1)
        features_drop = F.dropout(features, training=self.is_training)
        hidden_out = self.conv2hidden(features_drop)
        hidden_out_relu = F.relu(hidden_out)

        # hidden => tag
        hidden_out_drop = F.dropout(hidden_out_relu, training=self.is_training)
        tag_out = self.hidden2tag(hidden_out_drop)

        return tag_out

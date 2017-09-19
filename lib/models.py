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
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

#########
# types #
#########
class Ner(nn.Module):
    """
    named entity recognizer pytorch model
    """
    def __init__(self, window, voca, gazet, is_phoneme, is_gazet_1hot, is_positional_encoding):
        """
        :param  window:  left/right window size from current character
        :param  voca:  vocabulary
        """
        super().__init__()
        self.window = window
        self.voca = voca
        self.gazet = gazet
        self.is_phoneme = is_phoneme
        self.is_gazet_1hot = is_gazet_1hot
        self.is_positional_encoding = is_positional_encoding

    def forward(self, *inputs):
        raise NotImplementedError

    def save(self, path):
        """
        모델을 저장하는 메소드
        :param  path:  경로
        """
        if torch.cuda.is_available():
            self.cpu()
        torch.save(self, path)
        if torch.cuda.is_available():
            self.cuda()

    @classmethod
    def load(cls, path):
        """
        저장된 모델을 로드하는 메소드
        :param  path:  경로
        :return:  모델 클래스 객체
        """
        model = torch.load(path)
        if torch.cuda.is_available():
            model.cuda()
        return model

    def positional_encoding(self, context_len, embed_dim):
        """
        Positional encoding Variable 출력
        embeds [batch_size, context_len, embed_dim]에 Broadcasting 으로 더해짐
        :return: pe [context_len, embed_dim]
        """
        pe = torch.zeros([context_len, embed_dim])

        for j in range(context_len):
            j += 1 # 1-based indexing
            for k in range(embed_dim):
                k += 1 # 1-based indexing
                pe[j-1,k-1] = (1-j/context_len) - (k/embed_dim)*(1-2*j/context_len)

        pe = Variable(pe)
        if torch.cuda.is_available():
            pe = pe.cuda()

        return pe


class Fnn5(Ner):
    """
    feed-forward neural network based named entity recognizer
    """
    def __init__(self, window, voca, gazet, embed_dim, hidden_dim, is_phoneme, is_gazet_1hot, is_positional_encoding):
        """
        :param  window:  left/right window size from current character
        :param  voca:  vocabulary
        :param  embed_dim:  character embedding dimension
        :param  hidden_dim:  hidden layer dimension
        """
        super().__init__(window, voca, gazet, is_phoneme, is_gazet_1hot, is_positional_encoding)
        context_len = 2 * window + 1
        self.embedding = nn.Embedding(len(voca['in']), embed_dim)
        gazet_dim = len(voca['out'])+4

        if not self.is_gazet_1hot:
            gazet_dim = embed_dim//2
            self.gazet_embedding = nn.Embedding(int(math.pow(2, len(voca['out'])+4)), gazet_dim)

        if self.is_phoneme:
            self.pho2syl = nn.Conv2d(1, embed_dim, (3, embed_dim), 3)

        # Add Positional Encoding
        if self.is_positional_encoding:
            self.pe = self.positional_encoding(context_len, embed_dim)

        self.relu = nn.ReLU()
        self.embeds2hidden = nn.Linear(context_len * (embed_dim + gazet_dim), hidden_dim)
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
            embeds = F.relu(self.pho2syl(embeds.unsqueeze(1)).squeeze().transpose(1, 2))

        if self.is_positional_encoding:
            embeds += self.pe 

        if self.is_gazet_1hot:
            embeds = torch.cat([embeds, gazet], 2)
        else:
            gazet_embeds = self.gazet_embedding(gazet)
            embeds = torch.cat([embeds, gazet_embeds], 2)

        hidden_out = self.embeds2hidden(embeds.view(len(contexts), -1))
        hidden_relu = self.relu(hidden_out)
        hidden_drop = F.dropout(hidden_relu, training=self.training)
        tag_out = self.hidden2tag(hidden_drop)
        return tag_out


class Cnn7(Ner):    # pylint: disable=too-many-instance-attributes
    """
    convolutional neural network based named entity recognizer
    """
    def __init__(self, window, voca, gazet, embed_dim, hidden_dim, is_phoneme, is_gazet_1hot, is_positional_encoding):
        """
        :param  window:  left/right window size from current character
        :param  voca:  vocabulary
        :param  embed_dim:  character embedding dimension
        :param  hidden_dim:  hidden layer dimension
        """
        super().__init__(window, voca, gazet, is_phoneme, is_gazet_1hot, is_positional_encoding)
        self.context_len = 2 * window + 1

        self.embedding = nn.Embedding(len(voca['in']), embed_dim)
        if self.is_phoneme:
            self.pho2syl = nn.Conv1d(embed_dim, embed_dim, 3, 3)

        # Add Positional Encoding
        if self.is_positional_encoding:
            self.pe = self.positional_encoding(self.context_len, embed_dim)

        concat_dim = embed_dim + len(voca['out']) + 4
        # conv2_1
        self.conv2_1 = nn.Conv1d(concat_dim, concat_dim, kernel_size=2)    # 20
        self.pool2_1 = nn.MaxPool1d(kernel_size=2, ceil_mode=True)    # 10
        # conv2_2
        self.conv2_2 = nn.Conv1d(concat_dim, concat_dim, kernel_size=2)    # 9
        self.pool2_2 = nn.MaxPool1d(kernel_size=2, ceil_mode=True)    # 5
        # conv2_3
        self.conv2_3 = nn.Conv1d(concat_dim, concat_dim, kernel_size=2)    # 4
        self.pool2_3 = nn.MaxPool1d(kernel_size=2, ceil_mode=True)    # 2
        # conv2_4
        self.conv2_4 = nn.Conv1d(concat_dim, concat_dim, kernel_size=2)    # 1

        # conv3_1
        self.conv3_1 = nn.Conv1d(concat_dim, concat_dim, kernel_size=3, padding=1)    # 21
        self.pool3_1 = nn.MaxPool1d(kernel_size=2, ceil_mode=True)    # 11
        # conv3_2
        self.conv3_2 = nn.Conv1d(concat_dim, concat_dim, kernel_size=3, padding=1)    # 11
        self.pool3_2 = nn.MaxPool1d(kernel_size=2, ceil_mode=True)    # 6
        # conv3_3
        self.conv3_3 = nn.Conv1d(concat_dim, concat_dim, kernel_size=3, padding=1)    # 6
        self.pool3_3 = nn.MaxPool1d(kernel_size=2, ceil_mode=True)    # 3
        # conv3_4
        self.conv3_4 = nn.Conv1d(concat_dim, concat_dim, kernel_size=3)    # 1

        # conv4_1
        self.conv4_1 = nn.Conv1d(concat_dim, concat_dim, kernel_size=4, padding=1)    # 20
        self.pool4_1 = nn.MaxPool1d(kernel_size=2, ceil_mode=True)    # 10
        # conv4_2
        self.conv4_2 = nn.Conv1d(concat_dim, concat_dim, kernel_size=4, padding=1)    # 9
        self.pool4_2 = nn.MaxPool1d(kernel_size=2, ceil_mode=True)    # 5
        # conv4_3
        self.conv4_3 = nn.Conv1d(concat_dim, concat_dim, kernel_size=4, padding=1)    # 4
        # conv4_4
        self.conv4_4 = nn.Conv1d(concat_dim, concat_dim, kernel_size=4)    # 1

        # conv5_1
        self.conv5_1 = nn.Conv1d(concat_dim, concat_dim, kernel_size=5, padding=2)    # 21
        self.pool5_1 = nn.MaxPool1d(kernel_size=2, ceil_mode=True)    # 11
        # conv5_2
        self.conv5_2 = nn.Conv1d(concat_dim, concat_dim, kernel_size=5, padding=2)    # 11
        self.pool5_2 = nn.MaxPool1d(kernel_size=2, ceil_mode=True)    # 6
        # conv5_3
        self.conv5_3 = nn.Conv1d(concat_dim, concat_dim, kernel_size=5, padding=2)    # 6
        self.pool5_3 = nn.MaxPool1d(kernel_size=2, ceil_mode=True)    # 3
        # conv5_4
        self.conv5_4 = nn.Conv1d(concat_dim, concat_dim, kernel_size=5, padding=1)    # 1

        # conv => hidden
        self.conv2hidden = nn.Linear(concat_dim * 4, hidden_dim)

        # hidden => tag
        self.hidden2tag = nn.Linear(hidden_dim, len(voca['out']))

    def forward(self, contexts_gazet):    # pylint: disable=arguments-differ,too-many-locals
        contexts, gazet = contexts_gazet
        embeds = self.embedding(contexts)
        # batch_size x context_len x embedding_dim => batch_size x embedding_dim x context_len
        if self.is_phoneme:
            embeds_t = F.relu(self.pho2syl(embeds.transpose(1, 2)))
            embeds = embeds_t.transpose(1, 2)

        # Add Positional Encoding
        if self.is_positional_encoding:
            embeds = embeds + self.pe 

        # concat gazet feature
        embeds = torch.cat([embeds, gazet], 2)
        embeds_t = embeds.transpose(1, 2)

        # conv2_1
        conv2_1 = F.relu(self.conv2_1(embeds_t))
        pool2_1 = self.pool2_1(conv2_1)
        # conv2_2
        conv2_2 = F.relu(self.conv2_2(pool2_1))
        pool2_2 = self.pool2_2(conv2_2)
        # conv2_3
        conv2_3 = F.relu(self.conv2_3(pool2_2))
        pool2_3 = self.pool2_3(conv2_3)
        # conv2_4
        conv2_4 = F.relu(self.conv2_4(pool2_3))

        # conv3_1
        conv3_1 = F.relu(self.conv3_1(embeds_t))
        pool3_1 = self.pool3_1(conv3_1)
        # conv3_2
        conv3_2 = F.relu(self.conv3_2(pool3_1))
        pool3_2 = self.pool3_2(conv3_2)
        # conv3_3
        conv3_3 = F.relu(self.conv3_3(pool3_2))
        pool3_3 = self.pool3_3(conv3_3)
        # conv3_4
        conv3_4 = F.relu(self.conv3_4(pool3_3))

        # conv4_1
        conv4_1 = F.relu(self.conv4_1(embeds_t))
        pool4_1 = self.pool4_1(conv4_1)
        # conv4_2
        conv4_2 = F.relu(self.conv4_2(pool4_1))
        pool4_2 = self.pool4_2(conv4_2)
        # conv4_3
        conv4_3 = F.relu(self.conv4_3(pool4_2))
        # conv4_4
        conv4_4 = F.relu(self.conv4_4(F.relu(conv4_3)))

        # conv5_1
        conv5_1 = F.relu(self.conv5_1(embeds_t))
        pool5_1 = self.pool5_1(conv5_1)
        # conv5_2
        conv5_2 = F.relu(self.conv5_2(pool5_1))
        pool5_2 = self.pool5_2(conv5_2)
        # conv5_3
        conv5_3 = F.relu(self.conv5_3(pool5_2))
        pool5_3 = self.pool5_3(conv5_3)
        # conv5_4
        conv5_4 = F.relu(self.conv5_4(pool5_3))

        # conv => hidden
        features = torch.cat([conv2_4.view(len(contexts), -1),
                              conv3_4.view(len(contexts), -1),
                              conv4_4.view(len(contexts), -1),
                              conv5_4.view(len(contexts), -1)], dim=1)
        features_drop = F.dropout(features, training=self.training)
        hidden_out = F.relu(self.conv2hidden(features_drop))

        # hidden => tag
        hidden_out_drop = F.dropout(hidden_out, training=self.training)
        tag_out = self.hidden2tag(hidden_out_drop)
        return tag_out

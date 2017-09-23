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

from pos_models import PosTagger, FnnTagger, CnnTagger    # pylint: disable=unused-import


#########
# types #
#########
class Ner(nn.Module):
    """
    named entity recognizer pytorch model
    """
    def __init__(self, window, embed_dim, voca, gazet, phoneme, gazet_embed, pos_enc):
        """
        :param  window:  left/right window size from current character
        :param  voca:  vocabulary
        """
        super().__init__()
        self.window = window
        self.embed_dim = embed_dim
        self.voca = voca
        self.gazet = gazet
        self.phoneme = phoneme
        self.gazet_embed = gazet_embed
        self.pos_enc = pos_enc
        self.pe_tensor = None

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

    @classmethod
    def positional_encoding(cls, context_len, embed_dim):
        """
        Positional encoding Variable 출력
        embeds [batch_size, context_len, embed_dim]에 Broadcasting 으로 더해짐
        :return: pe [context_len, embed_dim]
        """
        pe_tensor = torch.zeros([context_len, embed_dim])
        for j in range(context_len):
            j += 1 # 1-based indexing
            for k in range(embed_dim):
                k += 1 # 1-based indexing
                pe_tensor[j-1, k-1] = (1-j/context_len) - (k/embed_dim)*(1-2*j/context_len)

        pe_tensor = Variable(pe_tensor)
        if torch.cuda.is_available():
            pe_tensor = pe_tensor.cuda()
        return pe_tensor


class Fnn5(Ner):
    """
    feed-forward neural network based named entity recognizer
    """
    def __init__(self, window, voca, gazet, embed_dim, hidden_dim, phoneme, gazet_embed, pos_enc):
        """
        :param  window:  left/right window size from current character
        :param  voca:  vocabulary
        :param  embed_dim:  character embedding dimension
        :param  hidden_dim:  hidden layer dimension
        """
        super().__init__(window, embed_dim, voca, gazet, phoneme, gazet_embed, pos_enc)
        context_len = 2 * window + 1
        self.embedding = nn.Embedding(len(voca['in']), embed_dim)
        gazet_dim = len(voca['out'])+4

        if self.gazet_embed:
            gazet_dim = 20
            self.gazet_embedding = nn.Embedding(int(math.pow(2, len(voca['out'])+4)), gazet_dim)

        if self.phoneme:
            self.pho2syl = nn.Conv2d(1, embed_dim, (3, embed_dim), 3)

        self.embeds2hidden = nn.Linear(context_len * (embed_dim + gazet_dim + 626), hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, len(voca['out']))

    def forward(self, inputs):    # pylint: disable=arguments-differ
        """
        forward path
        :param  contexts:  batch size list of character and context
        :return:  output score
        """
        contexts, gazet, pos_embed = inputs
        embeds = self.embedding(contexts)
        if self.phoneme:
            embeds = F.relu(self.pho2syl(embeds.unsqueeze(1)).squeeze().transpose(1, 2))

        # Add Positional Encoding
        if self.pos_enc:
            if self.pe_tensor is None:
                context_len = self.window * 2 + 1
                self.pe_tensor = self.positional_encoding(context_len, self.embed_dim)
            embeds += self.pe_tensor

        if self.gazet_embed:
            gazet_embeds = self.gazet_embedding(gazet)
            embeds = torch.cat([embeds, gazet_embeds], 2)
        else:
            embeds = torch.cat([embeds, gazet], 2)

        # PoS feature
        embeds = torch.cat([embeds, pos_embed], 2)

        hidden_out = F.relu(self.embeds2hidden(embeds.view(len(contexts), -1)))
        hidden_drop = F.dropout(hidden_out, training=self.training)
        tag_out = self.hidden2tag(hidden_drop)
        return tag_out


class Cnn7(Ner):    # pylint: disable=too-many-instance-attributes
    """
    convolutional neural network based named entity recognizer
    """
    def __init__(self, window, voca, gazet, embed_dim, hidden_dim, phoneme, gazet_embed, pos_enc):
        """
        :param  window:  left/right window size from current character
        :param  voca:  vocabulary
        :param  embed_dim:  character embedding dimension
        :param  hidden_dim:  hidden layer dimension
        """
        super().__init__(window, embed_dim, voca, gazet, phoneme, gazet_embed, pos_enc)
        self.context_len = 2 * window + 1
        self.embedding = nn.Embedding(len(voca['in']), embed_dim)

        gazet_dim = len(voca['out'])+4
        if self.gazet_embed:
            gazet_dim = 20
            self.gazet_embedding = nn.Embedding(int(math.pow(2, len(voca['out'])+4)), gazet_dim)

        if self.phoneme:
            self.pho2syl = nn.Conv1d(embed_dim, embed_dim, 3, 3)

        concat_dim = embed_dim + gazet_dim
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

    def forward(self, inputs):    # pylint: disable=arguments-differ,too-many-locals
        contexts, gazet, _ = inputs
        embeds = self.embedding(contexts)
        # batch_size x context_len x embedding_dim => batch_size x embedding_dim x context_len
        if self.phoneme:
            embeds_t = F.relu(self.pho2syl(embeds.transpose(1, 2)))
            embeds = embeds_t.transpose(1, 2)

        # Add Positional Encoding
        if self.pos_enc:
            if self.pe_tensor is None:
                context_len = self.window * 2 + 1
                self.pe_tensor = self.positional_encoding(context_len, self.embed_dim)

            embeds = embeds + self.pe_tensor

        if self.gazet_embed:
            gazet_embeds = self.gazet_embedding(gazet)
            embeds = torch.cat([embeds, gazet_embeds], 2)
        else:
            embeds = torch.cat([embeds, gazet], 2)

        # concat gazet feature
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


class Rnn1(Ner):
    """
    recurrent neural network based named entity recognizer
    """
    def __init__(self, window, voca, gazet, embed_dim, rnn_dim, hidden_dim, phoneme, gazet_embed,
                 pos_enc):
        """
        :param  window:  left/right window size from current character
        :param  voca:  vocabulary
        :param  embed_dim:  character embedding dimension
        :param  hidden_dim:  hidden layer dimension
        """
        super().__init__(window, embed_dim, voca, gazet, phoneme, gazet_embed, pos_enc)
        self.rnn_dim = rnn_dim
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(len(voca['in']), embed_dim)
        gazet_dim = len(voca['out'])+4

        if self.gazet_embed:
            gazet_dim = 20
            self.gazet_embedding = nn.Embedding(int(math.pow(2, len(voca['out'])+4)), gazet_dim)

        if self.phoneme:
            self.pho2syl = nn.Conv2d(1, embed_dim, (3, embed_dim), 3)

        self.rnn = nn.LSTM(embed_dim + gazet_dim, rnn_dim, 2, batch_first=True, bidirectional=True)
        self.rnn2hidden = nn.Linear(rnn_dim * 2, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, len(voca['out']))

    def forward(self, inputs):    # pylint: disable=arguments-differ
        """
        forward path
        :param  contexts:  batch size list of character and context
        :return:  output score
        """
        contexts, gazet, _ = inputs
        embeds = self.embedding(contexts)
        if self.phoneme:
            embeds = F.relu(self.pho2syl(embeds.unsqueeze(1)).squeeze().transpose(1, 2))

        # Add Positional Encoding
        if self.pos_enc:
            if self.pe_tensor is None:
                context_len = self.window * 2 + 1
                self.pe_tensor = self.positional_encoding(context_len, self.embed_dim)
            embeds += self.pe_tensor

        if self.gazet_embed:
            gazet_embeds = self.gazet_embedding(gazet)
            embeds = torch.cat([embeds, gazet_embeds], 2)
        else:
            embeds = torch.cat([embeds, gazet], 2)

        # initialize hidden and cell state
        hs0 = Variable(torch.zeros(2 * 2, embeds.size(0), self.rnn_dim))
        cs0 = Variable(torch.zeros(2 * 2, embeds.size(0), self.rnn_dim))
        if torch.cuda.is_available():
            hs0 = hs0.cuda()
            cs0 = cs0.cuda()

        rnn_out, _ = self.rnn(embeds, (hs0, cs0))
        hidden_out = self.rnn2hidden(rnn_out[:, self.window, :])
        hidden_drop = F.dropout(hidden_out, training=self.training)
        tag_out = self.hidden2tag(hidden_drop)
        return tag_out


class Rnn2(Ner):
    """
    recurrent neural network based named entity recognizer
    """
    def __init__(self, window, voca, gazet, embed_dim, rnn_dim, hidden_dim, phoneme, gazet_embed,
                 pos_enc):
        """
        :param  window:  left/right window size from current character
        :param  voca:  vocabulary
        :param  embed_dim:  character embedding dimension
        :param  hidden_dim:  hidden layer dimension
        """
        from sru import SRU

        super().__init__(window, embed_dim, voca, gazet, phoneme, gazet_embed, pos_enc)
        self.rnn_dim = rnn_dim
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(len(voca['in']), embed_dim)
        gazet_dim = len(voca['out'])+4

        if self.gazet_embed:
            gazet_dim = 20
            self.gazet_embedding = nn.Embedding(int(math.pow(2, len(voca['out'])+4)), gazet_dim)

        if self.phoneme:
            self.pho2syl = nn.Conv2d(1, embed_dim, (3, embed_dim), 3)

        self.rnn = SRU(embed_dim + gazet_dim, rnn_dim, bidirectional=True)
        self.rnn2hidden = nn.Linear(rnn_dim * 2, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, len(voca['out']))

    def forward(self, inputs):    # pylint: disable=arguments-differ
        """
        forward path
        :param  contexts:  batch size list of character and context
        :return:  output score
        """
        contexts, gazet, _ = inputs
        embeds = self.embedding(contexts)
        if self.phoneme:
            embeds = F.relu(self.pho2syl(embeds.unsqueeze(1)).squeeze().transpose(1, 2))

        # Add Positional Encoding
        if self.pos_enc:
            if self.pe_tensor is None:
                context_len = self.window * 2 + 1
                self.pe_tensor = self.positional_encoding(context_len, self.embed_dim)
            embeds += self.pe_tensor

        if self.gazet_embed:
            gazet_embeds = self.gazet_embedding(gazet)
            embeds = torch.cat([embeds, gazet_embeds], 2)
        else:
            embeds = torch.cat([embeds, gazet], 2)

        # One time step
        rnn_out, _ = self.rnn(embeds.transpose(0, 1))
        hidden_out = self.rnn2hidden(rnn_out[self.window])
        hidden_drop = F.dropout(hidden_out, training=self.training)
        tag_out = self.hidden2tag(hidden_drop)
        return tag_out

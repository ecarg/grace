# -*- coding: utf-8 -*-


"""
Pytorch models
__author__ = 'Jamie (jamie.lim@kakaocorp.com)'
__copyright__ = 'Copyright (C) 2017-, Kakao Corp. All rights reserved.'
"""


# pylint: disable=no-member


###########
# imports #
###########
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


#########
# types #
#########
class PosTagger(nn.Module):
    """
    part-of-speech tagger pytorch model
    """
    def __init__(self, cfg):
        """
        :param  cfg:  configuation
        """
        super().__init__()
        self.cfg = cfg
        setattr(cfg, 'context_len', 2 * cfg.window + 1)
        self.embedding = nn.Embedding(len(cfg.voca['in']), cfg.embed_dim)
        if cfg.phoneme:
            self.pho2chr = nn.Conv1d(cfg.embed_dim, cfg.embed_dim, 3, 3)
        self.pe_tensor = None   # for positional encoding

    def forward(self, *inputs):
        raise NotImplementedError

    def make_embedding(self, contexts):
        """
        임베딩을 생성하는 메소드
        :param  contexts:  contexts of batch size
        :return:  embedding
        """
        embeds = self.embedding(contexts)

        # 자소 -> 음절 convolution
        if self.cfg.phoneme:
            embeds_t = F.relu(self.pho2chr(embeds.transpose(1, 2)))
            embeds = embeds_t.transpose(1, 2)

        # Add Positional Encoding
        if self.cfg.pos_enc:
            if self.pe_tensor is None:
                self.cfg.context_len = self.cfg.window * 2 + 1
                self.pe_tensor = self.positional_encoding(self.cfg.context_len, self.cfg.embed_dim)
            embeds += self.pe_tensor

        return embeds

    @classmethod
    def positional_encoding(cls, context_len, embed_dim):
        """
        Positional encoding Variable 출력
        embeds [batch_size, context_len, embed_dim]에 Broadcasting 으로 더해짐
        :return: pe [context_len, embed_dim]
        """
        pe_tensor = torch.zeros([context_len, embed_dim])
        for j in range(context_len):
            j += 1    # 1-based indexing
            for k in range(embed_dim):
                k += 1    # 1-based indexing
                pe_tensor[j-1, k-1] = (1-j/context_len) - (k/embed_dim)*(1-2*j/context_len)
        pe_tensor = Variable(pe_tensor)
        if torch.cuda.is_available():
            pe_tensor = pe_tensor.cuda()
        return pe_tensor

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


class FnnTagger(PosTagger):
    """
    feed-forward neural network based part-of-speech tagger
    """
    def __init__(self, cfg):
        """
        :param  cfg:  configuration
        """
        super().__init__(cfg)
        setattr(cfg, 'hidden_dim', (cfg.context_len * cfg.embed_dim + len(cfg.voca['out'])) // 2)
        self.hidden = Variable(torch.zeros(1, 1, cfg.hidden_dim))
        self.embeds2hidden = nn.Linear(cfg.context_len * cfg.embed_dim, cfg.hidden_dim)
        self.hidden2tag = nn.Linear(cfg.hidden_dim, len(cfg.voca['out']))

    def forward(self, contexts):    # pylint: disable=arguments-differ
        """
        forward path
        :param  contexts:  batch size list of character and context
        :return:  output score
        """
        embeds = self.make_embedding(contexts)
        hidden_out = F.relu(self.embeds2hidden(embeds.view(contexts.size(0), -1)))
        hidden_drop = F.dropout(hidden_out)
        tag_out = self.hidden2tag(hidden_drop)
        return tag_out


class CnnTagger(PosTagger):    # pylint: disable=too-many-instance-attributes
    """
    convolutional neural network based part-of-speech tagger
    """
    def __init__(self, cfg):
        """
        :param  cfg:  configuation
        """
        super().__init__(cfg)
        setattr(cfg, 'hidden_dim', (cfg.embed_dim * 4 + len(cfg.voca['out'])) // 2)

        # conv2_1
        self.conv2_1 = nn.Conv1d(cfg.embed_dim, cfg.embed_dim, kernel_size=2)    # 20
        self.pool2_1 = nn.MaxPool1d(kernel_size=2, ceil_mode=True)    # 10
        # conv2_2
        self.conv2_2 = nn.Conv1d(cfg.embed_dim, cfg.embed_dim, kernel_size=2)    # 9
        self.pool2_2 = nn.MaxPool1d(kernel_size=2, ceil_mode=True)    # 5
        # conv2_3
        self.conv2_3 = nn.Conv1d(cfg.embed_dim, cfg.embed_dim, kernel_size=2)    # 4
        self.pool2_3 = nn.MaxPool1d(kernel_size=2, ceil_mode=True)    # 2
        # conv2_4
        self.conv2_4 = nn.Conv1d(cfg.embed_dim, cfg.embed_dim, kernel_size=2)    # 1

        # conv3_1
        self.conv3_1 = nn.Conv1d(cfg.embed_dim, cfg.embed_dim, kernel_size=3, padding=1)    # 21
        self.pool3_1 = nn.MaxPool1d(kernel_size=2, ceil_mode=True)    # 11
        # conv3_2
        self.conv3_2 = nn.Conv1d(cfg.embed_dim, cfg.embed_dim, kernel_size=3, padding=1)    # 11
        self.pool3_2 = nn.MaxPool1d(kernel_size=2, ceil_mode=True)    # 6
        # conv3_3
        self.conv3_3 = nn.Conv1d(cfg.embed_dim, cfg.embed_dim, kernel_size=3, padding=1)    # 6
        self.pool3_3 = nn.MaxPool1d(kernel_size=2, ceil_mode=True)    # 3
        # conv3_4
        self.conv3_4 = nn.Conv1d(cfg.embed_dim, cfg.embed_dim, kernel_size=3)    # 1

        # conv4_1
        self.conv4_1 = nn.Conv1d(cfg.embed_dim, cfg.embed_dim, kernel_size=4, padding=1)    # 20
        self.pool4_1 = nn.MaxPool1d(kernel_size=2, ceil_mode=True)    # 10
        # conv4_2
        self.conv4_2 = nn.Conv1d(cfg.embed_dim, cfg.embed_dim, kernel_size=4, padding=1)    # 9
        self.pool4_2 = nn.MaxPool1d(kernel_size=2, ceil_mode=True)    # 5
        # conv4_3
        self.conv4_3 = nn.Conv1d(cfg.embed_dim, cfg.embed_dim, kernel_size=4, padding=1)    # 4
        # conv4_4
        self.conv4_4 = nn.Conv1d(cfg.embed_dim, cfg.embed_dim, kernel_size=4)    # 1

        # conv5_1
        self.conv5_1 = nn.Conv1d(cfg.embed_dim, cfg.embed_dim, kernel_size=5, padding=2)    # 21
        self.pool5_1 = nn.MaxPool1d(kernel_size=2, ceil_mode=True)    # 11
        # conv5_2
        self.conv5_2 = nn.Conv1d(cfg.embed_dim, cfg.embed_dim, kernel_size=5, padding=2)    # 11
        self.pool5_2 = nn.MaxPool1d(kernel_size=2, ceil_mode=True)    # 6
        # conv5_3
        self.conv5_3 = nn.Conv1d(cfg.embed_dim, cfg.embed_dim, kernel_size=5, padding=2)    # 6
        self.pool5_3 = nn.MaxPool1d(kernel_size=2, ceil_mode=True)    # 3
        # conv5_4
        self.conv5_4 = nn.Conv1d(cfg.embed_dim, cfg.embed_dim, kernel_size=5, padding=1)    # 1

        # conv => hidden
        self.conv2hidden = nn.Linear(cfg.embed_dim * 4, cfg.hidden_dim)

        # hidden => tag
        self.hidden2tag = nn.Linear(cfg.hidden_dim, len(cfg.voca['out']))

    def forward(self, contexts):    # pylint: disable=arguments-differ
        embeds = self.make_embedding(contexts)
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
        features = torch.cat([conv2_4.view(contexts.size(0), -1),
                              conv3_4.view(contexts.size(0), -1),
                              conv4_4.view(contexts.size(0), -1),
                              conv5_4.view(contexts.size(0), -1)], dim=1)
        features_drop = F.dropout(features)
        hidden_out = F.relu(self.conv2hidden(features_drop))

        # hidden => tag
        hidden_out_drop = F.dropout(hidden_out)
        tag_out = self.hidden2tag(hidden_out_drop)
        return tag_out

# -*- coding: utf-8 -*-


"""
Pytorch models
__author__ = 'Jamie (krikit@naver.com)'
__copyright__ = 'No copyright. Just copyleft!'
"""


# pylint: disable=no-member
# pylint: disable=invalid-name


###########
# imports #
###########
import torch
import torch.nn as nn
from embedder import Embedder

#############
# Ner Class #
#############
class Ner(nn.Module):
    """
    named entity recognizer pytorch model
    """
    def __init__(self, embedder, encoder, decoder):
        """
        * embedder (Embedder)
            [sentence_len, context_len] => [sentence_len, context_len, embed_dim]
        * encoder (nn.Module)
            [sentence_len, context_len, embed_dim] => [sentence_len, hidden_dim]
        * decoder (nn.Module)
            [sentence_len, hidden_dim] => [sentence_len, n_tags]
        """
        super().__init__()
        self.embedder = embedder
        self.encoder = encoder
        self.decoder = decoder

        assert isinstance(embedder, Embedder)
        assert isinstance(encoder, nn.Module)
        assert isinstance(decoder, nn.Module)

    def forward(self, sentence, gazet):
        # [sentence_len, context_len] => [sentence_len, context_len, embed_dim]
        sentence_embed = self.embedder(sentence, gazet)
        
        # [sentence_len, context_len, embed_dim] => [sentence_len, hidden_dim]
        hidden = self.encoder(sentence_embed)
        
        # [sentence_len, hidden_dim]   => [sentence_len, n_tags]
        predicted_tags = self.decoder(hidden)

        return predicted_tags

    def save(self, path):
        """
        모델을 저장하는 메소드
        :param  path:  경로
        """
        if torch.cuda.is_available():
            self.cpu()
        torch.save(self, str(path))
        if torch.cuda.is_available():
            self.cuda()

    @classmethod
    def load(cls, path):
        """
        저장된 모델을 로드하는 메소드
        :param  path:  경로
        :return:  모델 클래스 객체
        """
        model = torch.load(str(path))
        if torch.cuda.is_available():
            model.cuda()
        return model

#################
# Encoder Class #
#################

class Fnn5(nn.Module):
    """
    2-Layer Full-Connected Neural Networks
    """
    def __init__(self, context_len=21, in_dim=50, hidden_dim=500):
        super(Fnn5, self).__init__()

        self.context_len = context_len
        self.hidden_dim = hidden_dim
        self.out_dim = hidden_dim

        self.net = nn.Sequential(
            nn.Linear(context_len*in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x):
        """
        Args:
            x: [sentence_len, context_len, in_dim]
        Return:
            x: [sentence_len, out_dim]
        """
        sentence_len = x.size(0)
        x = x.view(sentence_len, -1) # [sentence_len, context_len x in_dim]
        x = self.net(x) # [setence_len, out_dim]
        return x


class Cnn7(nn.Module):
    """
    ConvNet kernels=[2,3,4,5] + Fully-Connected
    """
    def __init__(self, in_dim=50, hidden_dim=500):
        """
        """
        super(Cnn7, self).__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = in_dim * 4

        self.conv2 = nn.Sequential(
            nn.Conv1d(in_dim, in_dim, kernel_size=2),    # 20
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, ceil_mode=True),    # 10
            nn.Conv1d(in_dim, in_dim, kernel_size=2),    # 9
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, ceil_mode=True),    # 5
            nn.Conv1d(in_dim, in_dim, kernel_size=2),    # 4
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, ceil_mode=True),    # 2
            nn.Conv1d(in_dim, in_dim, kernel_size=2),    # 1
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(in_dim, in_dim, kernel_size=3, padding=1),    # 21
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, ceil_mode=True),    # 11
            nn.Conv1d(in_dim, in_dim, kernel_size=3, padding=1),    # 11
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, ceil_mode=True),    # 6
            nn.Conv1d(in_dim, in_dim, kernel_size=3, padding=1),    # 6
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, ceil_mode=True),    # 3
            nn.Conv1d(in_dim, in_dim, kernel_size=3),    # 1
        )


        self.conv4 = nn.Sequential(
            nn.Conv1d(in_dim, in_dim, kernel_size=4, padding=1),    # 20
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, ceil_mode=True),    # 10
            nn.Conv1d(in_dim, in_dim, kernel_size=4, padding=1),    # 9
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, ceil_mode=True),    # 5
            nn.Conv1d(in_dim, in_dim, kernel_size=4, padding=1),    # 4
            nn.ReLU(),

            nn.Conv1d(in_dim, in_dim, kernel_size=4),    # 1
        )

        self.conv5 = nn.Sequential(
            nn.Conv1d(in_dim, in_dim, kernel_size=5, padding=2),    # 21
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, ceil_mode=True),    # 11
            nn.Conv1d(in_dim, in_dim, kernel_size=5, padding=2),    # 11
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, ceil_mode=True),    # 6
            nn.Conv1d(in_dim, in_dim, kernel_size=5, padding=2),    # 6
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, ceil_mode=True),    # 3
            nn.Conv1d(in_dim, in_dim, kernel_size=5, padding=1),    # 1
        )

    def forward(self, x):
        """
        Args:
            x: [sentence_length, context_len, in_dim]
        Return:
            x: [sentence_length, in_dim * 4]
        """

        # [sentence_length, in_dim, context_len]
        x = x.transpose(1, 2)

        conv2 = self.conv2(x).squeeze(-1) # [sentence_len, in_dim]
        conv3 = self.conv3(x).squeeze(-1) # [sentence_len, in_dim]
        conv4 = self.conv4(x).squeeze(-1) # [sentence_len, in_dim]
        conv5 = self.conv5(x).squeeze(-1) # [sentence_len, in_dim]

        # [sentence_len, in_dim * 4]
        out = torch.cat([conv2, conv3, conv4, conv5], dim=1)

        return out


class Cnn8(nn.Module):
    """
    9-layer Conv NN + Batch Norm + Residual
    """
    def __init__(self, context_len=21, in_dim=64, hidden_dim=None):
        super(Cnn8, self).__init__()

        self.context_len = context_len

        # conv block 64
        self.conv_block1_1 = self.conv_block(in_dim, 2, False)
        self.conv_block1_2_1 = self.conv_block(in_dim, 1, False)
        self.conv_block1_2_2 = self.conv_block(in_dim, 1, True)
        self.pool1 = nn.MaxPool1d(kernel_size=2, padding=1, ceil_mode=True)

        # conv block 128
        self.conv_block2_1 = self.conv_block(in_dim*2, 2, False)
        self.conv_block2_2_1 = self.conv_block(in_dim*2, 1, False)
        self.conv_block2_2_2 = self.conv_block(in_dim*2, 1, True)
        self.pool2 = nn.MaxPool1d(kernel_size=2, padding=1, ceil_mode=True)

        # conv block 256
        self.conv_block3_1 = self.conv_block(in_dim*4, 2, False)
        self.conv_block3_2_1 = self.conv_block(in_dim*4, 1, False)
        self.conv_block3_2_2 = self.conv_block(in_dim*4, 1, True)
        self.pool3 = nn.MaxPool1d(kernel_size=2)

        # conv block 512
        self.conv_block4_1 = self.conv_block(in_dim*8, 2, False)
        self.conv_block4_2_1 = self.conv_block(in_dim*8, 1, False)
        self.conv_block4_2_2 = self.conv_block(in_dim*8, 1, True)
        self.pool4 = nn.MaxPool1d(kernel_size=3)
        
        self.out_dim = in_dim*16

    @classmethod
    def conv_block(cls, in_dim=64, depth=2, double=True):
        """
        Args:
            [batch_size, dim, length]
        Return:
            [batch_size, dim*2, length] if double=True
            [batch_size, dim, length] if double=False
        """
        out_dim = in_dim
        layers = []
        for i in range(depth):
            if double:
                if i == depth - 1:
                    out_dim = in_dim * 2
            layers.append(nn.Conv1d(in_dim, out_dim, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def forward(self, sentence):
        """
        Args:
            sentence: [sentence_len, context_len, embed_dim]
        Return:
            logit: [batch_size, out_dim]
        """
        # [sentence_len, embed_dim, context_len]
        x = sentence.transpose(1, 2)

        # conv block 64
        x = self.conv_block1_1(x) + x # [batch, in_dim, 21]
        x = self.conv_block1_2_1(x) + x # [batch, in_dim, 21]
        x = self.conv_block1_2_2(x) # [batch, in_dim*2, 21]
        x = self.pool1(x) # [batch, in_dim*2, 11]

        # conv block 128
        x = self.conv_block2_1(x) + x # [batch, in_dim*2, 11]
        x = self.conv_block2_2_1(x) + x # [batch, in_dim*2, 11]
        x = self.conv_block2_2_2(x) # [batch, in_dim*4, 11]
        x = self.pool2(x) # [batch, in_dim*4, 6]

        # conv block 256
        x = self.conv_block3_1(x) + x # [batch, in_dim*4, 6]
        x = self.conv_block3_2_1(x) + x # [batch, in_dim*4, 6]
        x = self.conv_block3_2_2(x) # [batch, in_dim*8, 6]
        x = self.pool3(x) # [batch, in_dim*8, 3]

        # conv block 512
        x = self.conv_block4_1(x) + x # [batch, in_dim*8, 3]
        x = self.conv_block4_2_1(x) + x # [batch, in_dim*8, 3]
        x = self.conv_block4_2_2(x) # [batch, in_dim*16, 3]
        x = self.pool4(x) # [batch_size, in_dim*16, 1]
        x = x.squeeze(-1) # [batch, in_dim*16]

        return x


class RnnEncoder(nn.Module):
    """
    RNN Encoder Module
    """
    def __init__(self, context_len=21, in_dim=1024, out_dim=1024,
                 num_layers=2, cell='gru'):
        super(RnnEncoder, self).__init__()
        
        self.hidden_dim = out_dim // 2
        
        if cell == 'gru':
            self.rnn = nn.GRU(
                input_size=in_dim,
                hidden_size=self.hidden_dim,
                num_layers=num_layers,
                dropout=0.5,
                bidirectional=True)
            
        if cell == 'lstm':
            self.rnn = nn.LSTM(
                input_size=in_dim,
                hidden_size=self.hidden_dim,
                num_layers=num_layers,
                dropout=0.5,
                bidirectional=True)
            
        elif cell == 'sru':
            from sru import SRU
            self.rnn = SRU(
                input_size=in_dim,
                hidden_size=self.hidden_dim,
                num_layers=num_layers,
                dropout=0.5,
                bidirectional=True)

    def forward(self, x):
        """
        Args:
            x: [sentence_len, context_len, input_size]
        Return:
            x: [sentence_len, hidden_size]
        """
        # input (seq_len, batch, input_size)
        # h_0 (num_layers * num_directions, batch, hidden_size)

        # output (seq_len, batch, hidden_size * num_directions)
        # h_n (num_layers * num_directions, batch, hidden_size)

        #   [sequence_len, context_len, input_size]
        # =>[sentence_len, context_len, hidden_size x 2]
        x, _ = self.rnn(x)

        # [sequence_len, hidden_size x 2]
        x = x[:, 10, :]

        return x

    
#################
# Decoder Class # 
#################

class FCDecoder(nn.Module):
    """
    Fully-Connected Decoder
    """
    def __init__(self, in_dim, hidden_dim, n_tags):
        
        assert len(dims) == 2
        
        self.net = nn.Sequential(
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden_dim, n_tags)
        )        
        
    def forward(self, x):
        """
        [sentence_len, in_dim] => [sentence_len, n_tags]
        """
        return self.net(x)
    
class RnnDecoder(nn.Module):
    """
    RNN-based Decoder
    """
    def __init__(self, in_dim=1024, hidden_dim=512, n_tags=11,
                 num_layers=2, cell='gru'):
        
        if cell == 'gru':
            self.rnn = nn.GRU(
                input_size=in_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                dropout=0.5,
                bidirectional=True)
            
        if cell == 'lstm':
            self.rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=0.5,
                bidirectional=True)
            
        elif cell == 'sru':
            from sru import SRU
            self.rnn = SRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=0.5,
                bidirectional=True)
            
        self.out = nn.Sequential(
            nn.ReLU()
            nn.Dropout()
            nn.Linear(hidden_size * 2, n_tags)
        )
        
    def forward(self, x):
        """
        [sentence_len, in_dim] => [sentence_len, n_tags]
        """
        # input (seq_len, batch, input_size)
        # h_0 (num_layers * num_directions, batch, hidden_size)

        # output (seq_len, batch, hidden_size * num_directions)
        # h_n (num_layers * num_directions, batch, hidden_size)

        # [sentence_len, batch=1, input_size]
        x = x.unsqueeze(1)

        # x: [sentence_len, batch=1, hidden_size x 2]
        # h_n: [num_layers * 2, batch=1, hidden_size]
        # c_n: [num_layers * 2, batch=1, hidden_size]
        x, _ = self.rnn(x)

        # [sequence_len, hidden_size x 2]
        x = x.squeeze(1)

        # [sequence_len, n_tags]
        x = self.out(x)

        return x        

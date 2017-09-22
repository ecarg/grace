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
import torch.autograd as autograd


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

        hidden_out = self.embeds2hidden(embeds.view(len(contexts), -1))
        hidden_relu = self.relu(hidden_out)
        hidden_drop = F.dropout(hidden_relu, training=self.training)
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

    def forward(self, contexts_gazet):    # pylint: disable=arguments-differ,too-many-locals
        contexts, gazet = contexts_gazet
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



class Fnn6(Fnn5):    # pylint: disable=too-many-instance-attributes
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
        self.START_TAG = 'BOS'
        self.STOP_TAG = 'EOS'
        voca['out'][self.START_TAG] = len(voca['out'])
        voca['out'][self.STOP_TAG] = len(voca['out'])
        super().__init__(window, voca, gazet, embed_dim, hidden_dim, phoneme, gazet_embed, pos_enc)
        # 태그셋 크기가 10이라면, 11,12번째 엔트리로 START/END를 넣은 것처럼..
        self.tagset_size = len(voca['out'])

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[voca['out'][self.START_TAG], :] = -10000
        self.transitions.data[:, voca['out'][self.STOP_TAG]] = -10000

    # Compute log sum exp in a numerically stable way for the forward algorithm
    @classmethod
    def to_scalar(cls, var):
        # returns a python float
        return var.view(-1).data.tolist()[0]

    @classmethod
    def argmax(cls, vec):
        # return the argmax as a python int
        _, idx = torch.max(vec, 1)
        return cls.to_scalar(idx)

    @classmethod
    def log_sum_exp(cls, vec):
        max_score = vec[0, cls.argmax(vec)]
        max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
        return max_score + \
            torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        if torch.cuda.is_available():
            init_alphas = torch.cuda.FloatTensor(1, self.tagset_size).fill_(-10000.)
        else:
            init_alphas = torch.FloatTensor(1, self.tagset_size).fill_(-10000.)

        # START_TAG has all of the score.
        init_alphas[0][self.voca['out'][self.START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = autograd.Variable(init_alphas)

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward variables at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(self.log_sum_exp(next_tag_var))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.voca['out'][self.STOP_TAG]]
        alpha = self.log_sum_exp(terminal_var)
        return alpha

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.Tensor(1, self.tagset_size).fill_(-10000.)
        init_vvars[0][self.voca['out'][self.START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = autograd.Variable(init_vvars)
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = self.argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id])
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.voca['out'][self.STOP_TAG]]
        best_tag_id = self.argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.voca['out'][self.START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        if torch.cuda.is_available():
            score = autograd.Variable(torch.cuda.FloatTensor([0]))
            lts = torch.cuda.LongTensor([self.voca['out'][self.START_TAG]])
        else:
            score = autograd.Variable(torch.FloatTensor([0]))
            lts = torch.LongTensor([self.voca['out'][self.START_TAG]])
        tags = torch.cat([lts, tags.data], 0)
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.voca['out'][self.STOP_TAG], tags[-1]]
        return score

    def neg_log_likelihood(self, contexts_gazet, train_labels):
        tag_out = super().forward(contexts_gazet)
        forward_score = self._forward_alg(tag_out)
        gold_score = self._score_sentence(tag_out, train_labels)

        return forward_score - gold_score

    def forward(self, contexts_gazet):    # pylint: disable=arguments-differ,too-many-locals
        tag_out = super().forward(contexts_gazet)
        _, tag_seq = self._viterbi_decode(tag_out)
        if torch.cuda.is_available():
            ret = autograd.Variable(torch.cuda.LongTensor(tag_seq))
        else:
            ret = autograd.Variable(torch.LongTensor(tag_seq))
        return ret

# coding=utf-8
"""
@Author:        李扬名
@StartTime:     18/11/18
@FileName:      test.py
@Software:      Pycharm
@LastModify:    18/11/19
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


SOS_TOKEN = 0
EOS_TOKEN = 1

# 这里是限制输入句子的长度, 且不包含标点符号.
MAX_LENGTH = 3


class Lang:

    def __init__(self, name):
        self._name = name
        self._word2index = {}
        self._index2word = {0: "SOS", 1: "EOS"}
        self._n_words = 2

    def add_sent(self, sent):
        for word in sent.split():
            self.add_word(word)

    def add_word(self, word):
        if word not in self._word2index.keys():
            self._word2index[word] = self._n_words
            self._index2word[self._n_words] = word
            self._n_words += 1

    def get_index(self, word):
        return self._word2index[word]

    def get_word(self, index):
        return self._index2word[index]

    def sent2index(self, sent, set_sos, set_eos):
        word_list = sent.strip().split()
        i_list = [self.get_index(word) for word in word_list]

        if set_sos:
            i_list.insert(0, SOS_TOKEN)
        if set_eos:
            i_list.append(EOS_TOKEN)

        return i_list

    def __len__(self):
        return self._n_words


class EncoderRNN(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super(EncoderRNN, self).__init__()
        self._hidden_dim = hidden_dim

        self._embedding = nn.Embedding(input_dim, hidden_dim)
        self._gru = nn.GRU(hidden_dim, hidden_dim)

    @property
    def hidden_dim(self):
        return self._hidden_dim

    def forward(self, input_x, hidden):
        embedded_x = self._embedding(input_x).view(1, 1, -1)
        output_x, hidden_x = self._gru(embedded_x, hidden)
        return output_x, hidden_x

    def init_hidden(self):
        return Variable(torch.zeros(1, 1, self._hidden_dim))


class DecoderRNN(nn.Module):

    def __init__(self, hidden_dim, output_dim, dropout_p=0.1):
        super(DecoderRNN, self).__init__()
        self._hidden_dim = hidden_dim

        self._embedding = nn.Embedding(hidden_dim, output_dim)
        self._gru = nn.GRU(hidden_dim, hidden_dim)
        self._out = nn.Linear(hidden_dim, output_dim)
        self._dropout = nn.Dropout(dropout_p)

    def forward(self, input_x, hidden):
        embedded_x = self._embedding(input_x).view(1, 1, -1)
        dropout_x = self._dropout(embedded_x)

        relu_x = F.relu(dropout_x)
        gru_x, out_hidden = self._gru(relu_x, hidden)
        out_x = F.softmax(self._out(gru_x[0]))
        return out_x, out_hidden

    def init_hidden(self):
        return Variable(torch.zeros(1, 1, self._hidden_dim))


class AttnDecoderRNN(nn.Module):

    def __init__(self, hidden_dim, output_dim, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self._hidden_dim = hidden_dim

        self._embedding = nn.Embedding(output_dim, hidden_dim)
        self._attn = nn.Linear(hidden_dim * 2, MAX_LENGTH + 2)
        self._combine = nn.Linear(hidden_dim * 2, hidden_dim)
        self._dropout = nn.Dropout(dropout_p)
        self._gru = nn.GRU(hidden_dim, hidden_dim)
        self._out = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_x, hidden, e_hidden):
        embedded_x = self._embedding(input_x).view(1, 1, -1)
        dropout_x = self._dropout(embedded_x)

        attn_weight = F.softmax(
            self._attn(torch.cat([dropout_x[0], hidden[0]], dim=1)), dim=1
        )

        # bmm 是 batch mm, 即 batch 矩阵乘法.
        attn_hidden = torch.bmm(
            attn_weight.unsqueeze(0), e_hidden.unsqueeze(0)
        )

        cat_x = torch.cat([dropout_x[0], attn_hidden[0]], dim=1)
        combine_x = self._combine(cat_x).unsqueeze(0)

        relu_x = F.relu(combine_x)
        gru_x, gru_hidden = self._gru(relu_x, hidden)

        out_x = F.log_softmax(self._out(gru_x[0]), dim=1)
        return out_x, gru_hidden, attn_weight

    def init_hidden(self):
        return Variable(torch.zeros(1, 1, self._hidden_dim))

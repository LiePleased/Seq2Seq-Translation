# coding=utf-8
"""
@Author:        李扬名
@StartTime:     18/11/19
@FileName:      main.py
@Software:      Pycharm
@LastModify:    18/11/19
"""

import tqdm
import json
import time
import pickle
import codecs
import random
import numpy as np

from torch import optim

from refer import *


def json_print(obj):
    """
    json.dumps 支持打印非 ascii 字符.
    """

    print json.dumps(obj, ensure_ascii=False)


eng_prefixes = (
    "i am ", "i m ", "i'm",
    "he is", "he s",
    "she is", "she s",
    "you are", "you re", "you're",
    "we are", "we re", "we're",
    "they are", "they re", "they're"
)


def str_replace(sent):
    if sent[-2] == " ":
        return sent
    else:
        return sent[:-1] + " " + sent[-1]


def read_file(file_path, reverse=False):
    left_lines = []  # 左边: 英语数据.
    right_lines = []  # 右边: 法语数据.

    with codecs.open(file_path, "r", encoding="utf-8") as fr:
        for line in fr.readlines():
            items = line.strip().split("\t")

            if len(items) == 2:
                eng = items[0].strip().lower()
                fra = items[1].strip().lower()

                for prefix in eng_prefixes:
                    if eng.startswith(prefix) and len(eng.split()) <= MAX_LENGTH:
                        left_lines.append(str_replace(eng))
                        right_lines.append(str_replace(fra))
                        break

    if reverse:
        return right_lines, left_lines
    else:
        return left_lines, right_lines


in_lines, out_lines = read_file("data/eng-fra.txt")

# 将过滤的数据保存下来.
with codecs.open("data/extract.txt", "w", encoding="utf-8") as fw:
    for in_line, out_line in zip(in_lines, out_lines):
        fw.write(in_line + u'\t' + out_line + u"\n")

try:
    with open("model/in_lang.d", "rb") as fr:
        in_lang = pickle.loads(fr.read())
    with open("model/out_lang.d", "rb") as fr:
        out_lang = pickle.loads(fr.read())
except Exception:
    in_lang, out_lang = Lang("material"), Lang("translation")

    # tqdm 可用于读取数据的产生进度条.
    for in_line, out_line in tqdm.tqdm(zip(in_lines, out_lines)):
        in_lang.add_sent(in_line)
        out_lang.add_sent(out_line)

    with open("model/in_lang.d", 'wb') as fw:
        fw.write(pickle.dumps(in_lang))
    with open("model/out_lang.d", "wb") as fw:
        fw.write(pickle.dumps(out_lang))

TEACHER_FORCING_RATIO = 0.8


def train(input_tensor, target_tensor, encoder, decoder,
          e_optimizer, d_optimizer, criterion):

    encoder.train()
    decoder.train()

    # 初始化用于迭代序列的隐向量.
    e_hidden = encoder.init_hidden()

    # 输入序列和输出序列的长度.
    input_len = input_tensor.size(0)
    target_len = target_tensor.size(0)

    # 记录编码器 encoder 的输出, 这里 MAX_LENGTH 实际
    # 上相当于做了一次 Padding, 不够长的句子补 0.
    e_output = Variable(torch.zeros(MAX_LENGTH + 2, encoder.hidden_dim))
    loss = 0

    for word_i in range(0, input_len):
        e_out, e_hidden = encoder(input_tensor[word_i], e_hidden)
        e_output[word_i] = e_out[0, 0]

    # 解码器以编码器的隐层 hidden 作为输入.
    d_hidden = e_hidden
    d_in = Variable(torch.LongTensor([[SOS_TOKEN]]))

    force_rate = random.random()
    if force_rate < TEACHER_FORCING_RATIO:
        # Teacher forcing: 将目标作为下一次 RNN 迭代的输入.

        for word_i in range(target_len):
            d_out, d_hidden, d_attn = decoder(d_in, d_hidden, e_output)
            loss += criterion(d_out, target_tensor[word_i])
            d_in = target_tensor[word_i]
    else:
        # Without forcing: 使用自身的预测作为下一次输入.

        for word_i in range(target_len):
            d_out, d_hidden, d_attn = decoder(d_in, d_hidden, e_output)
            top_v, top_i = d_out.topk(1, dim=1)

            # Detach from the gradient flow.
            d_in = top_i.squeeze().detach()

            loss += criterion(d_out, target_tensor[word_i])
            if d_in.cpu().data.numpy()[0] == EOS_TOKEN:
                break

    e_optimizer.zero_grad()  # 清理梯度缓存.
    d_optimizer.zero_grad()

    loss.backward()  # 反向求解梯度.

    e_optimizer.step()  # 更新编码器的梯度.
    d_optimizer.step()  # 更新解码器的梯度.

    return loss.cpu().data.numpy()[0] / target_len


NETWORK_HIDDEN_DIM = 128
NETWORK_DROPOUT_RATE = 0.2
LEARNING_RATE = 1e-3
TRAINING_EPOCH = 500
EPOCH_NUM_SAMPLE = 300
EVERY_SAVE_MODEL = 100

# 实例化编码器 encoder 和解码器 decoder.
try:
    e_rnn = torch.load("model/e_rnn.m")
    d_rnn = torch.load("model/d_rnn.m")
except Exception:
    e_rnn = EncoderRNN(len(in_lang), NETWORK_HIDDEN_DIM)
    d_rnn = AttnDecoderRNN(NETWORK_HIDDEN_DIM, len(out_lang), NETWORK_DROPOUT_RATE)

e_optim = optim.Adam(e_rnn.parameters(), lr=LEARNING_RATE)
d_optim = optim.Adam(d_rnn.parameters(), lr=LEARNING_RATE)
judge = nn.NLLLoss()


time.sleep(1)
print "开始训练模型."

loss_list = []
for epoch in range(0, TRAINING_EPOCH):
    total_loss = 0

    index_list = list(range(0, len(in_lines)))
    np.random.shuffle(index_list)
    train_index_list = index_list[:EPOCH_NUM_SAMPLE]
    # train_index_list = index_list

    for index in tqdm.tqdm(train_index_list):
        in_sent = in_lines[index]
        out_sent = out_lines[index]

        in_word_list = in_lang.sent2index(in_sent, False, False)
        out_word_list = out_lang.sent2index(out_sent, False, True)

        # 定义输入, 输出变量的形式.
        var_in = Variable(torch.LongTensor(in_word_list))
        var_out = Variable(torch.LongTensor(out_word_list))

        total_loss += train(
            var_in, var_out,
            e_rnn, d_rnn,
            e_optim, d_optim,
            judge
        )

        if index % EPOCH_NUM_SAMPLE == 0:
            torch.save(e_rnn, "model/e_rnn.m")
            torch.save(d_rnn, "model/d_rnn.m")

    print "[轮数: {:6d}], 总损失为 {:.6f};".format(epoch, total_loss)

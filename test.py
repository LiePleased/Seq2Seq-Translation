# coding=utf-8
"""
@Author:        李扬名
@StartTime:     18/11/19
@FileName:      test.py
@Software:      Pycharm
@LastModify:    18/11/20
"""

import pickle

# pickle 序列化模型
from refer import *


# 加载数据集.
with open("model/in_lang.d", "rb") as fr:
    in_lang = pickle.loads(fr.read())
with open("model/out_lang.d", "rb") as fr:
    out_lang = pickle.loads(fr.read())

# 加载编解码模型.
e_rnn = torch.load("model/e_rnn.m")
d_rnn = torch.load("model/d_rnn.m")

e_rnn.eval()
d_rnn.eval()


def evaluate(in_sent, max_pred_len=10):
    index_list = in_lang.sent2index(in_sent.lower(), False, False)
    var_sent = Variable(torch.LongTensor(index_list))

    input_len = var_sent.size(0)

    e_output = Variable(torch.zeros(MAX_LENGTH + 2, e_rnn.hidden_dim))
    e_hidden = e_rnn.init_hidden()

    for word_i in range(0, input_len):
        e_out, e_hidden = e_rnn(var_sent[word_i], e_hidden)
        e_output[word_i] = e_out[0, 0]

    d_hidden = e_hidden
    d_in = Variable(torch.LongTensor([[SOS_TOKEN]]))

    output_list = []
    for _ in range(0, max_pred_len):
        d_out, d_hidden, _ = d_rnn(d_in, d_hidden, e_output)
        _, top_i = d_out.topk(1, dim=1)
        d_in = top_i.squeeze()

        pred_i = top_i.cpu().data.numpy()[0][0]
        if pred_i == EOS_TOKEN:
            break
        else:
            output_list.append(pred_i)

    out_sent = [out_lang.get_word(i) for i in output_list]
    return " ".join(out_sent)


while True:
    line = raw_input("输入英文: ")

    if line[-2] != " " and not line[-1].isalpha():
        line = line[:-1] + " " + line[-1]

    answer = evaluate(line)
    print u"法语译文: " + answer[:-2] + answer[-1]
    print ""


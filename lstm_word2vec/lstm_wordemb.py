# -*- coding: utf-8 -*-
"""
Created on  2018/5/21 13:52
 学习苏剑林大神的文章，把大神的代码跑一跑https://kexue.fm/archives/3863
 文本情感分类
@author: lhua
"""

'''
word embedding测试
在GPU上，16s一轮
经过30轮迭代，训练集准确率为99.01%，测试集准确率为90.81%
Dropout不能用太多，否则信息损失太严重
'''

import numpy as np
import pandas as pd
import yaml
import jieba
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Embedding
from keras.layers import LSTM
from keras.models import model_from_yaml
import time

# batch size
batch_size = 128
# divide the test set and training set
train_num = 15000
epochs = 30
# truncation word number
maxlen = 100
# words with fewer occurrences are removed for dimension reduction
min_count = 5


# uniofrm corpus length
def doc2num(s, maxlen, word_set, abc):
    # take out the words that exist in the dictionary
    s = [i for i in s if i in word_set]
    # depending on the maximum length,you can determine whether you need to fill the empty string
    s = s[:maxlen] + [''] * max(0, maxlen - len(s))
    # return the sequential encoding of the word
    return list(abc[s])


# save the word dictionary
def savewordDictionary(abc):
    try:
        f_open = open('/data/dictionary.txt', 'w')
        for item in abc.index:
            f_open.write(item + "\t" + str(abc[item]) + "\n")
        f_open.close()
    except:
        print("save the word dictionary fialed!")


# read the word dictionary for prediction
def readwordDictionary():
    f_open = open('/data/dictionary.txt', 'r')
    word_dict = {}
    for line in f_open:
        try:
            lineinfo = line.strip().split("\t")
            if len(lineinfo) > 1:
                index = lineinfo[0]
                content = lineinfo[1]
                word_dict[index] = content
        except:
            print("read word dictionary,line is failed :" + line)
    if len(word_dict) > 0:
        abc = pd.Series(word_dict)
        abc[''] = 0
        return abc

    else:
        print("read word dictionary failed !")


def loaddata():
    print("loading data...")
    # set up the label
    pos = pd.read_excel('/data/pos.xls', header=None)
    pos['label'] = 1
    neg = pd.read_excel('/data/neg.xls', header=None)
    neg['label'] = 0
    # merge the positive data and the negative data
    all_ = pos.append(neg, ignore_index=True)
    # use jieba tools for participles,the lstm model dose not need to remove the stop words
    all_['words'] = all_[0].apply(lambda s: list(jieba.cut(s)))

    content = []
    for i in all_['words']:
        content.extend(i)

    # calculate the frequency of each word,turn list to the Series of pandas
    abc = pd.Series(content).value_counts()
    abc = abc[abc >= min_count]
    # encode the dictionary in order
    abc[:] = list(range(1, len(abc) + 1))
    abc[''] = 0
    # save the word dictionary
    savewordDictionary(abc)
    # get the reserved words
    word_set = set(abc.index)

    # get the input data of the model
    all_['doc2num'] = all_['words'].apply(lambda s: doc2num(s, maxlen, word_set, abc))

    # manually scrambled data
    idx = list(range(len(all_)))
    np.random.shuffle(idx)
    all_ = all_.loc[idx]

    # change data format according to the model input data format
    x = np.array(list(all_['doc2num']))
    y = np.array(list(all_['label']))
    # adjust the label data format
    y = y.reshape((-1, 1))
    return x, y, len(abc)


# train model
def trainmodel(x, y, input_dim):
    print("train model ...")
    # build the model
    model = Sequential()
    model.add(Embedding(input_dim, 256, input_length=maxlen))
    model.add(LSTM(128))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.summary()

    model.fit(x[:train_num], y[:train_num], batch_size=batch_size, nb_epoch=epochs)

    score = model.evaluate(x[train_num:], y[train_num:], batch_size=batch_size)
    print("accuracy is ")
    print(score)
    # save the trained lstm model
    yaml_string = model.to_yaml()
    with open('/data/lstm_emb.yml', 'w') as outfile:
        outfile.write(yaml.dump(yaml_string, default_flow_style=True))
    model.save_weights('/data/lstm_emb.h5')


def predict_one(s):  # 单个句子的预测函数
    print('loading model......')
    with open('/data/lstm_emb.yml', 'r') as f:
        yaml_string = yaml.load(f)
    model = model_from_yaml(yaml_string)

    print('loading weights......')
    model.load_weights('/data/lstm_emb.h5')
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    abc = readwordDictionary()
    word_set = set(abc.index)
    s = np.array(doc2num(list(jieba.cut(s)), maxlen, word_set, abc))
    s = s.reshape((1, s.shape[0]))
    print(s)
    return model.predict_classes(s, verbose=0)[0][0]


if __name__ == '__main__':
    start = time.clock()
    x, y, input_dim = loaddata()
    trainmodel(x, y, input_dim)
    start0 = time.clock()
    print("load data and train model cost time : " + str(start0 - start))
    # string='电池充完了电连手机都打不开.简直烂的要命.真是金玉其外,败絮其中!连5号电池都不如'
    # string='牛逼的手机，从3米高的地方摔下去都没坏，质量非常好'
    # string='酒店的环境非常好，价格也便宜，值得推荐'
    s = '屏幕较差，拍照也很粗糙。'
    # string='我是傻逼'
    # string='你是傻逼'
    # string = '屏幕较差，拍照也很粗糙。'
    # string='质量不错，是正品 ，安装师傅也很好，才要了83元材料费'
    # string='东西非常不错，安装师傅很负责人，装的也很漂亮，精致，谢谢安装师傅！'
    print('predict result: ' + str(predict_one(s)))
    start1 = time.clock()
    print("load data and train model cost time : " + str(start1 - start0))
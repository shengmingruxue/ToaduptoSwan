# -*- coding: utf-8 -*-
"""
Created on  2018/5/18 13:30

@author: lhua
"""
import imp
import sys

imp.reload(sys)
import numpy as np
import pandas as pd
import jieba
import re
from gensim.models import Word2Vec
from keras.preprocessing import sequence
from gensim.corpora.dictionary import Dictionary
import multiprocessing
from sklearn.model_selection import train_test_split
import yaml
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout, Activation
from keras.models import model_from_yaml

np.random.seed(1337)  # For Reproducibility
# the dimension of word vector
vocab_dim = 300
# sentence length
maxlen = 100
# iter num
n_iterations = 1
# the number of words appearing
n_exposures = 10
# what is the maximum distance between the current word and the prediction word in a sentence, what is the maximum distance between the current and the prediction word in a sentence
window_size = 7
# batch size
batch_size = 32
# epoch num
n_epoch = 20
# input length
input_length = 100
# multi processing cpu number
cpu_count = multiprocessing.cpu_count()


# loading training file
def loadfile():
    neg = pd.read_excel('/data/liuhua/code/kerasTest/data/neg.xls', header=None, index=None)
    pos = pd.read_excel('/data/liuhua/code/kerasTest/data/pos.xls', header=None, index=None)
    #merge all data
    neg = np.array(neg[0])
    post = np.array(pos[0])
    return neg,post

#generating set of disused words
def getstopword(stopwordPath):
    stoplist = set()
    for line in stopwordPath:
        stoplist.add(line.strip())
        # print line.strip()
    return stoplist

#divide the sentence and remove the disused words
def wordsege(text):
    # get disused words set
    stopwordPath = open('/data/liuhua/code/kerasTest/data/stopwords1.txt', 'r')
    stoplist = getstopword(stopwordPath)
    stopwordPath.close()

    # divide the sentence and remove the disused words with jieba,return list
    text_list = []
    for document in text:

        seg_list = jieba.cut(document.strip())
        fenci = []

        for item in seg_list:
            if item not in stoplist and re.match(r'-?\d+\.?\d*', item) == None and len(item.strip()) > 0:
                fenci.append(item)
        # if the word segmentation of the sentence is null,the label of the sentence should be deleted accordingly
        if len(fenci) > 0:
            text_list.append(fenci)
    return text_list
def tokenizer(neg, post):
    neg_sege = wordsege(neg)
    post_sege = wordsege(post)
    combined = np.concatenate((post_sege,neg_sege))
    # generating label and meging label data
    y = np.concatenate((np.ones(len(post_sege), dtype=int), np.zeros(len(neg_sege), dtype=int)))
    return combined,y


# create a dictionary of words and phrases,return the index of each word,vector of words,and index of words corresponding to each sentence
def create_dictionaries(model=None,
                        combined=None):
    ''' Function does are number of Jobs:
        1- Creates a word to index mapping
        2- Creates a word to vector mapping
        3- Transforms the Training and Testing Dictionaries
    '''
    if (combined is not None) and (model is not None):
        gensim_dict = Dictionary()
        gensim_dict.doc2bow(model.wv.vocab.keys(),
                            allow_update=True)
        # the index of a word which have word vector is not 0
        w2indx = {v: k + 1 for k, v in gensim_dict.items()}
        # integrate all the corresponding word vectors into the word vector matrix
        w2vec = {word: model[word] for word in w2indx.keys()}

        # a word without a word vector is indexed 0,return the index of word
        def parse_dataset(combined):
            ''' Words become integers
            '''
            data = []
            for sentence in combined:
                new_txt = []
                for word in sentence:
                    try:
                        new_txt.append(w2indx[word])
                    except:
                        new_txt.append(0)
                data.append(new_txt)
            return data

        combined = parse_dataset(combined)
        # unify the length of the sentence with the pad_sequences function of keras
        combined = sequence.pad_sequences(combined, maxlen=maxlen)
        # return index, word vector matrix and the sentence with an unifying length and indexed
        return w2indx, w2vec, combined
    else:
        print('No data provided...')


# the training of the word vector
def word2vec_train(combined):
    model = Word2Vec(size=vocab_dim,
                     min_count=n_exposures,
                     window=window_size,
                     workers=cpu_count,
                     iter=n_iterations)
    # build the vocabulary dictionary
    model.build_vocab(combined)
    # train the word vector model
    model.train(combined, total_examples=model.corpus_count, epochs=50)
    # save the trained model
    model.save('/data/liuhua/code/kerasTest/data/Word2vec_model.pkl')
    # index, word vector matrix and the sentence with an unifying length and indexed based on the trained model
    index_dict, word_vectors, combined = create_dictionaries(model=model, combined=combined)

    return index_dict, word_vectors, combined


def get_data(index_dict, word_vectors, combined, y):
    # total number of word including the word without word vector
    n_symbols = len(index_dict) + 1
    # build word vector matrix which corresponding to the word index one by one
    embedding_weights = np.zeros((n_symbols, vocab_dim))
    for word, index in index_dict.items():  # 从索引为1的词语开始，对每个词语对应其词向量
        embedding_weights[index, :] = word_vectors[word]
    # partition test set and training set
    x_train, x_test, y_train, y_test = train_test_split(combined, y, test_size=0.2)
    print(x_train.shape, y_train.shape)
    # return the input parameters needed of the lstm model
    return n_symbols, embedding_weights, x_train, y_train, x_test, y_test


##定义网络结构
def train_lstm(n_symbols, embedding_weights, x_train, y_train, x_test, y_test):
    print('Defining a Simple Keras Model...')
    model = Sequential()  # or Graph or whatever
    model.add(Embedding(output_dim=vocab_dim,
                        input_dim=n_symbols,
                        mask_zero=True,
                        weights=[embedding_weights],
                        input_length=input_length))  # Adding Input Length
    model.add(LSTM(output_dim=50, activation='sigmoid', inner_activation='hard_sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    print ('Compiling the Model...')
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    model.summary()
    print ("Train...")
    model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=n_epoch, verbose=1)

    print ("Evaluate...")
    score = model.evaluate(x_test, y_test,
                           batch_size=batch_size)
    # save the trained lstm model
    yaml_string = model.to_yaml()
    with open('/data/liuhua/code/kerasTest/data/lstm.yml', 'w') as outfile:
        outfile.write(yaml.dump(yaml_string, default_flow_style=True))
    model.save_weights('/data/liuhua/code/kerasTest/data/lstm.h5')
    print ('Test score:', score)


# 训练模型，并保存
def train():
    print ('Loading Data...')
    neg, post = loadfile()

    print('Tokenising...')
    combined,y = tokenizer(neg, post)
    print(len(combined), len(y))
    print('Training a Word2vec model...')
    index_dict, word_vectors, combined = word2vec_train(combined)
    print('Setting up Arrays for Keras Embedding Layer...')
    n_symbols, embedding_weights, x_train, y_train, x_test, y_test = get_data(index_dict, word_vectors, combined, y)
    print(x_train.shape, y_train.shape)
    train_lstm(n_symbols, embedding_weights, x_train, y_train, x_test, y_test)

# building the input format data
def input_transform(string):
    words = jieba.cut(string)
    # reshape the list to bilayer list
    words = np.array(words).reshape(1, -1)
    model = Word2Vec.load('/data/liuhua/code/kerasTest/data/Word2vec_model.pkl')
    # create a dictionary of words and phrases,return the index of each word,vector of words,and index of words corresponding to each senten
    _, _, combined = create_dictionaries(model, words)
    return combined


def lstm_predict(string):
    print('loading model......')
    with open('/data/liuhua/code/kerasTest/data/lstm.yml', 'r') as f:
        yaml_string = yaml.load(f)
    model = model_from_yaml(yaml_string)

    print('loading weights......')
    model.load_weights('/data/liuhua/code/kerasTest/data/lstm.h5')
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    data = input_transform(string)
    data.reshape(1, -1)
    # predict the new data
    result = model.predict_classes(data)
    if result[0][0] == 1:
        print(string, ' positive')
    else:
        print(string, ' negative')


if __name__ == '__main__':
    #train()
    # string='电池充完了电连手机都打不开.简直烂的要命.真是金玉其外,败絮其中!连5号电池都不如'
    # string='牛逼的手机，从3米高的地方摔下去都没坏，质量非常好'
    # string='酒店的环境非常好，价格也便宜，值得推荐'
    string='屏幕较差，拍照也很粗糙。'
    # string='我是傻逼'
    # string='你是傻逼'
    # string = '屏幕较差，拍照也很粗糙。'
    # string='质量不错，是正品 ，安装师傅也很好，才要了83元材料费'
    # string='东西非常不错，安装师傅很负责人，装的也很漂亮，精致，谢谢安装师傅！'

    lstm_predict(string)

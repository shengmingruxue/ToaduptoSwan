# -*- coding: utf-8 -*-
"""
Created on  2018/5/17 16:42

@author: lhua
"""
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.externals import joblib
import xlrd
import jieba
import re

n_dim = 300


# get the set of disused words
def getstopword(stopwordPath):
    stoplist = set()
    for line in stopwordPath:
        stoplist.add(line.strip())
        # print line.strip()
    return stoplist


# participle and removal of discontinuation words
def cutStopword(x, stoplist):
    seg_list = jieba.cut(x.strip())
    fenci = []

    for item in seg_list:
        if item not in stoplist and re.match(r'-?\d+\.?\d*', item) == None and len(item.strip()) > 0:
            fenci.append(item)
    return fenci


# read data files,get training data and test data
def loadfile():
    neg = pd.read_excel('/data/neg.xls', header=None, index=None)
    pos = pd.read_excel('/data/pos.xls', header=None, index=None)
    stopwordPath = open('/data/stopwords1.txt', 'r')
    stoplist = getstopword(stopwordPath)

    pos['words'] = pos[0].apply(cutStopword, args=(stoplist,))
    neg['words'] = neg[0].apply(cutStopword, args=(stoplist,))
    print(pos['words'][:10])

    # use 1 for positive sentiment,0 for negative
    y = np.concatenate((np.ones(len(pos)), np.zeros(len(neg))))
    x = np.concatenate((pos['words'], neg['words']))
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    np.save('/data/y_train.npy', y_train)
    np.save('/data/y_test.npy', y_test)
    return x, x_train, x_test, y_train, y_test


# get summation of word vectors of all word in a copus,and then get the average,as the input of the model
def buildWordVector(text, size, imdb_w2v):
    vec = np.zeros(size).reshape((1, size))
    count = 0
    for word in text:
        try:
            vec += imdb_w2v[word].reshape((1, size))
            count += 1
        except KeyError:
            continue
    if count != 0:
        vec /= count

    return vec


# calculating test set and training set
def get_train_vecs(x, x_train, x_test):
    # Initialize model and build vocab
    imdb_w2v = Word2Vec(size=n_dim, min_count=10, seed=1)
    imdb_w2v.build_vocab(x)
    # Train the model over train_reviews (this may take several minutes)
    imdb_w2v.train(x, total_examples=imdb_w2v.corpus_count, epochs=50)
    imdb_w2v.save('/data/w2v_model.pkl')
    train_vecs = np.concatenate([buildWordVector(z, n_dim, imdb_w2v) for z in x_train])
    # train_vecs = scale(train_vecs)

    np.save('/data/train_vecs.npy', train_vecs)
    print(train_vecs.shape)
    # Train word2vec on test tweets
    # imdb_w2v.train(x_test)

    # Build test tweet vectors then scale
    test_vecs = np.concatenate([buildWordVector(z, n_dim, imdb_w2v) for z in x_test])
    # test_vecs = scale(test_vecs)
    np.save('/data/test_vecs.npy', test_vecs)
    print(test_vecs.shape)
    return train_vecs, test_vecs


# train svm model with sklearn
def svm_train(train_vecs, y_train, test_vecs, y_test):
    clf = SVC(kernel='rbf', verbose=True)
    clf.fit(train_vecs, y_train)
    joblib.dump(clf, '/data/model.pkl')
    print(clf.score(test_vecs, y_test))


# load word2vec and smv model and use them to predict
def svm_predict(str):
    clf = joblib.load('/data/model.pkl')
    model = Word2Vec.load('/data/w2v_model.pkl')
    stopwordPath = open('/data/stopwords1.txt', 'r')
    stoplist = getstopword(stopwordPath)
    str_sege = cutStopword(str, stoplist)
    str_pre = np.array(str_sege).reshape(1, -1)
    str_vecs = np.concatenate([buildWordVector(z, n_dim, model) for z in str_pre])
    pred_result = clf.predict(str_vecs)
    print(pred_result)


if __name__ == '__main__':
    print("loading data ...")
    x, x_train, x_test, y_train, y_test = loadfile()
    print("train word2vec model and get the input of svm model")
    train_vecs, test_vecs = get_train_vecs(x, x_train, x_test)
    print("train svm model...")
    svm_train(train_vecs, y_train, test_vecs, y_test)

    print("use svm model to predict...")
    str = '屏幕较差，拍照也很粗糙。'
    # str ='质量不错，是正品 ，安装师傅也很好，才要了83元材料费'
    # str ='东西非常不错，安装师傅很负责人，装的也很漂亮，精致，谢谢安装师傅！'
    svm_predict(str)
# -*- coding: utf-8 -*-
"""
Created on  2018/5/23 16:58
 https://blog.csdn.net/u013518890/article/details/74938758
https://machinelearningmastery.com/understanding-stateful-lstm-recurrent-neural-networks-python-keras/
 我们需要将NumPy数组重新构造为LSTM网络所期望的格式，即[samples示例, time steps时间步数, features特征]。
 在相同的条件下：
 格式为：[[[A],[B],[C]],[[C],[D],[E]],...],准确率100%
 若格式为[[[A,B,C]],[[C,D,E]],...]则没有用到序列关系，准确率86.96%
 对输入数据的reshape是将输入序列作为一个特性的time step序列，而不是多个特性的单一time step。
 也就是说我们把ABC看成独立的一个特征组成的多个时间序列，而不是把ABC看成一个多个特征组成一个时间序列。
@author: lhua
"""
# Naive LSTM to learn three-char window to one-char mapping
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils import np_utils
# fix random seed for reproducibility
numpy.random.seed(7)
# define the raw dataset
alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
# create mapping of characters to integers (0-25) and the reverse
char_to_int = dict((c, i) for i, c in enumerate(alphabet))
int_to_char = dict((i, c) for i, c in enumerate(alphabet))
# prepare the dataset of input to output pairs encoded as integers
seq_length = 3
dataX = []
dataY = []
for i in range(0, len(alphabet) - seq_length, 1):
    seq_in = alphabet[i:i + seq_length]
    seq_out = alphabet[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])
    print(seq_in, '->', seq_out)
# importance !!!! reshape X to be [samples, time steps, features]
#X = numpy.reshape(dataX, (len(dataX),1,seq_length)) this is wrong
X = numpy.reshape(dataX, (len(dataX),seq_length,1))
# normalize
X = X / float(len(alphabet))
print(X.shape)
# one hot encode the output variable
y = np_utils.to_categorical(dataY)
print(y.shape)
# create and fit the model
model = Sequential()
model.add(LSTM(32, input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=500, batch_size=1, verbose=2)
model.summary()
# summarize performance of the model
scores = model.evaluate(X, y, verbose=0)
print("Model Accuracy: %.2f%%" % (scores[1]*100))
# demonstrate some model predictions
for pattern in dataX:
    x = numpy.reshape(pattern, (1, seq_length, 1))
    x = x / float(len(alphabet))
    prediction = model.predict(x, verbose=0)
    index = numpy.argmax(prediction)
    result = int_to_char[index]
    seq_in = [int_to_char[value] for value in pattern]
    print(seq_in, "->", result)
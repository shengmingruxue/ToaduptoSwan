# -*- coding: utf-8 -*-
"""
Created on  2018/3/13 18:15

@author: lhua

"""
# Python提供了__future__模块，把下一个新版本的特性导入到当前版本，
# 于是我们就可以在当前版本中测试一些新版本的特性。详见廖雪官网
from __future__ import print_function

# 先翻译下开头：
# 用MNIST手写数字识别数据训练了一个简单的CNN模型。在迭代12次之后得到了99.25%的准确率
# （参数还有调整的余地）在Grid K520的GPU上跑一轮需要12秒
'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''
# 引入需要的包：数据集，序贯（Sequential），全连接层（Dense）,Dropout层，Flatten层，二维卷积层（Conv2D）,
# 空域信号最大池化层（MaxPooling2D），后端（backend,这里keras的backend选用的是tensorflow,详见 vi ~/.keras/keras.json 配置文件里面可以修改backend)
# tensorflow as tf 为了最后清空模型所占的内存
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import tensorflow as tf
import numpy as np

# 定义一些超参数（可调）
# batch_size为每次喂给神经网络的样本数。1>可以一次性将全部样本喂给神经网络，让神经网络用全部样本计算迭代梯度（即传统的梯度下降法）
# 2> 一次只喂给一个样本，让神经网络一个样本一个样本的迭代参数；3>每次将一部分样本喂给神经网络，让神经网络一部分样本一部分样本的迭代（即batch梯度下降法）
batch_size = 128

# num_classes：指的是label的维度数，这里用了10维onehot编码向量，即1:[1,0,0,0,0,0,0,0,0,0] 2:[0,1,0,0,0,0,0,0,0,0]etc
num_classes = 10

# epochs:所以训练数据完整的过一遍的次数，即样本轮多少次，或者多少波吧，训练打印日志就能看出来，这个是啥意思啦！
epochs = 12

# 图像尺寸28*28(input image dimensions)
img_rows, img_cols = 28, 28

# 下载MNIST数据集，同时获取训练集、验证集（the data, split between train and test sets）
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
from keras.utils.data_utils import get_file

path = '/data/mnist.npz'
f = np.load(path)
x_train, y_train = f['x_train'], f['y_train']
x_test, y_test = f['x_test'], f['y_test']
f.close()
# 在如何表示一组彩色图片的问题上，Theano和TensorFlow发生了分歧，'th'模式，
# 也即Theano模式会把100张RGB三通道的16×32（高为16宽为32）彩色图表示为下面这种形式（100,3,16,32），
# Caffe采取的也是这种方式。第0个维度是样本维，代表样本的数目，第1个维度是通道维，代表颜色通道数。后面两个就是高和宽了。
# 这种theano风格的数据组织方法，称为“channels_first”，即通道维靠前。而TensorFlow，的表达形式是（100,16,32,3），
# 即把通道维放在了最后，这种数据组织方式称为“channels_last”。（详见：vi ~/.keras/keras.json)
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

# 对训练和测试数据处理，转为float
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# 对数据进行归一化到0-1 因为图像数据最大是255
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# 将标签（label)进行转换为one-hot编码（convert class vectors to binary class matrices）
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# 搭建网络结构
# 序贯（Sequential)模型是多个网络层的线性堆叠，也就是一条路走到黑
model = Sequential()

# 1.可以通过向Sequential模型传递一个layer的list来构造该模型 2.可以通过.add()方法一个个的将layer加入模型
# 这里采取了通过.add()方法一个个的将layer层加入模型
# 卷积层1.一维卷积层（即时域卷积Conv1D层），用以在一维输入信号上进行邻域滤波。当使用该层作为首层时，需要提供关键字参数input_shape。
# 例如(10,128)代表一个长为10的序列，序列中每个信号为128向量。而(None, 128)代表变长的128维向量序列。
# 该层生成将输入信号与卷积核按照单一的空域（或时域）方向进行卷积。如果use_bias=True，则还会加上一个偏置项，若activation不为None，则输出为经过激活函数的输出。
# 卷积层2.二维卷积层，（即对图像的空域卷积Conv2D层）。该层对二维输入进行滑动窗卷积，当使用该层作为第一层时，应提供input_shape参数。
# 例如input_shape = (128,128,3)代表128*128的彩色RGB图像（data_format='channels_last'）
# 还有SeparableConv2D层、Conv2DTranspose层、Conv3D层、Cropping1D层、Cropping2D层、Cropping3D层、UpSampling1D层、UpSampling2D层、UpSampling3D层
# ZeroPadding1D层、ZeroPadding2D层、ZeroPadding3D层暂时没有接触过，所以就不再赘述，详见keras文档吧！
# 这里添加的是Conv2D层，卷积核数目（即输出的维度）为32,卷积核的宽度和长度均为3，移动步长默认为1，激活函数为relu,use_bias默认有偏置项
# keras的后端是tensorflow，所以data_format选取的是'channel_last'模式
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))

# 再添加一层Conv2D层，卷积核数目（即输出维度）为64，卷积核长宽均为3，激活函数还是relu,use_bias默认有偏置项
model.add(Conv2D(64, (3, 3), activation='relu'))

# 池化层（Pooling层) 的本质，其实是采样。Pooling 对于输入的 Feature Map，选择某种方式对其进行压缩。
# 例如：表示的就是对 Feature Map 2 * 2 邻域内的值，选择最大值输出到下一层，这叫做 Max Pooling。
# 于是一个 2N * 2N 的 Feature Map 被压缩到了 N * N 。
# Pooling 的意义，主要有两点：其中一个显而易见，就是减少参数。通过对 Feature Map 降维，有效减少后续层需要的参数
# 另一个则是 Translation Invariance。它表示对于 Input，当其中像素在邻域发生微小位移时，Pooling Layer 的输出是不变的。
# 这就使网络的鲁棒性增强了，有一定抗扰动的作用
model.add(MaxPooling2D(pool_size=(2, 2)))

# dropout layer的目的是为了防止CNN 过拟合，只需要按一定的概率（retaining probability）p 来对weight layer 的参数进行随机采样，
# 将这个子网络作为此次更新的目标网络。可以想象，如果整个网络有n个参数，那么我们可用的子网络个数为 2^n 。 并且，当n很大时，每次迭代更新
# 使用的子网络基本上不会重复，从而避免了某一个网络被过分的拟合到训练集上。
model.add(Dropout(0.25))

# Flatten层用来将输入“压平”，即把多维的输入一维化，常用在从卷积层到全连接层的过渡。
model.add(Flatten())

# Dense就是常用的全连接层，所实现的运算是output = activation(dot(input, kernel)+bias)。其中activation是逐元素计算的激活函数，
# kernel是本层的权值矩阵，bias为偏置向量，只有当use_bias=True才会添加。
# 这里输出维度为128，激活函数为relu,默认use_bias使用偏置项
model.add(Dense(128, activation='relu'))

# dropout layer的目的是为了防止CNN 过拟合，只需要按一定的概率（retaining probability）p 来对weight layer 的参数进行随机采样，
# 将这个子网络作为此次更新的目标网络。可以想象，如果整个网络有n个参数，那么我们可用的子网络个数为 2^n 。 并且，当n很大时，每次迭代更新
# 使用的子网络基本上不会重复，从而避免了某一个网络被过分的拟合到训练集上。
model.add(Dropout(0.5))

# 将Densen作为输出层，激活函数为softmax,输出维度与我们label处理的one-hot维度一致
# 激活函数有softmax,elu,selu: 可伸缩的指数线性单元（Scaled Exponential Linear Unit），参考Self-Normalizing Neural Networks
# softplus,softsign,relu,tanh,sigmoid,hard_sigmoid,linear
model.add(Dense(num_classes, activation='softmax'))

# 打印出模型概况,它实际调用的是keras.utils.print_summary
# 形如：
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# conv2d_1 (Conv2D)            (None, 26, 26, 32)        320
# _________________________________________________________________
# conv2d_2 (Conv2D)            (None, 24, 24, 64)        18496
# _________________________________________________________________
# max_pooling2d_1 (MaxPooling2 (None, 12, 12, 64)        0
# _________________________________________________________________
# dropout_1 (Dropout)          (None, 12, 12, 64)        0
# _________________________________________________________________
# flatten_1 (Flatten)          (None, 9216)              0
# _________________________________________________________________
# dense_1 (Dense)              (None, 128)               1179776
# _________________________________________________________________
# dropout_2 (Dropout)          (None, 128)               0
# _________________________________________________________________
# dense_2 (Dense)              (None, 10)                1290
# =================================================================
# Total params: 1,199,882
# Trainable params: 1,199,882
# Non-trainable params: 0
model.summary()

# 编译：优化器optimizer，该参数可指定为已预定义的优化器名，如rmsprop、adagrad，或一个Optimizer类的对象
# 损失函数loss，该参数为模型试图最小化的目标函数，它可为预定义的损失函数名，如categorical_crossentropy、mse，也可以为一个损失函数。
# 指标列表metrics：对分类问题，我们一般将该列表设置为metrics=['accuracy']。
# 指标可以是一个预定义指标的名字,也可以是一个用户定制的函数.指标函数应该返回单个张量,或一个完成metric_name - > metric_value映射的字典.
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
# 训练模型
# fit函数返回一个History的对象，其History.history属性记录了损失函数和其他指标的数值随epoch变化的情况，
# 如果有验证集的话，也包含了验证集的这些指标变化情况
# batch_size：整数，指定进行梯度下降时每个batch包含的样本数
# pochs：整数，训练终止时的epoch值，训练将在达到该epoch值时停止，当没有设置initial_epoch时，
# 它就是训练的总轮数，否则训练的总轮数为epochs - inital_epoch
# verbose：日志显示，0为不在标准输出流输出日志信息，1为输出进度条记录，2为每个epoch输出一行记录
# validation_data：形式为（X，y）的tuple，是指定的验证集。
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

# evaluate函数按batch计算在某些输入数据上模型的误差,返回一个测试误差的标量值（如果模型没有其他评价指标），
# 或一个标量的list（如果模型还有其他的评价指标）。model.metrics_names将给出list中各个值的含义。
score = model.evaluate(x_test, y_test, verbose=0)
print(model.metrics_names)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# 使用完模型之后，清空之前model占用的内存
K.clear_session()
tf.reset_default_graph()

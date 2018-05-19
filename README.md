# ToaduptoSwan
**********************************************************************************************************************
Work hard every day and become a big cow in the morning and evening.
**********************************************************************************************************************

lstm_word2vec.py主要是用gensim训练word2vec，然后使用keras的lstm模型对评论数据进行情感分析
svm_word2vec.py主要是用gensim训练word2vec，然后使用sklearn的svm模型对评论数据进行情感分析
将传统与神经网络模型进行对比，代码大部分非我自己所写，出自https://github.com/BUPTLdy/Sentiment-Analysis这位大神之手，我只是进行了小部分修改，例如添加了去停用词的部分。
在理解代码的过程中发现了一些比较好的博文：
https://buptldy.github.io/2016/07/20/2016-07-20-sentiment%20analysis/
http://blog.sina.com.cn/s/blog_1450ac3c60102x79x.html
https://kexue.fm/archives/3863
https://zybuluo.com/hanbingtao/note/581764
https://juejin.im/entry/5acc23f26fb9a028d1416bb3

从训练结果来看，svm准确率能达到86%，而lstm能达到90%，并没有像其原文所说的那样达到92%（可能lstm模型这个还需要调整下）

lstm 训练以及测试结果（测试语料：str = '屏幕较差，拍照也很粗糙。'）
Using TensorFlow backend.

_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_1 (Embedding)      (None, 100, 300)          2141100   
_________________________________________________________________
lstm_1 (LSTM)                (None, 50)                70200     
_________________________________________________________________
dropout_1 (Dropout)          (None, 50)                0         
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 51        
_________________________________________________________________
activation_1 (Activation)    (None, 1)                 0         
=================================================================

Test score: [0.53292876407062173, 0.90011862418829758]

屏幕较差，拍照也很粗糙。  positive



svm训练以及测试结果（测试语料：str = '屏幕较差，拍照也很粗糙。'）
[LibSVM]0.860696517413
屏幕较差，拍照也很粗糙。
[ 0.]


使用的工具版本情况：
Python 3.5.4
gensim (2.3.0)
jieba (0.36)
Keras (2.0.5)
numpy (1.13.1)
pandas (0.20.3)
scikit-learn (0.19.1)
scipy (0.19.1)
sklearn (0.0)
tensorflow (1.2.1)
xlrd (1.1.0)





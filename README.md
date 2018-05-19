# ToaduptoSwan
**********************************************************************************************************************
Work hard every day and become a big cow in the morning and evening.
**********************************************************************************************************************

lstmword2vec.py主要是用gensim训练word2vec，然后使用keras的lstm模型对评论数据进行情感分析
svmword2vec.py主要是用gensim训练word2vec，然后使用sklearn的svm模型对评论数据进行情感分析
将传统与神经网络模型进行对比，代码大部分非我自己所写，出自https://github.com/BUPTLdy/Sentiment-Analysis这位大神之手，我只是进行了小部分修改，例如添加了去停用词的部分。
在理解代码的过程中发现了一些比较好的博文：
https://buptldy.github.io/2016/07/20/2016-07-20-sentiment%20analysis/
http://blog.sina.com.cn/s/blog_1450ac3c60102x79x.html
https://kexue.fm/archives/3863
https://zybuluo.com/hanbingtao/note/581764

从训练结果来看，svm准确率能达到86%，而lstm能达到90%，并没有像其原文所说的那样达到92%（可能lstm模型这个还需要调整下）
lstm 训练以及测试结果
Using TensorFlow backend.
Loading Data...
Tokenising...
Building prefix dict from /data/liuhua/env3.5/lib/python3.5/site-packages/jieba/dict.txt ...
Loading model from cache /tmp/jieba.cache
Loading model cost 0.9321155548095703 seconds.
Prefix dict has been built succesfully.
21073 21073
Training a Word2vec model...
Setting up Arrays for Keras Embedding Layer...
(16858, 100) (16858,)
(16858, 100) (16858,)
Defining a Simple Keras Model...
2018-05-19 20:10:21.559406: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
2018-05-19 20:10:21.559465: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
2018-05-19 20:10:21.559477: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
2018-05-19 20:10:21.559519: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX2 instructions, but these are available on your machine and could speed up CPU computations.
2018-05-19 20:10:21.559536: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use FMA instructions, but these are available on your machine and could speed up CPU computations.
2018-05-19 20:10:22.009069: I tensorflow/core/common_runtime/gpu/gpu_device.cc:940] Found device 0 with properties: 
name: Tesla P40
major: 6 minor: 1 memoryClockRate (GHz) 1.531
pciBusID 0000:08:00.0
Total memory: 22.38GiB
Free memory: 22.21GiB
2018-05-19 20:10:22.419658: W tensorflow/stream_executor/cuda/cuda_driver.cc:523] A non-primary context 0x42f1870 exists before initializing the StreamExecutor. We haven't verified StreamExecutor works with that.
2018-05-19 20:10:22.421585: I tensorflow/core/common_runtime/gpu/gpu_device.cc:940] Found device 1 with properties: 
name: Tesla P40
major: 6 minor: 1 memoryClockRate (GHz) 1.531
pciBusID 0000:0b:00.0
Total memory: 22.38GiB
Free memory: 22.21GiB
2018-05-19 20:10:22.825439: W tensorflow/stream_executor/cuda/cuda_driver.cc:523] A non-primary context 0x4267c20 exists before initializing the StreamExecutor. We haven't verified StreamExecutor works with that.
2018-05-19 20:10:22.827339: I tensorflow/core/common_runtime/gpu/gpu_device.cc:940] Found device 2 with properties: 
name: Tesla P40
major: 6 minor: 1 memoryClockRate (GHz) 1.531
pciBusID 0000:0e:00.0
Total memory: 22.38GiB
Free memory: 22.21GiB
2018-05-19 20:10:23.264829: W tensorflow/stream_executor/cuda/cuda_driver.cc:523] A non-primary context 0x41ce7f0 exists before initializing the StreamExecutor. We haven't verified StreamExecutor works with that.
2018-05-19 20:10:23.266738: I tensorflow/core/common_runtime/gpu/gpu_device.cc:940] Found device 3 with properties: 
name: Tesla P40
major: 6 minor: 1 memoryClockRate (GHz) 1.531
pciBusID 0000:11:00.0
Total memory: 22.38GiB
Free memory: 22.21GiB
2018-05-19 20:10:23.275244: I tensorflow/core/common_runtime/gpu/gpu_device.cc:961] DMA: 0 1 2 3 
2018-05-19 20:10:23.275274: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971] 0:   Y Y Y Y 
2018-05-19 20:10:23.275285: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971] 1:   Y Y Y Y 
2018-05-19 20:10:23.275294: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971] 2:   Y Y Y Y 
2018-05-19 20:10:23.275303: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971] 3:   Y Y Y Y 
2018-05-19 20:10:23.275327: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1030] Creating TensorFlow device (/gpu:0) -> (device: 0, name: Tesla P40, pci bus id: 0000:08:00.0)
2018-05-19 20:10:23.275339: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1030] Creating TensorFlow device (/gpu:1) -> (device: 1, name: Tesla P40, pci bus id: 0000:0b:00.0)
2018-05-19 20:10:23.275366: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1030] Creating TensorFlow device (/gpu:2) -> (device: 2, name: Tesla P40, pci bus id: 0000:0e:00.0)
2018-05-19 20:10:23.275502: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1030] Creating TensorFlow device (/gpu:3) -> (device: 3, name: Tesla P40, pci bus id: 0000:11:00.0)
word2vecLstm_notes.py:178: UserWarning: Update your `LSTM` call to the Keras 2 API: `LSTM(activation="sigmoid", recurrent_activation="hard_sigmoid", units=50)`
  model.add(LSTM(output_dim=50, activation='sigmoid', inner_activation='hard_sigmoid'))
Compiling the Model...
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
Total params: 2,211,351
Trainable params: 2,211,351
Non-trainable params: 0
_________________________________________________________________
Train...
/data/liuhua/env3.5/lib/python3.5/site-packages/keras/models.py:851: UserWarning: The `nb_epoch` argument in `fit` has been renamed `epochs`.
  warnings.warn('The `nb_epoch` argument in `fit` '
Epoch 1/20
2018-05-19 20:10:28.335764: I tensorflow/core/common_runtime/gpu/pool_allocator.cc:247] PoolAllocator: After 5792 get requests, put_count=3496 evicted_count=1000 eviction_rate=0.286041 and unsatisfied allocation rate=0.586326
2018-05-19 20:10:28.335823: I tensorflow/core/common_runtime/gpu/pool_allocator.cc:259] Raising pool_size_limit_ from 100 to 110
   96/16858 [..............................] - ETA: 315s - loss: 0.6938 - acc: 0.56252018-05-19 20:10:28.982154: I tensorflow/core/common_runtime/gpu/pool_allocator.cc:247] PoolAllocator: After 1417 get requests, put_count=2398 evicted_count=1000 eviction_rate=0.417014 and unsatisfied allocation rate=0.0268172
2018-05-19 20:10:28.982222: I tensorflow/core/common_runtime/gpu/pool_allocator.cc:259] Raising pool_size_limit_ from 212 to 233
  192/16858 [..............................] - ETA: 211s - loss: 0.7076 - acc: 0.54172018-05-19 20:10:29.652687: I tensorflow/core/common_runtime/gpu/pool_allocator.cc:247] PoolAllocator: After 861 get requests, put_count=1906 evicted_count=1000 eviction_rate=0.524659 and unsatisfied allocation rate=0.00464576
2018-05-19 20:10:29.652732: I tensorflow/core/common_runtime/gpu/pool_allocator.cc:259] Raising pool_size_limit_ from 542 to 596
  320/16858 [..............................] - ETA: 168s - loss: 0.7109 - acc: 0.55312018-05-19 20:10:30.475043: I tensorflow/core/common_runtime/gpu/pool_allocator.cc:247] PoolAllocator: After 726 get requests, put_count=1829 evicted_count=1000 eviction_rate=0.546747 and unsatisfied allocation rate=0.00275482
2018-05-19 20:10:30.475077: I tensorflow/core/common_runtime/gpu/pool_allocator.cc:259] Raising pool_size_limit_ from 1158 to 1273
  544/16858 [..............................] - ETA: 140s - loss: 0.7093 - acc: 0.54412018-05-19 20:10:31.903954: I tensorflow/core/common_runtime/gpu/pool_allocator.cc:247] PoolAllocator: After 7489 get requests, put_count=7554 evicted_count=1000 eviction_rate=0.13238 and unsatisfied allocation rate=0.15209
2018-05-19 20:10:31.903996: I tensorflow/core/common_runtime/gpu/pool_allocator.cc:259] Raising pool_size_limit_ from 2253 to 2478
16858/16858 [==============================] - 109s - loss: 0.4731 - acc: 0.7738     
Epoch 2/20
16858/16858 [==============================] - 107s - loss: 0.2969 - acc: 0.8868     
Epoch 3/20
16858/16858 [==============================] - 107s - loss: 0.2283 - acc: 0.9198     
Epoch 4/20
16858/16858 [==============================] - 108s - loss: 0.1827 - acc: 0.9368     
Epoch 5/20
16858/16858 [==============================] - 107s - loss: 0.1564 - acc: 0.9459     
Epoch 6/20
16858/16858 [==============================] - 108s - loss: 0.1256 - acc: 0.9565     
Epoch 7/20
16858/16858 [==============================] - 107s - loss: 0.1092 - acc: 0.9647     
Epoch 8/20
16858/16858 [==============================] - 105s - loss: 0.0957 - acc: 0.9696     
Epoch 9/20
16858/16858 [==============================] - 104s - loss: 0.0908 - acc: 0.9697     
Epoch 10/20
16858/16858 [==============================] - 106s - loss: 0.0740 - acc: 0.9757     
Epoch 11/20
16858/16858 [==============================] - 107s - loss: 0.0629 - acc: 0.9813     
Epoch 12/20
16858/16858 [==============================] - 108s - loss: 0.0576 - acc: 0.9818     
Epoch 13/20
16858/16858 [==============================] - 107s - loss: 0.0559 - acc: 0.9832     
Epoch 14/20
16858/16858 [==============================] - 107s - loss: 0.0442 - acc: 0.9862     
Epoch 15/20
16858/16858 [==============================] - 106s - loss: 0.0529 - acc: 0.9840     
Epoch 16/20
16858/16858 [==============================] - 105s - loss: 0.0456 - acc: 0.9860     
Epoch 17/20
16858/16858 [==============================] - 107s - loss: 0.0512 - acc: 0.9850     
Epoch 18/20
16858/16858 [==============================] - 107s - loss: 0.0414 - acc: 0.9865     
Epoch 19/20
16858/16858 [==============================] - 107s - loss: 0.0349 - acc: 0.9898     
Epoch 20/20
16858/16858 [==============================] - 106s - loss: 0.0285 - acc: 0.9913     
Evaluate...
4215/4215 [==============================] - 7s      
Test score: [0.53292876407062173, 0.90011862418829758]
loading model......
loading weights......
1/1 [==============================] - 0s
屏幕较差，拍照也很粗糙。  positive
svm训练以及测试结果（测试语料：str = '屏幕较差，拍照也很粗糙。'）
Using TensorFlow backend.
loading data ...
Building prefix dict from /data/liuhua/env3.5/lib/python3.5/site-packages/jieba/dict.txt ...
Loading model from cache /tmp/jieba.cache
Loading model cost 1.189817190170288 seconds.
Prefix dict has been built succesfully.
0    [做, 父母, 刘墉, 心态, 学习, 补充, 新鲜血液, 一颗, 年轻, 心, 想, 孩子...
1    [作者, 真有, 英国人, 严谨, 风格, 提出, 观点, 论述, 论证, 物理学, 不深,...
2    [作者, 长篇大论, 借用, 详细, 报告, 数据处理, 工作, 计算结果, 支持, 其新,...
3    [作者, 战, ＂, 拥抱, ＂, 令人, 叫绝, 日本, 战败, 美军, 占领, 没胡, ...
4    [作者, 少年, 时即, 喜, 阅读, 精读, 无数, 经典, 庞大, 内心世界, 作品, ...
5    [作者, 一种, 专业, 谨慎, 若能, 有幸, 学习, 原版, 也许, 更好, 简体版, ...
6    [作者, 诗, 语言, 如水般, 清澈, 透明, 思想, 娓娓道来, 经验丰富, 智慧, 老...
7    [作者, 提出, 一种, 工作, 生活, 方式, 咨询, 界, 元老, 提出, 理念, 身体...
8    [作者, 妙语连珠, 年代, 层出不穷, 摇滚, 巨星, 故事, 紧紧, 相连, 乡愁, 摇...
9    [作者, 逻辑, 严密, 一气呵成, 一句, 废话, 深入浅出, 循循善诱, 环环相扣, 平...
Name: words, dtype: object
train word2vec model and get the input of svm model
(16884, 300)
(4221, 300)
train svm model...
.....
Warning: using -h 0 may be faster
*.*
optimization finished, #iter = 6131
obj = -6939.104617, rho = -3.228429
nSV = 7870, nBSV = 7540
Total nSV = 7870
[LibSVM]0.860696517413
use svm model to predict...
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





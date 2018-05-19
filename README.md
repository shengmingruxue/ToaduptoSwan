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





## Word2Vec
### 基本原理
参考链接：
- [中文word2vec数学原理详解及公式推导](https://www.cnblogs.com/peghoty/p/3857839.html)
- [中文word2vec原理介绍加总结版本数学原理推导](https://blog.csdn.net/kejizuiqianfang/article/details/99838249)
- [中文word2vec详解及公式推导](https://zhuanlan.zhihu.com/p/99616677)
- [中文word2vec深入浅出包含前世今生详解](https://zhuanlan.zhihu.com/p/114538417)
- [中文word2vec优化部分总结](https://zhuanlan.zhihu.com/p/88874759)
- [中文word2vec带例子的直白讲解](https://mp.weixin.qq.com/s/cpzBBntlFw6BDNUs6emCWw)
- [英文word2vec系列讲解](https://medium.com/nearist-ai/word2vec-tutorial-the-skip-gram-model-c7926e1fdc09)
- [word2vec详细复现](https://github.com/Link-Li/Word2Vec_python/tree/master)
- [word2vec论文解释](https://arxiv.org/abs/1411.2738)
- [word2vec斯坦福公开课课件](https://web.stanford.edu/class/cs224n/readings/cs224n-2019-notes01-wordvecs1.pdf)
### 优点
- Word2Vec会考虑上下文，词义的表达更好更准确。
- 使用Embedding方法维度更少，速度更快。
- 同时首次提出用稠密向量表示一个单词，为之后的ELMO，Transformer，BERT，GPT打下基础。
### 缺点
- 由于词和向量是一对一的关系，所以多义词的问题无法解决，或者说多义词在不同context的情况想表示意义不一样，但是是同一个index词，word2vec后很难表达全面。
- Word2Vec是一种静态的方式，虽然通用性强，但是无法针对特定任务做动态优化。因为静态词向量的弊端，是同一个词不管处于什么语境，都只能对应同一个固定的词向量。
- Word2Vec只考虑到上下文信息，而忽略的全局信息。
- Word2Vec只考虑了上下文的共现性，而忽略的了彼此之间的顺序性。
### 知识点提炼
- 参数解释
  - embedding_dim，embedding层的维度，同时也是最终一个词由多少维度的vector表示，也是影藏层神经元个数。
  - context_len，上下文的选择长度，比如选择w_t，context_len = 2，则有 w_t-1，w_t-2，w_t+1，w_t+2 作为上下文。
  - num_negative_samples，负采样的样本个数，用于加速优化的时候选择负样本的抽样个数。
  - min_count，当词出现的评论低于此参数的时候不记录词库及vocabulary。
  - init，初始化参数的方式，这里初始化方式会影响后续的参数converge速度，因为word2vec没有激活函数，也只有两层，梯度爆炸和梯度消失的情况很少见，不过对于不同激活函数初始化方式还是需要应该不一样，因为不同的激活函数梯度函数不一样。
  - optimizer，优化器的方式，每次反向传播后要更新此层的参数，不同的优化器采用不同的方式更新参数，具体参考优化器文档。
  - act_fn，需要注意的是word2vec并没有激活函数，影藏层和输出层都是存线性转换。
- 其它要点 
  - CBOW是一个词预测它上下文context_len范围的词，skip-gram是反过来，上下文context_len范围的词预测中间词。同样CBOW需要再影藏层后对每个预测词得出的h进行mean，而skip-gram则需要对输出层每个预测词的输出后的loss进行sum才是最终loss。
  - Word2Vec提出基于层次的softmax进行加速softmax的计算，因为softmax本身是指数级别的计算很费事，并且我们需要做的是V(词典大小)个class级别的分类问题。
  - Word2Vec同时提出另一种负采样的方式进行softmax的计算，此方式可以理解成，每次采样同一个比例的负样本最为softmax的分母，这样V(词典大小)个class级别的分类问题转换成K个class级别的分类问题，同时V >> K。
  - Skip-Gram的速度比CBOW慢一点，小数据集中对低频次的效果更好。因为Skip-Gram计算loss这一步更慢一些，小数据集中低频次的词和高频的词共享的context在Skip-Gram中影响不大，反过来CBOW因为高频词的context和低频词的context在小数据集上交集更多，则更容易受影响。大数据集的话交集不多无论高频低频，所以两种model的影响没那么明显。
  - Sub-Sampling Frequent Words可以同时提高算法的速度和精度，Sample 建议取值为 (10^-5, 10^-3)。
  - Hierarchical Softmax对低词频的更友好，因为Hierarchical Softmax利用了Huffman树依据词频建树，词频大的节点离根节点较近，词频低的节点离根节点较远，距离远参数数量就多，在训练的过程中，低频词的路径上的参数能够得到更多的训练，所以效果会更好。
  - Negative Sampling对高词频更友好。因为高频词的负样本对正确样本的影响比低频词的负样本对正确样本的影响要明显。显然低频词的负样本可能softmax后和正确样本的softmax后差异并不大。
  - context_len参数，一般来说Skip-Gram一般10左右，CBOW一般为5左右。
  - Word2Vec总共参数个数由两层MLP参数控制，参数数量 = 2 * (词向量的维度 * 词典长度)。
  - 现如今提取词向量对于复杂的语句情况，我们已经不再使用Word2Vec的得出词的embedding，更多的采用更加复杂的Transformer来提取词的embedding，而且一般embedding直接作为输入层的第一层及input之后，直接转化成统一维度后再进入encoder层提取信息。
  - **Hierarchical Softmax和Negative Sampling的细节有待更新。**
### Engineer Work
- 实际工作中可以调用gensim，或者Spacy包自带的model。或者用pytorch里面的embedding层自己搭建一个model。
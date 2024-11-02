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
  - Skip-gram是一个词预测它上下文context_len范围的词，CBOW是反过来，上下文context_len范围的词预测中间词。同样CBOW需要再影藏层后对每个预测词得出的h进行mean，而skip-gram则需要对输出层每个预测词的输出后的loss进行sum才是最终loss。
  - Word2Vec提出基于层次的softmax进行加速softmax的计算，因为softmax本身是指数级别的计算很费事，并且我们需要做的是V(词典大小)个class级别的分类问题。
  - Word2Vec同时提出另一种负采样的方式进行softmax的计算，此方式可以理解成，每次采样同一个比例的负样本作为softmax的分母，这样V(词典大小)个class级别的分类问题转换成K个class级别的分类问题，同时V >> K。
  - Skip-Gram的速度比CBOW慢一点，小数据集中对低频次的效果更好。因为Skip-Gram计算loss这一步更慢一些，小数据集中低频次的词和高频的词共享的context在Skip-Gram中影响不大，反过来CBOW因为高频词的context和低频词的context在小数据集上交集更多，则更容易受影响。大数据集的话交集不多无论高频低频，所以两种model的影响没那么明显。
  - Sub-Sampling Frequent Words可以同时提高算法的速度和精度，Sample 建议取值为 (10^-5, 10^-3)。
  - Hierarchical Softmax对低词频的更友好，因为Hierarchical Softmax利用了Huffman树依据词频建树，词频大的节点离根节点较近，词频低的节点离根节点较远，距离远参数数量就多，在训练的过程中，低频词的路径上的参数能够得到更多的训练，所以效果会更好。
  - Negative Sampling对高词频更友好。因为高频词的负样本对正确样本的影响比低频词的负样本对正确样本的影响要明显。显然低频词的负样本可能softmax后和正确样本的softmax后差异并不大。
  - context_len参数，一般来说Skip-Gram一般10左右，CBOW一般为5左右。
  - Word2Vec总共参数个数由两层MLP参数控制，参数数量 = 2 * (词向量的维度 * 词典长度)。
  - 词向量维度代表了词语的特征，特征越多能够更准确的将词与词区分。但是在实际应用中维度太多训练出来的模型会越大，虽然维度越多能够更好区分，但是词与词之间的关系也就会被淡化，这与我们训练词向量的目的是相反的，我们训练词向量是希望能够通过统计来找出词与词之间的联系，维度太高了会淡化词之间的关系，但是维度太低了又不能将词区分，所以词向量的维度选择依赖于实际应用场景。一般说来200-400维是比较常见的。
  - 现如今提取词向量对于复杂的语句情况，我们已经不再使用Word2Vec的得出词的embedding，更多的采用更加复杂的Transformer来提取词的embedding，而且一般embedding直接作为输入层的第一层及input之后，直接转化成统一维度后再进入encoder层提取信息。不过对于一些静态短词组特征的embedding，比如tag，我们一般还是用Word2Vec更快速的转化成向量数组。
  - <mark>TODO: Hierarchical Softmax和Negative Sampling的细节有待更新。</mark>
### Engineer Work
- 实际工作中可以调用gensim，或者Spacy包自带的model。或者用pytorch里面的embedding层自己搭建一个model。

## Transformer
### 基本原理
参考链接：
- [英文图文系列解释](https://jalammar.github.io/illustrated-transformer/)
- [英文原论文详细注释](https://nlp.seas.harvard.edu/annotated-transformer/)
- [中文Transformer系列讲解](https://zhuanlan.zhihu.com/p/178610196)
- [中文Transformer初始化参数化标准化详细讲解](https://zhuanlan.zhihu.com/p/400925524)
- [中文Transformer位置编码总结](https://www.zhihu.com/question/347678607/answer/2301693596)
- [中文Transformer李沫大神的讲解注释](https://mp.weixin.qq.com/s/cpzBBntlFw6BDNUs6emCWw)
- [中文关于BN和LN的理解](https://blog.csdn.net/u010159842/article/details/109326409)
- [中文Transformer带例子的从零推导](https://zhuanlan.zhihu.com/p/648127076)
- [中文Transformer各种细节总结1](https://zhuanlan.zhihu.com/p/153183322)
- [中文Transformer各种细节总结2](https://zhuanlan.zhihu.com/p/132554155)
- [中文Transformer各种细节总结3](https://zhuanlan.zhihu.com/p/559495068)
- [中文Transformer各种细节总结4](https://zhuanlan.zhihu.com/p/165510026)
### 优点
- 对比LSTM，Transformer有完全的并行计算，Transformer的attention和feed-forward，均可以并行计算。而LSTM则依赖上一时刻，必须串行。
- 减少长距离依赖，利用self-attention将每个字之间距离缩短为1，大大缓解了长距离依赖问题，同时避免了梯度消失和爆炸的问题当文本太长的时候，也同样避免了后面的词对前面的词信息丢失的问题当句子很长的时候。
- 提高网络深度。由于大大缓解了长程依赖梯度衰减问题，Transformer网络可以很深，基于Transformer的BERT甚至可以做到24层。而LSTM一般只有2层或者4层。网络越深，高阶特征捕获能力越好，模型performance也可以越高。
- 真正的双向网络。Transformer可以同时融合前后位置的信息，而双向LSTM只是简单的将两个方向的结果相加，严格来说仍然是单向的。
- 可解释性强。完全基于attention的Transformer，可以表达字与字之间的相关关系，可以可视化解释每一层每一个词对其它词的权重，可解释性更强。
### 缺点
- 当文本长度很长时，比如篇章级别，计算量爆炸。self-attention的计算量为O(n^2), n为文本长度。Transformer-xl利用层级方式，将计算速度提升了1800倍。
- Transformer位置信息只靠position encoding，效果比较一般。当语句较短时，比如小于10个字，Transformer效果不一定比LSTM好。
- Transformer参数量较大，在大规模数据集上，效果远好于LSTM。但在小规模数据集上，如果不是利用pretrain models，效果不一定有LSTM好。
- Transformer固定了句子长度为512，短于 512：填充句子方式 padding，长于 512：截断句子或者将句子划分为多个 seg，分段训练
### 知识点提炼
- 参数解释
  - src_vocab，source 字典的长度，一般包含特殊字符的长度。
  - tgt_vocab，target 字典的长度，一般包含特殊字符开始翻译和结束翻译。
  - num_layers，encode 和 decode的层数，一般编码层和解码层层数一样，default = 6。
  - d_model，隐藏层的输出维度，每一个encode和decode的输入输出都是一样的维度，default = 512。
  - d_ff，FFN全连层的内部隐藏维度，default = 2048。
  - heads，多头注意力机制中的头的个数，default = 8。
  - dropout，dropout rate，全模型每个地方的drop out rate都一样，default = 0.1。
- 其它要点 
  - 对于位置编码。transformer利用三角函数编码方式实现，无需训练，并且不受长度限制，虽然原始论文中设置上线是5000。而bert则采用训练embedding_lookup方式，当语料库够大的时候能准确度比较好，不过最长长度为512，预测阶段受限制。 
  - attention的本质是一个向量的加权求和，query 点成 key 可以看做是进行每个词对其他词的权重计算。之后点成 value 可以看做每个词对其他词的加权系数乘以每个维度v向量，然后加起来，表示每个词在每个维度的加权系数和。
  - mask这里和输入矩阵shape相同，mask矩阵中值为0位置对应的输入矩阵的值更改为-1e9，一个非常非常小的数，经过softmax后趋近于0。mask在decode阶段的src-attention和encode阶段对于padding的部分都会应用mask。
  - query，key，value在self-attention的时候来源一样都是embedding之后的输出，但是在source-attention中query是target预测词的embedding输出，而key，value是source及encode后的输出，体现了encoder对decoder的加权贡献，算出来的是预测阶段的词对于source encode阶段词的加权系数和，attention相当于在 encoder 和 decoder 之间传递信息。
  - 整个transformer model只有在全连层FFN阶段有激活函数，并且是固定的relu，这个可以改成gelu在package里面，还有self-attention层里面有softmax进行非线性转换，其它地方都是线性转换。
  - 在每层的self-attention和feed-forward模块中，均应用了残差连接。残差连接先对输入进行layerNorm归一化，然后送入attention或feed-forward模块，然后经过dropout，最后再和原始输入相加。这样做的好处是，让每一层attention和feed-forward模块的输入值，均是经过归一化的，保持在一个量级上，从而可以加快收敛速度。
  - decode层和encode不一样的是，decode先是self-attention，然后是source-attention，再是FFN。decode的self-attention有mask，从而避免未来信息的穿越问题，比如预测第一个词的时候不会看到后面的词，主要原因是训练的时候不能看到后面的词，如果看的到，当预测的时候并没有ground truth的时候model并不知道去预测什么词，其它每个词同理，mask为一个上三角矩阵，上三角全为1，下三角和对角线全为0。上三角为需要mask的地方。
  - 输出层注意，每次预测一个词，所以我们拿出来的是当前预测词的输出维度，原始论文中是 [1, 512] 这个维度，并不需要前面的词的输出向量，虽然我们每次输出都包含之前所有词的输出向量。再去做softmax。
  - 在Transformer之前，点积注意力在高维度下表现不太好，可能跟query和key的尺度缩放问题有关。由于query和key都是独立学习的，query和key向量中的每个系数都有大致固定的尺度（这取决于计算query和key的全连接层参数，因此是不太可控的；作为向量，query和key都没有特意做归一化），因此每个对应系数的积也有固定的尺度（也就是固定的方差），也就是说，维度越高，点积 q⋅k 值的尺度就会越大（点积方差正比于维度数量）。表示复杂的语义概念需要很高的query和key维度，造成很大的点积绝对值，会在softmax中造成问题，因为其中用到指数运算，绝对值很大的点积在训练中会收到几乎为0的梯度，因为最大值softmax会更加靠近于1，剩下那些值就会更加靠近于0。值就会更加向两端靠拢，算梯度的时候，梯度比较小。softmax会让大的数据更大，小的更小。导致训练进展缓慢。对此，Transformer的解决方法是在点积 q⋅k 上除以 sqrt(d_k)，这就正好抵消了维度增加造成的点积尺度放大效应，保证了不论维度多高，点积的方差都是恒定的。
  - 本质上说，参数的初始化，点积后进行缩放除以sqrt(d_k)及参数化，或者输出进行标准化及各种BN/LN，都属于对输入输出的分布转化成相同的均值和方差，从而达到避免梯度的消失或者爆炸，加速model收敛和稳定性。
  - 位置编码的作用主要是因为self-attention是无向的，并且每个词的距离都变成了1，然而句子本身是有先后顺序的，所以我们要想办法把句子先后顺序的信息喂给model，也就是在embedding的时候加入了位置编码。
  - LN比BN更适用于时间序列的input，因为LN是对每个样本所有维度做标准化，而BN是对所有mini-batch里面的样本的同一个维度做标准化，时间序列的input长度可能都不一样，对不同样本做的话，如果样本长度变化比较大的时候，每次计算小批量的均值和方差，均值和方差的抖动大。全局的均值和方差：测试时遇到一个特别长的全新样本，训练时未见过，训练时计算的均值和方差可能不好用。LN每个样本自己算均值和方差，不需要存全局的均值和方差。更稳定，不管样本长还是短，均值和方差是在每个样本内计算。还有一个原因是LN是对同一个样本做，这样并没有破坏一个句子的内部语义联系，然后BN是跨样本做，没有任何意义对不同的单词同一个维度做标准化，反而会破坏句子的内部联系。
  - multi-head attention 给 h 次机会去学习不一样的投影的方法，借鉴了CNN里面的多核的机制，在低纬度多个独立的特征空间，使得在投影进去的度量空间里面能够去匹配不同模式需要的一些相似函数，然后把 h 个 heads 拼接起来，最后再做一次投影。这样希望可以学习到更多信息。
  - attention相当于对全局信息先做了一个aggregation，然后 MLP 做语义的转换，映射成我更想要的那个语义空间。对比RNN，RNN 是把上一个时刻的信息输出传入下一个时候做输入。Transformer 通过一个 attention 层，去全局的拿到整个序列里面信息，再用 MLP 做语义的转换。其实和GNN里面的message-passing一样，都是先做信息aggregation然后再MLP投影到想要的空间。
  - 使用Q/K/V不相同的权重矩阵生成，可以保证在不同空间进行投影，增强了表达能力，提高了泛化能力。线性变换的好处：在 Q * K^T 部分，线性变换矩阵将KQ投影到了不同的空间，增加了表达能力（这一原理可以同理SVM中的核函数-将向量映射到高维空间以解决非线性问题），这样计算得到的注意力矩阵的泛化能力更高。
  - self-attention的部分是计算量最大的部分，ex: 1 batch，12 heads，length 256，d_k 64，则 Q * K^T -> score 有 [256, 64] * [64, 256] -> 256 * 64 * 256 次计算，score * V 有 [256, 256] * [256, 64] -> 256 * 256 * 64 次计算，所以总共有 12 * （256 * 64 * 256 + 256 * 256 * 64）= 1亿次乘法。总的来说对于每个头时间复杂度是 1. 算score: length^2 * d_k + 2. 算softmax: length^2 + 3. 算score * v: length^2 * d_k。不过每个头是并行运算。
### Engineer Work
- 实际工作中可以调用pytorch里面的nn.Transformer来直接实现src，tgt的训练。或者使用hugging face的API。


## 面试问题总结
### Word2Vec 系列
1. Word2Vec有哪两种模型，各自的计算过程中损失函数是什么，优缺点是什么？
   - CBOW，context词预测中心词的思路。以下图展示计算过程。![CBOW](/pics/w2v_CBOW.png)
   - Skip-gram，与CBOW相反，中心词预测context词。以下图展示计算过程。![skip-gram](/pics/w2v_skip_gram.png)
   - 总结来说skip-gram更好的表达小数据集低频词，CBOW能更好的表达高频词。
2. 简单介绍一下Word2Vec的两种加速方式。
   - 层次softmax。简单点说就是把原本需要输出v个类别的softmax，转化成输出log(v)个类别的softmax，v为词典的长度。
   - 负采样。原本需要输出v个类别的softmax，转化成输出k + 1个类别的softmax，k为负样本的个数。
   - <mark>TODO: 具体数学推导细节有待更新。</mark>
3. Word2Vec的缺点。
    - 不能解决一词多义的embedding信息。
    - 没有考虑全局信息。
    - 没有考虑词在句子里面出现的顺序。
    - 没有正则化处理参数防止过拟合。
    - 对于新的词没有出现过的词效果不好。准确说是不能处理没出现的词，找不到词的index在embedding里面。
4. Word2Vec输入输出是什么？隐藏层的激活函数是什么？输出层的激活函数是什么？
    - 输入：每个词的one-hot embedding，大小为 [N, V]，N -> batch size，V -> dictionary size。
    - 输出：输出层softmax后的结果，大小同样为 [N, V]，其中每个样本对应softmax最大值的index就是我们预测的那个词的index。
    - 隐藏层没有激活函数，或者说只有线性激活函数，因为word2vec不是为了做语言模型，它不需要预测得更准，线性激活会让模型收敛更快并且更简单。
    - 输出层的激活函数是softmax，为了爆炸输出是符合概率分布的结果。
5. Word2Vec如何获取词向量？
    - 影藏层的参数W就是最终我们要的embedding向量。大小为 [N, E]，N -> batch size，E -> embedding size。
6. 推导一下Word2Vec参数如何更新？
    - <mark>TODO: 具体数学推导细节有待更新。</mark>
7. Word2Vec和隐狄利克雷模型(LDA)有什么区别与联系？
    - <mark>TODO: 待补充。</mark>
8. Word2vec和Tf-idf在相似度计算时的区别？
    - Word2vec是稠密的向量，而 tf-idf 则是稀疏的向量。
    - Word2vec的向量可以表达语义信息，但是 tf-idf 的向量不可以。

### Transformer 系列
1. Transformer为何使用多头注意力机制？（为什么不使用一个头）
   - 简单回答就是，多头保证了transformer可以注意到不同子空间的信息，捕捉到更加丰富的特征信息。
   - 同时多头会解决模型在对当前位置的信息进行编码时，会过度的将注意力集中于自身位置的问题。
   - 参考[链接](https://www.zhihu.com/question/341222779)
2. Transformer为什么Q和K使用不同的权重矩阵生成，为何不能使用同一个值进行自身的点乘？
    - 使用Q/K/V不相同可以保证在不同空间进行投影，增强了表达能力，提高了泛化能力。
    - K和Q的点乘是为了得到一个attention score 矩阵，原本V里的各个单词只用word embedding表示，相互之间没什么关系。但是经过与attention score相乘后，V中每个token的向量（即一个单词的word embedding向量），在300维的每个维度上（每一列）上，都会对其他token做出调整（关注度不同）。与V相乘这一步，相当于提纯，让每个单词关注该关注的部分。K和Q使用了不同的W_k, W_Q来计算，可以理解为是在不同空间上的投影。正因为有了这种不同空间的投影，增加了表达能力，这样计算得到的attention score矩阵的泛化能力更高。因为K和Q使用了不同的W_k, W_Q来计算，得到的也是两个完全不同的矩阵，所以表达能力更强。但是如果不用Q，直接拿K和K点乘的话，attention score 矩阵是一个对称矩阵。因为是同样一个矩阵，都投影到了同样一个空间，所以泛化能力很差。这样的矩阵导致对V进行提纯的时候，效果也不会好。
    - 参考[链接](https://www.zhihu.com/question/319339652)
3. 为什么在进行softmax之前需要对attention进行scaled（为什么除以dk的平方根），并使用公式推导进行讲解。
    - 总结来说就是稳定参数的分布，归一化效果，保证softmax的时候不会容易遇到极值导致梯度消失。
    - 参考[链接](https://www.zhihu.com/question/339723385)
4. 在计算attention score的时候如何对padding做mask操作？
    - padding位置置为负无穷(一般来说-1000就可以)，和self target attention里面的方法几乎一样。
5. Transformer对比RNN/LSTM有什么优势？
    - RNN并行计算能力差，RNN是sequence by sequence的，后一个一定要等前一个计算完。
    - Transformer特征抽取能力比RNN强。
6. Transformer是如何训练的？测试阶段如何进行测试呢？
    - 训练和测试唯一不同的地方是decoder，训练的时候通过mask操作来完成self target attention，但是在inference的时候，没有ground truth，所以需要拿y^_t-1作为当期的输入。
7. Transformer内是否有相互share的weight矩阵?
    - 输入层的embedding和最后一层线性层的权重是相互share的，linear层 -> [n_embd, vocab_size]，embedding层 -> [vocab_size, n_embd]。
    - 第一，本质上这两层干的事情是一样的，embedding是把词变成dense向量，linear层是把最终的输出变回一样的dense向量，共享同样的权重矩阵可以更好的捕捉词意。
    - 第二，这一层大概占据最小号的模型30%的参数，也就是说共享权重矩阵可以省去30%的参数，这样可以减少参数量加上训练。

### GPT 系列
1. GPT在inference阶段生成新的下一个单词的时候是怎么生成的，为什么？
   - GPT在inference时候生成下一个单词的时候，用的是前一个词的结果并且concat之前所有词一起作为输入房间model，比如 input: [1, 1] 代表batch = 1, length = 1。第一个输入词。
   - model输出为下一个词在整个字典中的每个词的预测logits，需要自行softmax后，再从multinomial distribution里面根据之前计算出来的每个词的概率权重，抽取一个词作为下一个预测的新词。
   - 本质上是控制文本生成的多样性和随机性，这里有两种普遍的做法，一种是温度缩放，一种是Top-k采样。
     - 先区别一下概率采样和argmax，传统意义上我们使用argmax选择概率最大的作为下一个词，但是为了实现概率采样我们可以使用multinomial distribution来替代argmax，这样下一个词会根据概率分布比例进行生成下一个词。此时虽然大部分情况和argmax的输出是一样的，但是也有概率选择概率值不是最大的那个token作为生成的词。这样就实现了生成文本的多样性。当然也有概率生成一些毫无逻辑的词。
     - 温度缩放，本质上是对概率缩放的程度控制。温度缩放的公式，scaled_logits = logits / temperate。其实就是对输出进行等比例变化。
       - 小于1的温度，会使分布更加shape，使得最可能也就是概率最大的那个数的值变的概率更大，这样multinomial distribution进行采样的时候更加可能的选择最大值对应的词。
       - 等于1的温度，没有任何缩放，等价于不缩放，直接使用原始计算出来的概率分布进行multinomial distribution采样。
       - 大于1的温度，会使分布更加均匀，这样其它词被选中的概率增加，生成的文本添加更多变化。当然也会有更大概率输出一些毫无意义的词。
     - Top-k采样，我们可以只关注logits值最大的k个词，比如k=4，此时我们把其它词的logits进行mask掉，和训练过程中的mask技巧一样，然后再进行softmax，得到其它token的概率为0，四个最大值的词对应的概率值。之后再进行温度缩放方式的multinomial distribution采样，以在这些非零数值中选择下一个预测词。Top-k只关注选中的范围内的词，从而避免一些毫无意义的词的预测输出。


### Tokenization 系列
1. 为什么不能直接用Unicode来实现tokenization?
   - 如果直接用Unicode来实现的话，那么这个字典会很大，基本上在15k左右，这样做后续的softmax很慢，并且对于长句子或者篇章级别的输入，embedding后会变得非常长，导致模型效果不好。
   - Unicode会一直在变动，不稳定，这样每次token后的结果会不一样。
2. 为什么在regex token的时候需要一些pattern来定义如何split？
   - 保证一些连续有意义的词被split开，比如单词和数字，ex: "word123" -> "word", "_123"，这样后续的BPE token的merge不会出现奇怪的组合。
   - 保证一下没有意义的划分不会出现，比如单词和空格，ex: "hello word" -> "hello", "_word"，这样后续的BPE token不会出现 "o_"这种奇怪的token。
   - 保证一些标点符号的划分，比如一句话最后一个单词和标点符号，ex: "you!!!?" -> "you", "!!!?"。
   - 更多的划分参考官方[link](https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py)
3. 为什么需要special token？
   - 在GPT2里面我们的vocabulary大小有50257，256来raw bytes token，50000来着BPE token里面的merged token，另外一个是special token '<|endoftext|>'
   - special token存在的意义是比如让model知道可以停止生成文章了。或者还有其它special token表示可以开始生成文章了。
4. 为什么需要BPE token，并简单描述一下BPE token的过程？
   - 简单来说是因为，如果不用压缩merge字符的形式处理的话，如果只用character level的形式进行token的话，每个句子tokenizer后太长了，并且token之间毫无意义，对于模型学习收敛和performance并不友好。
   - 直觉上来说，常出现的配对的字符应该被合并在一起传给model去学习。这样可以提供更多的语义相比较字符level的tokenizer。
   - BPE的过程可以简单概述为：
     - 先把输入的句子的每个单词转化成utf-8 char，然后转化成list of int。
     - 按照相邻两两配对的形式进行计算出现的词频
     - 把最高出现词频的两个词进行merge，并用一个新的token index代替句子内所有的出现的这两个词。
     - 记录下来merge的词和记录配对结果进vocabulary，重复上面 2 - 4步骤，直到达到vocabulary的上线
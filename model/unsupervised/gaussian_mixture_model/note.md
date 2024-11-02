
### 基本原理
参考链接：
- [中文原理分析和公式推导](https://blog.csdn.net/omade/article/details/27194451)
- [中文原理讲解](https://blog.csdn.net/lin_limin/article/details/81048411)
- [英文链接图文解释](https://towardsdatascience.com/gaussian-mixture-model-clearly-explained-115010f7d4cf)
- [sklearn中相关应用文档](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html)
### 优点
- 速度还可以对于10K，10维数据，每一轮EM需要从新计算所有点的likelihood比较耗时。
- GMM模型参数很多，这取决与K个cluster的选择，模型的参数拟合能力可以达到不错的水平。
- 相对KMeans可用于更复杂的数据分布处理，可以用多个高斯分布近似描述数据分布。
- 相对KMeans考虑数据的variance当拟合模型的时候。换句话来说KMeans就是只考虑mean情况下的GMM，并且KMeans也是EM算法的逻辑。
- 对比其它clustering模型，因为是概率生成式model，可以用于生成全新的数据，因为可以通过对多元高斯分布采样来生成新数据。
### 缺点
- 需要手动设置参数n_cluster，和初始化mean covariance，初始化的covariance值会对结果产生比较大的影响。
- GMM模型对于noise和异常点比较敏感，因为是density probability base的model。
- 对于不一定是高斯分布的多维数据，没有选好n_cluster后的拟合结果不一定更好。
### 知识点提炼
- 基于概率的clustering模型，需要assumption数据符合多维高斯分布，但这个假设不是强假设，因为可以通过选择cluster个数来多个分布近似模拟。
- 使用EM算法来估计GMM参数，E步骤计算影变量P(Ck|Xi)，M步骤更新likelihood的参数mean，cov和权重priors。
- EM算法最终得到的局部最优点，但是GMM使用soft label，模型允许一个点属于多个高斯分布，所以即使是局部最优最小值也处于比较优的状态下。
- 本质上GMM不是clustering模型，更像是找到多个多元高斯分布来近似描述全局数据的分布。此处的n_cluster不能很好的描述聚类中的分类，和数据类别。同理实际中如果有更多的背景知识对于数据本身，做clustering时候并不推荐GMM。
- 对比KMeans，如果选择GMM的n_cluster够精准GMM能更好的handle复杂的非线性patterns数据分布。
- 对比KMeans，KMeans更快更准确当数据大于10K并且分布比较separate的时候，GMM会更准确当数据小或者数据重叠比较多的情况下。
### Engineer Work
- 一般来说直接调用sklearn中的GMM包来实现。实际生产中目前来说在(10k, 10)的数据集上需要5分钟左右在8cores/16Memory上。
- 实际工作中数据量巨大超过1B，同样一般不推荐在pipeline里面设计GMM算法的component。因为重复迭代计算每个点的likelihood非常慢。可以通过层层clustering后再用GMM在中等样本上。

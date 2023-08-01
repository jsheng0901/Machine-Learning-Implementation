
### 基本原理
参考链接：
- [中文链接解释](https://bindog.github.io/blog/2016/06/04/from-sne-to-tsne-to-largevis/)
- [英文链接图文解释](https://towardsdatascience.com/t-sne-clearly-explained-d84c537f53a)
- [英文链接解释及参考复现代码](https://nlml.github.io/in-raw-numpy/in-raw-numpy-t-sne/)
- [sklearn中相关应用文档](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)
### 优点
- 对比PCA之类的线性降维，可以很好的handle特征直接有correlation的非线性数据降维，特别是manifold类型数据。
- 对比PCA，可以更好的capture到local和global信息从原始高位空间到降维后。
### 缺点
- 时间和空间复杂度都是O($n^2$)因为有大量的pairwise distance需要计算，而且当维度特别大的时候，每一个点的距离计算都会变的特别慢O(D)D是维度。总体计算特别耗时。
- TSNE倾向于保存局部特征，对于维数(intrinsic dimensionality)本身就很高的数据集，是不可能完整的映射到2-3维的空间。
- 对于全局结构的降维保留受到困惑度参数影响，并不能很好地找到全局结构特别当数据大于10K的时候。但这个问题可以通过PCA初始化点（使用init ='pca'）来缓解，或者手动自己PCA降维到50以下先。
- TSNE没有唯一最优解，因为是SGD，且没有predict部分。如果想要做predict，可以考虑降维之后，再构建一个回归方程之类的模型去做。但是要注意，TSNE中距离本身是没有意义，都是概率分布问题。
- 对比PCA需要调参，参数选择对结果影响很大。
### 知识点提炼
- 高纬空间中计算相对概率，引入softmax原理归一化处理概率Pj|i，可以保持距离的相对性。
- 引入对称SNE的概念，用Pji替换Pj|i和Pi|j，联合概率代表条件概率，这样可以有效降低数据中一些异常点对KL惩罚度上的影响。
- 高纬空间映射到低纬空间是存在一个拥挤问题，降维后不同类别的簇挤在一起，无法区分开来，失去了全局信息。这里引入t分布在低纬空间，t分布具有长尾巴，这样对于高维空间中相距较近的点，为了满足Pij=Qij低维空间中的距离需要稍小一点，而对于高维空间中相距较远的点，为了满足Pij=Qij，低维空间中的距离需要更远。这恰好满足了我们的需求，即同一簇内的点(距离较近)聚合的更紧密，不同簇之间的点(距离较远)更加疏远。
- 困惑度参数类似降维时考虑附近有效近点的个数，越大考虑越考虑全局数据，同时说明Pj|i的分布中大部分点越相似，因为越大的困惑度需要entropy越高，说明数据分布越flat。当然越大计算也越慢。
- 困惑度越大降维后越拥挤，无法将点有效区分开，越小越离散，不能保留高维数据的局部结构。一般我们选取5-50之间。
### Engineer Work
- 一般来说直接调用sklearn中的TSNE包来实现。实际生产中当维度达到50以上数据量大于10K的时候就不会那么高效了。解决方法可以混合PCA和TSNE，先PCA降维到50以下在TSNE。
- 实际工作中数据量巨大，一般不推荐在pipeline里面设计TSNE算法的component。多用于做可视化分析，协同理解数据分布的情况，来方便设计选择后续的clustering算法。
- 如一定需要降维，实际应用中推荐一个更快更能代表全局结构的算法UMAP。
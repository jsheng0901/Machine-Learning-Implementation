
### 基本原理
参考链接：
- [前置知识covariance matrix中文链接解释](https://zhuanlan.zhihu.com/p/464822791)
- [前置知识covariance matrix中文链接解释](https://blog.csdn.net/xiaojinger_123/article/details/130749074)
- [中文基本算法理解和应用](https://blog.csdn.net/weixin_45142381/article/details/127150708)
- [中文原理讲解](https://zhuanlan.zhihu.com/p/37777074/)
- [中文原理分析推导](https://zhuanlan.zhihu.com/p/260186662)
- [英文链接图文解释](https://medium.com/@dareyadewumi650/understanding-the-role-of-eigenvectors-and-eigenvalues-in-pca-dimensionality-reduction-10186dad0c5c)
- [sklearn中相关应用文档](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
- [方差与信息量直接的关系](https://www.zhihu.com/question/36481348)
### 优点
- 对于高纬数据，降维可以去noise，维度太高的数据很多feature可能并没有什么作用，降维后再跑supervise model可以有效的预防overfitting。
- 对于高纬数据，主成分之间相互正交，没有线性相关，降维可以去除multicollinearity，为后续feature importance的选择铺垫。
- 加速后续supervise model训练速度。
- 对比TSNE等其它降维方式，速度快很多，计算方式简单，几乎都是matrix计算。
### 缺点
- 由于改变原始数据的分布和坐标轴，主成分的解释性很模糊。
- 信息丢失，对应贡献量小的主成分维度不一定是没用信息，对于后续supervise model也许刚好是区别不同样本类别的重要信息。
- 对于不一定是高斯分布的数据，降维后的结果不一定更好。
- PCA是线性降维，对于数据本身就包含非线性correlation的feature降维后效果并不理想。
### 知识点提炼
- 对数据一定要做中心化预处理，不然不同维度的scale的feature数据会影响特征向量和特征值的计算，主要是covariance matrix的影响。
- 对于如何选取components，及最后的降维维度，可以取特征值的平方累加占总的特征值平方和的ratio来决定，一般需要取覆盖90%以上的维度。
- 如何选取降维方式取决于后续对于数据的需求，一般来说如果需要后续做clustering，我们则需要保留更多的local和global数据distribution信息，及选取TSNE或者UMAP之类降维model。如果后续是supervise model，则选择PCA之类的model更好的preprocessing数据，去除noise.
- 虽说PCA降维后可以一定prevent overfitting，但是最好还是使用regularization来prevent overfitting。个人理解，因为regularization是在model fitting阶段对于参数空间的限制和计算，然而PCA是对数据本身一开始就改变distribution，这种改变可能造成信息丢失。
- 一般来说当原始数据跑完model后performance还不错的情况下但是速度很慢，则推荐试一试先PCA降维数据，加速后续model training。
- 对于PCA只是线性降维，我们可以做kernel PCA，具体方法就是先把原始数据X映射到高纬度，再从新的高纬度降低到最后低纬度。应用核函数在X上，然后计算特征向量。但整体计算量巨大。
### Engineer Work
- 一般来说直接调用sklearn中的PCA包来实现。实际生产中目前来说500+维度1M sample都还是很高效的。
- 实际工作中数据量巨大，一般不推荐在pipeline里面设计TSNE算法的component。多用于做可视化分析，协同理解数据分布的情况，来方便设计选择后续的clustering算法。

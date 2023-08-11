
### 基本原理
参考链接：
- [中文原理讲解](https://zhuanlan.zhihu.com/p/75477709)
- [英文链接图文解释](https://towardsdatascience.com/understanding-k-means-clustering-in-machine-learning-6a6e67336aa1)
- [sklearn中相关应用文档](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
### 优点
- 优化加速过的KMeans速度很快，几乎是最快的clustering算法。
- 非常容易理解和解释结果和算法过程。
- 可用于大数据如1M以上的数据进行clustering。
- 保证可以coverage最终结果，不过是local optimal。
### 缺点
- 需要手动设置参数k，需要一些对于数据的背景知识或者用手肘法来调参，但是不一定是最优的。
- 初始化centroids点对于最终的结果的影响很大，当k不是很大的时候可以选择不同的初始值然后多次试验调整，但是当k很大的时候需要参考KMeans-seeding。
- 对于数据分布有很多non-convex的pattern并且数据很多的时候KMeans并不能很好的聚类。
- 对于异常点和noise点很敏感，会影响最终的centroids的结果或者影响整体聚类效果，建议remove异常点再做KMeans。
- 任何包含计算距离的算法当维度很大的时候都会有Curse of Dimensionality问题，建议先降维再clustering。
- 对比GMM，KMeans每个数据点只属于唯一的一个聚类，而GMM基于后验概率进行了软分配，KMeans可以看成GMM的特殊情形。(没有估计聚类的协方差，只估计了聚类的均值)
### 知识点提炼
- 对于数据需要先中心化一下，确保不同维度的数据在同一个scale下。
- 如何选取K这里需要引入手肘法或者prior知识来确定。
- 如何需要使用KMeans建议先remove异常点再聚类。
- KMeans如果需要对categorical数据进行聚类，建议使用KModes。
- KMeans如果需要对categorical和numerical数据同时进行聚类，建议使用KPrototype。
- 相比KMeans，GMM收敛之前，经历了更多次的迭代，每次迭代需要计算更多的计算量，通常运行KMeans找到GMM的一个合适的初始值，接下来使用GMM进行调节。可以得到更好的聚类拟合。
- 普通版本的KMeans时间复杂度是O(i * k * n * d)，空间复杂度是O(kd -> centroids + nd -> 数据集)，i是重复的次数，k是cluster数量，n是多少个点，d是每个点的维度。当维度很大并且数据量很多的时候，会变得非常耗时。
### Engineer Work
- 一般来说直接调用sklearn中的KMeans包来实现。实际生产中普通版本的KMeans目前来说在(100k, 10)的数据集上需要15分钟左右在8cores/16Memory上。可以使用GPU版本来加速，但本人没测试过目前。
- 实际工作中，涉及高纬的数据，一般都需要降维，一来是可以克服维度灾难诅咒对于距离计算，二来可以加速，因为distance的计算时间复杂度是O(d)级别。
- 实际工作中数据量巨大超过1B，同样一般不推荐在pipeline里面直接上KMeans。一般来说(1M, 10d)以上的数据都不推荐直接run聚类model。

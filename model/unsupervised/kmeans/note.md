
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
- 如何初始化中心点，方法包括K-Means++，一般是默认的方式。或者自己多尝试几次多设置几个不同的初始点，从中选最优，也就是具有最小SSE值的那组作为最终聚类。
- 如何选取K这里需要引入手肘法或者prior知识来确定。
- 如何需要使用KMeans建议先remove异常点再聚类。
- KMeans如果需要对categorical数据进行聚类，建议使用KModes。
- KMeans如果需要对categorical和numerical数据同时进行聚类，建议使用KPrototype。
- 相比KMeans，GMM收敛之前，经历了更多次的迭代，每次迭代需要计算更多的计算量，通常运行KMeans找到GMM的一个合适的初始值，接下来使用GMM进行调节。可以得到更好的聚类拟合。
- 一般我们可以把KMeans先作为大范围的cluster模型，对数据进行第一层次的分类，因为KMeans比较快，并且大数据量下基本存在convex的pattern，之后再对每个小的cluster进行更复杂更适用于不同形状cluster的模型，比如HDBSCAN。
- 普通版本的KMeans时间复杂度是O(i * k * n * d)，空间复杂度是O(kd -> centroids + nd -> 数据集)，i是重复的次数，k是cluster数量，n是多少个点，d是每个点的维度。当维度很大并且数据量很多的时候，会变得非常耗时。
### Engineer Work
- 一般来说直接调用sklearn中的KMeans包来实现。实际生产中普通版本的KMeans目前来说在(100k, 10)的数据集上需要15分钟左右在8cores/16Memory上。可以使用GPU版本来加速，但本人没测试过目前。
- 实际工作中，涉及高纬的数据，一般都需要降维，一来是可以克服维度灾难诅咒对于距离计算，二来可以加速，因为distance的计算时间复杂度是O(d)级别。
- 实际工作中数据量巨大超过1B，同样一般不推荐在pipeline里面直接上KMeans。一般来说(1M, 10d)以上的数据都不推荐直接run聚类model。


### 面试问题总结
1. kmeans的复杂度？
   - 时间上：O(i * k * n * d)，i为迭代次数，k为簇的数目，n为样本个数，d为维数空间复杂度。
   - 空间上：O(k * d + n * d)，每个点要计算自己到中心点的距离，每个cluster要计算中心点是多少。
2. 影响kmeans的主要因素有哪些?
   - 样本输入顺序，主要可能会影响初始化中心点的生成。
   - 模式相似性测度，主要影响样本被划分去哪个cluster的方式。
   - 初始类中心的选取，明显会影响样本被分类的类别。不过这个中心点又受到第一条的样本顺序的输入影响。
3. Kmeans初始类簇中心点的选取?
   - k-means++算法选择初始seeds的基本思想就是：初始的聚类中心之间的相互距离要尽可能的远。
   - 从输入的数据点集合中随机选择一个点作为第一个聚类中心
   - 对于数据集中的每一个点x，计算它与最近聚类中心(指已选择的聚类中心)的距离D(x)
   - 选择一个新的数据点作为新的聚类中心，选择的原则是：D(x)较大的点，被选取作为聚类中心的概率较大
   - 重复2和3直到k个聚类中心被选出来
   - 利用这k个初始的聚类中心来运行标准的k-means算法

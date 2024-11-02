
### 基本原理
参考链接：
- [中文链接解释](https://blog.csdn.net/ACM_hades/article/details/90752746)
- [英文链接解释](https://towardsdatascience.com/dbscan-clustering-explained-97556a2ad556)
- [聚类算法对比](https://scikit-learn.org/stable/modules/clustering.html)
### 优点
- 对于任意形状大小的数据集都可以进行clustering。
- 可以在cluster的时候同时发现noise点，并且相对于KMeans，本身对noise点或者说异常点outline并不敏感，因为clustering的过程是从local到global。
- 在clustering的时候没有偏倚，算法初始值对于clustering的结果影响不大。
- 不需要提前定义K，相对于没有prior knowledge的数据集更友好。
### 缺点
- 数据集密度不均匀，每一个cluster之间间距很大的时候，效果较差，因为clustering依靠local density由参数min sample，esp决定。所以组间距相对远的时候容易本分为多个小的cluster。
- 数据集太大的时候，收敛时间较长，因为default下distance定义为pairwise distance. Time complex为O($n^2$)。这一点可以通过构建Kd-tree或者Ball-tree来加速搜索邻居点。
- 相对于K类型的clustering算法，参数有两个，调参依赖自定义，特别是esp参数的定义依赖对于数据集本身的熟悉程度和理解。同时不同参数的组合对最终的影响很大。
### 知识点提炼
- 一般来说数据集是稠密并且不是凸的时候，用DBSCAN比KMeans更好。
### Engineer Work
- 工程上一般可以适用于1M以上的数据集，主要耗时间体现在计算距离并且搜索临近点上。此时distance计算可以由Kd-tree或者Ball-tree来改进。
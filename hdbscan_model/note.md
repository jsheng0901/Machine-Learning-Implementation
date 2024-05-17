
### 基本原理
参考链接：
- [中文链接解释](https://blog.csdn.net/ACM_hades/article/details/90906677)
- [聚类算法对比](https://scikit-learn.org/stable/modules/clustering.html)
- [官方文档](https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html)
- [最小生成树MST相关解释](https://zhuanlan.zhihu.com/p/34922624)
### 优点
- 对于任意形状大小的数据集都可以进行clustering。
- 输出结果的cluster可以是任意形状。
- 可以在cluster的时候同时发现noise点，并且相对于KMeans，本身对noise点并不敏感，因为clustering的过程是从local到global。
- 在clustering的时候没有偏倚，算法初始值对于clustering的结果影响不大。准确说没有随机初始值的影响，因为构建MST之后就是一个固定的tree graph。
- 不需要提前定义K，相对于没有prior knowledge的数据集更友好。
- 参数min_cluster_size比较直观，好依赖domain knowledge调参。
- 对比DBSCAN，不需要调参数eps，这个参数会影响整体cluster里面neighborhood的定义和结果。并且参数min_cluster_size比较好apply一些prior的domain knowledge。不是DBSCAN里面额eps，相对难选择。
- 更strong density的cluster结果，因为apply了新的mutual reachability distance对于dense cluster和noise更robust。这是因为密集区域的样本不受到MRD的影响，MRD还是两点之间距离，然而稀疏区域的样本点和其它点之间的距离被放大，MRD最小是核心距离。
- 引入cluster的stability来最终提纯cluster结果。
- 加入probability来显示一个点属于这个cluster的概率，由lambda_p得来。
### 缺点
- 数据集太大的时候，收敛时间较长，因为要构建mutual reachability distance用O($n^2$)time构建matrix, min spanning tree然后single linkage tree。
### 知识点提炼
- 一般来说数据集是稠密并且不是凸的时候，用HDBSCAN比KMeans更好。
- 参数选择上min_cluster_size large size will result as small number of clusters and more compact cluster shape。
- 参数选择上min_samples large value will cause data push away and more data consider as noise。因为这里的参数是k值，主要影响核心距离的定义，对于MRD距离都会被放大，这样更多的样本被分配到稀疏区域，原本密集的区域也将变的稀疏，所有更多的点会被归为noise。
### Engineer Work
- 工程上一般可以适用于1M以上的数据集，主要耗时间体现在构建mutual reachability distance matrix (用KDtree加速neighbor搜索)，min spanning tree(用Prim算法加速构建)然后single linkage tree(用UnionFind实现)。
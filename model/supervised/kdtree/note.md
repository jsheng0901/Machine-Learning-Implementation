
### 基本原理
参考链接：
- [中文链接解释](https://zhuanlan.zhihu.com/p/45346117)
- [中文链接解释及参考复现代码](https://leileiluoluo.com/posts/kdtree-algorithm-and-implementation.html)
- [中文链接图文并茂解释及基础数学知识](https://zhuanlan.zhihu.com/p/22557068)
- [英文解释及KNN相对知识点](https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote16.html)
- [sklearn中相关应用文档](https://scikit-learn.org/stable/modules/neighbors.html)
### 优点
- 每次查找都很准确，有绝对稳定地输出。
- 建立树的过程很直观很简单，构建多维度的二叉搜索树BST。
- 加速KNN等带有临近算法查找的model。
### 缺点
- 维度太大的时候容易造成Curse of Dimensionality，降低搜索和建立树的效率。
- 维度划分死板，每层树用同一个维度split数据，用数据点本身作为split point。
### 知识点提炼
- 一般来说建立KDtree的过程时间复杂度是O(n * log(n) * log(n))，因为每一次递归都要有O(n * log(n))时间对一半的数组进行排序，对于高度平衡的BST，总共有log(n)层。
- 搜索的过程是O(log(n))因为树本身已经是一颗BST。
- 由于高维度的诅咒问题，一般情况下我们需要样本数据 n >> 2^k，否则样本数量太少，第一在高纬度上计算距离意义很小，第二可能造成维度划分后没有数据在对应的region。
- 对于更高维度的数据，推荐使用LSH算法进行划分。详细见LSH章节。
### Engineer Work
- 一般来说直接调用sklearn中的KDtree包来实现。实际生产中当维度达到20以上的时候就不会那么高效了。对于低纬度多个点的case，ex: d <= 20, n > 1M 每次 search time < 1ms。
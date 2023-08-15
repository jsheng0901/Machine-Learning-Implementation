
### 基本原理
参考链接：
- [中文原理讲解](https://blog.csdn.net/chenhepg/article/details/105409153)
- [英文链接图文解释](https://medium.com/swlh/k-nearest-neighbor-ca2593d7a3c4)
- [sklearn中相关应用文档](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
### 优点
- 优化加速过的Knn速度很快。
- 非常容易理解和解释结果和算法过程。
- 对于有稀有类别的数据效果会更好。
- 特别适合多分类问题，当label有多个类别的时候，相比较SVM，LR之类的线性model。
### 缺点
- 需要手动设置参数k，k会随着数据量变化而变化。
- model是lazy learning模式，存储数据量巨大如果不优化的前提下，memory占用多。因为是基于距离的计算，没优化的model遇到维度高数据量大的dataset时候特别慢。
### 知识点提炼
- 参数解释
  - k，选取多少个邻居来决定最终点的label。如何选取K这里需要调参数，cross-validation进行grid search调参，大部分情况下K不能大于样本的平方根。
- 损失函数解释
  - 目标函数是找到最近的点，所以是相似距离，可以是曼哈顿距离，欧氏距离，或者cosine similarity。
- 正则化解释
  - model不适用
- 其它要点 
  - 对于数据需要先标准化/归一化一下，确保不同维度的数据在同一个scale下，否则会出现大维度的数据在距离计算中占比太大，影响结果，所有distance base的model都需要这样提。
  - 优化加速参考KDtree，KBall-tree。不优化的KNN时间复杂度对于一个点来说是O(nd)，空间复杂度是O(nd)，n是总样本数，d是每个样本维度。
### Engineer Work
- 一般来说直接调用sklearn中的KNN包来实现。实际生产中普通版本的KNN目前来说在(100, 4)的数据集上需要1ms左右在8cores/16Memory上。
- 实际工作中，涉及高纬的数据，一般都需要降维，一来是可以克服维度灾难诅咒对于距离计算，二来可以加速，因为distance的计算时间复杂度是O(d)级别。
- 实际工作中KNN一般都是优化过的KDtree版本，详细性能参考KDtree板块。
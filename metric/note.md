## supervised learning (有监督学习)

### classification (分类)
#### 参考链接：
- [中文翻译AAAMLP中的metric章节](https://zhuanlan.zhihu.com/p/476927099)
- [英文常见终于得metric总结](https://www.analyticsvidhya.com/blog/2019/08/11-important-model-evaluation-error-metrics/)
#### 知识点提炼:
1. Accuracy，准确率，100个样本有90个分类正确则为90%。但是当样本的label分类不均匀的时候并不推荐使用accuracy，100个样本，90个负样本，10个正样本，如果都predict为负，accuracy同样为90%当此时显然没有意义
2. Precision，精准率，Precision = TP/(TP+FP)，简单说就是模型预测为positive的样本里面有多少预测对的。如果我们在意negative样本的预测准确度，那么我们不希望把negative的样本预测为positive，也就是说我们不希望FP高，因此此时我们更加关注precision的值。
3. Recall，召回率，Recall = TP/(TP+FN)，简单说就是数据本身是positive的样本里面有多少预测对的。如果我们在意positive样本的预测准确度，那么我们不希望把positive的样本预测为negative，也就是说我们不希望FN高，因此此时我们更加关注recall的值。
4. F1 Score，F1 = (2 * Precision * Recall)/(Precision + Recall)，F1 score是一个结合了精准率和召回率的指标。它被定义为精准率和召回率的简单加权平均值。在样本标签分布不均匀的时候我们如果并不太在乎positive或者negative类别的分类情况，当我们选取阈值的时候，可以直接参考F1 score，取一个全局最优的情况。阈值为正样本负样本的threshold，一般为0.5。
5. ROC 曲线，TPR = TP/(TP+FN)，FPR = FP/(TN+FP)，x轴 - FPR，y轴 - TPR。TPR就是recall，而FPR是实际为negative的样本我们有多少个预测准确。此时我们希望的是TN更大，而FP更小，也就是FPR越小越好。
6. AUC，ROC曲线下面积，范围在[0, 1]，AUC=1意味着你有一个完美的模型，但一般不太可能，AUC=0表示模型非常差(或非常好！)。可以尝试反转预测的概率。AUC=0.5意味着预测是随机的。对于任何二分类问题，如果预测所有目标为0.5，将得到0.5的AUC。AUC的含义：例如AUC=0.85，表示随机从正样本选择一个样本然后随机从负样本选择一个样本，正样本的预测概率大于负样本的预测概率的可能性为0.85。AUC是对所有可能的分类阈值的效果进行综合衡量，一种解读方式是看作模型将某个随机正类别样本排列在某个随机负类别样本之上的概率。ROC和AUC可以用来选择阈值，一般来说越大的AUC越好对应选择的阈值。或者当我们更加关心positive样本的时候同一个FPR下TPR越高越好。一般情况下我们用来对比model的performance，而不是单一看具体某个threshold的model下的情况。
7. 多分类的precision和recall，先计算每一个label的precision和recall，每一个label的计算是把当前class当做1，其它class当做0，比如有三个class，y_true = [1, 2, 3]，y_pred = [1, 1, 3]，则对于class1 我们有 y_true = [1, 0, 0]，y_pred = [1, 1, 0] precision = 1/2。其它同理，具体有一下几种算总precision和recall的方法：
   - micro，precision = TP1 + TP2 + TP3 / TP1 + TP2 + TP3 + FP1 + FP2 + FP3，算出每一个class的TP和FP然后合在一起计算，recall同理。
   - macro，precision = precision1 + precision2 + precision3 / 3，单独计算后，直接求所有class一起的平均值，recall同理。
   - weighted，precision = class1 number/total number * precision1 + class2 number/total number * precision2 + class3 number/total number * precision3，和macro一样，就是对每个class不再是同样的权重而是根据占比分配权重。

### regression (回归)
#### 参考链接：
- [中文翻译AAAMLP中的metric章节](https://zhuanlan.zhihu.com/p/476927099)
#### 知识点提炼:
1. MAE，平均绝对误差，这是所有绝对误差的平均值，它找到预测值和真实值之间的平均绝对距离。
2. MSE，均方误差，可能是用于回归问题的最流行的评估指标，它本质上是找到预测值和真实值之间的平均平方误差。MAE 比 MSE 对异常值更稳健，主要原因是在 MSE 中，通过平方误差，异常值在误差中得到更多的关注，并影响模型参数。
3. RMSE，均方根误差，RMSE = SQRT(MSE)，常见的指标之一，用于降低MSE误差对于异常值的敏感度。
4. MAPE，平均绝对百分比误差，MAPE = Mean(np.abs(y_t - y_p) / y_t)。
5. R^2，决定系数，表示模型对数据的拟合程度。接近1.0的R方表示模型与数据吻合得很好，而接近0表示模型不太好。当模型只是做出荒谬的预测时，也可能是负的。


## clustering (聚类学习)

#### 参考链接：
- [sklearn官方文档总结聚类model和metric](https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation)
- [英文论文原链接密度类聚类指标DBCV](https://www.dbs.ifi.lmu.de/~zimek/publications/SDM2014/DBCV.pdf)
#### 知识点提炼:
这里只讨论内部指标，也就是当我们没有label的时候做clustering的评估指标，以上链接包含内部和外部指标：
1. WCSS，Within-Cluster Sum of Squares，计算每个cluster内的点到中心点的距离和，此数据越小越好，不过很明显只能用于有K选择的类型的clustering才有中心点，同时此数据会随着K越多越小，极端值每个点是一个cluster。只能作为一个选择K的elbow-method参考方法，没有太多其它价值。
2. Silhouette Coefficient，轮廓系数 [-1, 1] 之间，此系数描述的是样本的组内距离和cluster之间的组间距离的比例，很明显当分类比较好的时候此系数很高，但是同样也有个问题就是因为是计算距离，所以对于凸的cluster明显会更高，对比其它形状的cluster，比如用于密度类型的DBSCAN的结果就不会很好，这样也就不适用于密度类型的clustering指标。
3. Calinski-Harabasz Index，此系数越大越好，本质上此系数是衡量组间协方差和组内协方差的比例，数值越小说明组间协方差很小，组间边界不明显。此系数一般用来作为第二指标结合第一个一起，但是此系数的有点就是快，只需要计算协方差，而第一个需要计算每个点到其它点的距离，O(n^2)的计算量。不过同第一个一样不适合用作为密度类型的不同形状的clustering指标。
4. Davies-Bouldin Index，此系数越小越好，本质上是计算了每个组间的最大相似度的均值，所以组间相似度如果为0那说明分类极好。不过同样需要计算记录也不太适用于作为密度类的clustering指标。
5. DBCV，此系数 [-1, 1] 之间，越接近1越好，此方法本质上和轮廓系数一样还是计算的组内聚类和组间距离的比例，但是此算法参考HDBSCAN里面的核心距离点的定义和互达距离的定义来改善对于流体型的cluster普通欧式距离带来的问题，具体参考HDBSCAN章节的解释和论文具体公式。对于密度类的clustering或者说非凸的shape的clustering推荐此方法。
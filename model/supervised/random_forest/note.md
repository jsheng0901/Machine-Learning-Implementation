
### 基本原理
参考链接：
- [中文简单直接的原理讲解](https://zhuanlan.zhihu.com/p/97753849)
- [中文详细介绍和理论论文研读](https://blog.csdn.net/weixin_44750583/article/details/99431770)
- [英文原理讲解及例子](https://towardsdatascience.com/an-implementation-and-explanation-of-the-random-forest-in-python-77bf308a9b76)
- [sklearn中相关应用文档以及详细解释](https://scikit-learn.org/stable/modules/ensemble.html)
### 优点
- 每一个树都是CART树，容易理解和解释结果和算法过程。可以可视化结果解释。
- 对比简单的单一模型，集成学习模型可以防止过拟合。RF的最大好处在于，行抽样和列抽样的引入让模型具有抗过拟合和抗噪声的特性。 
- 对于树类型的模型，不需要对数据做预处理，满足各种类型的数据无论是离散和连续，不需要归一化，不需要标准化数据，可以将缺失值单独作为一类处理，也不需要满足什么数据前提假设。能灵活地处理各种类型的数据。
- 集成学习模型的非线性变换比较多在构建一系列的树的时候，表达能力强，而且不需要做复杂的特征工程和特征变换。
- 可解释性强，可以自动做特征重要性排序。因此常作为发现特征组合的有效思路。
- 可以解决几乎所有单一树模型的缺点。又同时集成了所有单一树模型的优点。
- 可以处理多维度输出的分类问题，及多分类问题。
- 可以交叉验证的剪枝来选择模型，从而提高泛化能力。
- 在相对较少的调参时间下，预测的准确度较高。
- 不同树的生成是并行的，训练速度优于GBDT算法，特别是当树的个数比较多的时候。
- RF的训练效率会高于GBDT，因为在单个决策树的构建中，GBDT使用的是‘确定性’决策树，在选择特征划分结点时，要对所有的特征进行考虑，而随机森林使用的是‘随机性’特征数，只需考虑特征的子集。
- RF相比较GBDT对异常值不敏感，这也是为什么RF更偏向于降低variance。
### 缺点
- 对比GBDT，RF的起始性能较差，特别当只有一个基学习器时，随机森林通常会收敛到更低的泛化误差。RF一般降低variance，而GBDT一般降低bias。
- 对于高维稀疏特征，如果feature个数太多，并且很稀疏，每一棵回归树都要耗费大量时间，这一点上大大不如SVM。并且很容易过拟合，即使拟合很多子树。
- 工程上虽然可以对每棵树进行parallel训练加速训练时间，但是会消耗memory。
### 知识点提炼
- 参数解释
  - n_estimators，构建多少个并行的DT，一般来说RF每棵树都很简单，所以总共多少棵树一般选择比较大的值比如200，然后配合early stopping防止过拟合。
  - max_depth，单个决策树的最大深度，和减枝有关，如果不设置，则一直build树到底。和GBDT不一样的是，这个参数对于RF来说一般不会限制，因为我们需要每棵树在训练样本的子集上达到最好，至于拟合问题，最后会通过集合所有树结果来达到降低过拟合。
  - min_samples_split，单个决策树分裂后子树最小需要多少个样本才可以去分裂一个树节点。
  - min_samples_leaf，单个决策树节点构建叶子结点的时候，叶子结点最少需要多少个样本来构建。如果分裂后低于这个参数则意味着不再分裂此节点。此参数可能会影响回归任务的平滑效果。
  - min_impurity_decrease，单个决策树节点最小需要降低多少impurity，如果分裂后低于此参数则意味着不再分裂此节点。此参数可以达到early stopping的效果，防止过拟合太多的单一决策树。
  - max_features，每次构建训练单一树用的子集的时候可以最多选择的特征个数，如果不给定的话，分类问题默认使用所有特征的个数开方后的结果，回归问题默认使用所有特征。
- 损失函数解释
  - 回归问题，CART回归树，平方误差最小化。及最小化分裂后区域target值的variance。
  - 分类问题，CART分类树，基尼指数（基尼不纯度）= 样本被选中的概率 * 样本被分错的概率。
- 正则化解释
  - bootstrap，行抽样，取值为整个样本个数，但是随机森林使用的是放回抽样，也就是说有的样本会被选中多次，如果抽样次数足够多的时候，大概每一轮抽样中会有36.8%的数据不被抽到。
  - max_features，列抽样，随机森林会随机选取m个特征 (m < M) 训练用于每一棵CART树的生成。当m越小时，模型的抗干扰性和抗过拟合性越强，但是模型的准确率会下降，因此在实际建模过程中，常需要用交叉验证等方式选择合适的m值。
  - 其它正则化的方法就是对每次拟合的单一CART树进行限制，此处参考decision tree里面的note详细介绍了单一树的正则化。
- 其它要点 
  - 大部分单一树的要点可以参考decision tree里面的note。
  - 总的来说，对于RF build的时间复杂度一般是O(m * n * log(n) * tree_depth)，predict的时间复杂度是O(log(n))，m是feature个数，n是样本个数，也就是构建每颗单一树的时间，因为所有树可以并行一起build，预测也是一样的原理，一起跑预测然后投票。
  - GBDT对比RF：
    - 1）集成的方式：随机森林属于Bagging思想，而GBDT是Boosting思想。
    - 2）偏差-方差权衡：RF不断地降低模型的方差，而GBDT不断地降低模型的偏差。
    - 3）训练样本方式：RF每次迭代的样本是从全部训练集中有放回抽样形成的，而GBDT每次使用全部样本，此处现在可以通过子采样来同样达到部分样本训练，不过是无放回的抽样。
    - 4）并行性：RF的树可以并行生成，而GBDT只能顺序生成(需要等上一棵树完全生成)。
    - 5）最终结果：RF最终是多棵树进行多数表决（回归问题是取平均），而GBDT是加权融合。
    - 6）数据敏感性：RF对异常值不敏感，而GBDT对异常值比较敏感。
    - 7）泛化能力：RF不易过拟合，而GBDT容易过拟合。
  - 所有树的模型都有一个共性就是可以得到特征重要性 (feature importance)，具体计算方式为，单一树此特征作为分裂点得到的impurity grain * 此特征被用作分裂点的次数 * 总共多少棵树用了此特征，再对所有特征的结果做归一化，得到最后每个特征的重要性。可用于特征选择，也可用于特征解释。
  - 随机森林虽然泛化能力很强，但是还是会过拟合，当树的个数越来越多的时候，泛化能力会趋于一个极限值，也就是会收敛到一个极限误差值，简单点说就是随机森林的泛化误差界与单个决策树的分类强度s成负相关，与决策树之间的相关性ρ成正相关，分类强度s越大且相关性ρ越小，泛化误差界越小，刚好随机森林中的随机性可以保证ρ越小，如果每棵树的s越大的话，泛化误差会收敛到一个很小的margin，这个margin越小越好，就是泛化误差越小。[参考讨论链接](https://www.zhihu.com/question/30295075)
### Engineer Work
- 一般来说直接调用sklearn中的RF包来实现。实际生产中普通版本目前来说在(150, 4)的数据集上每棵树拟合需要0.7s左右在8cores/16Memory上。
- 如需要再1M以上的大数据跑production，随机森林是个可行的选择。推荐参考spark版本的regression和classification。[参考link](https://spark.apache.org/docs/latest/ml-classification-regression.html)


### 面试问题总结
1. 随机森林需要交叉验证吗？
   - 直接答案是不需要，因为随机森林在训练过程中每次都会有一部分数据没有被用于训练，基于bootstrap的抽样，这部分数据我们一般叫OOB数据，我们可以用这部分数据作为验证集数据，而不需要额外划分数据作为验证集。[参考链接](https://blog.csdn.net/weixin_44750583/article/details/99431770)
   - 具体操作：
     - 对于数据集中每一个样本，收集出没有用此样本训练的决策树，组合成一个子随机森林
     - 对这个子随机森林进行误差计算，得到当前此样本的验证结果
     - 重复上述所有过程对每一个样本
     - 对所有样本最后的结果取平均值，作为整个随机森林的交叉验证结果
2. 如果有两个相关性很高的特征，跑完RF后，plot他们的feature importance会发生什么？
   - 首先对应模型本身的拟合来说，应该是没有任何问题的，无论这两个特征是否是重要的特征。
   - 如果这两个特征是不太重要的特征，那么两个的特征重要性不会有太多的区别。
   - 如果两个特征是重要的特征，当一个特征被选中为分裂特征的时候，此时数据集的不纯度已经很低了，另一个相关性高的特征几乎不会被选中为分裂特征，也就是说对应的特征重要性反而会很低。总的来说，两个相关性高的特征，特征重要性计算反而会有巨大的区别。
   - 终上所述，如果只考虑模型拟合，没有影响，只考虑特征选择也不会有影响，毕竟两个特征相关性高的特征本来就应该被drop掉一个。但是如果用作数据的解释，并不是一个可靠的方式。
   - 由此可见，虽然树模型对数据的输入没有硬性要求，但是无论何种模型，保证特征的diversity很重要。因此feature preprocess很重要，需要去除特征相关性高的特征。
3. 随机森林如何处理缺失值?
   - 方法1：简单的imputation
     - numerical 数值类型的缺失，采用同一个class下的中位数插补。
     - categorical 分类类型的缺失，采用同一个class下的众数插补。
   - 方法2：利用proximity matrix的插补
     - step1，先用方法1里面的操作对缺失值进行插补，然后构建森林并计算proximity matrix。
     - step2，回头看缺失值
       - 如果是分类变量，则用没有缺失的观测样本的proximity中的权重进行投票。选择频率加权后最高的那个样本对应的此特征的没有缺失的值来插补。
       - 如果是连续型变量，则用proximity矩阵进行加权平均的方法补缺失值，也就是把所有样本同一个特征没有的缺失值，根据proximity matrix结果进行加权，求出新的插补值。
     - step3，重复step2 4- 6此迭代，基本上可以得到一个performance更好的森林。
   - 注：proximity 用来衡量两个样本之间的相似性。原理就是如果两个样本落在树的同一个叶子节点的次数越多，则这两个样本的相似度越高。当一棵树生成后，让数据集通过这棵树，落在同一个叶子节点的样本对 (xi, xj) 的 proximity 值 P(i,j) 加 1 。所有的树生成之后，再利用树的数量来归一化 proximity matrix。 例如，proximity matrix 是一个(n, n)的矩阵，在构建森林后，p(i, j) 表示样本i和样本j掉入同一个叶子结点的次数通过树哥个数加权平均后的结果。
   - 参考[link](https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm)
4. 随机森林如何衡量变量重要性？
   - 同上述其它要点的倒数第二条，用降低Gini指数来衡量。
   - 降低accuracy来衡量，去OOB数据进行测试，得到误差1，然后随机改变OOB样本的第j列：保持其他列不变，对第j列进行随机的上下置换，得到误差2。至此，我们可以用误差1-误差2来刻画变量j的重要性。基本思想就是，如果一个变量j足够重要，那么改变它会极大地增加测试误差；反之，如果改变它测试误差没有增大，则说明该变量不是那么的重要。

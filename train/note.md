### Build Train Data (准备训练数据)
#### 参考链接：
- [cross-validation简单解释](https://www.cnblogs.com/pinard/p/5992719.html)
- [sklearn官方解释和例子](https://scikit-learn.org/stable/modules/cross_validation.html)
#### 知识点提炼:
1. 一般来说我们要先进行数据集split。分成train，validation，test，一般来说比例是8:1:1。或者train和validation在50% - 90%的比例之间。
2. Train主要用于训练模型，确定model的各种学习参数，尽量涵盖越多的数据类型，并且最好是能拿到实际数据的distribution下的同样的sample。可以最有效的避免后续over-fitting。
3. Validation主要用于调参数，并不参与梯度下降的过程或者说训练的过程，比如超参数：epoch，learning rate，number layers，number hidden state之类。
4. Test主要用于评估模型的performance，不参与训练也不参与调参，只在最后计算模型最终的metric使用，需要注意的是在实际情况中，test数据集一般采样最新的online数据，来达到模拟实际情况下的数据分布，从而更准确的来评估模型的performance。
5. 交叉验证cross-validation，当我们的数据量比如小于1万条数据，不是很多的时候我们才会使用cross-validation进行调参数，因为数据量够大的时候直接划分train，validation后足够满足训练的时候包含所有可能数据分布情况。注意核心思想训练的过程是学习一个函数的distribution的过程。越多的训练数据越准确，就不需要再通过重复的划分来达到训练的时候见到更多的数据。
   - LOOCV，对于具有n行的数据集，选择第1行进行验证，其余(n-1)行用于训练模型。对于下一个迭代，选择第2行进行验证，然后重置来训练模型。类似地，这个过程重复进行，直到n步或达到所需的操作次数。最终模型的performance是n个测试的平局值。
     - 优点
       - 简单，易于理解和实施，并且不受测试集合训练集划分方法的影响，最大限度的保证了训练见过更多的数据，来避免减低Bias，但是可能因为过度的训练会影响variance。
     - 缺点
       - 很明显所需的计算时间长，需要训练n次模型。
   - k-fold，原始数据集被平均分为k个子部分或折叠。 从k折或组中，对于每次迭代，选择一组作为验证数据，其余（k-1）个组选择为训练数据。最终模型的performance是k个测试的平局值。不难看出LOOCV就是k=n的特殊情况。一般情况k选择5-10.
     - 优点
       - 模型偏差低
       - 时间复杂度低
     - 缺点
       - 不适合不平衡数据集，随机的划分可能会导致有的validation里面没有少样本的数据，不过现在调用sklearn包里面的Stratified k-fold都会已经自动帮你划分好了fold，保证每个fold里面少样本的分布一致也就是。


### Parameters Tuning (调参)
#### 参考链接：
- [sklearn官方文档和例子](https://scikit-learn.org/stable/modules/grid_search.html#)
- [深度学习调参总结](https://zhuanlan.zhihu.com/p/46718023)
- [深度学习调参trick方法](https://zhuanlan.zhihu.com/p/24720954)
- [贝叶斯优化介绍](https://zhuanlan.zhihu.com/p/146633409)
- [贝叶斯优化原理带推导](https://www.cnblogs.com/milliele/p/17782631.html)
- [贝叶斯优化通俗易懂讲解](https://cloud.tencent.com/developer/article/1965403)
- [贝叶斯优化总结](https://cloud.tencent.com/developer/article/1381849?areaId=106001)
- [贝叶斯优化简述](https://www.jiqizhixin.com/articles/2020-10-05-2)
- [贝叶斯优化从简解释](https://leovan.me/cn/2020/06/bayesian-optimization/)
- [英文贝叶斯优化解释](https://medium.com/@okanyenigun/step-by-step-guide-to-bayesian-optimization-a-python-based-approach-3558985c6818)
- [英文贝叶斯优化从零实现](https://machinelearningmastery.com/what-is-bayesian-optimization/)
#### 知识点提炼:
1. Grid Search，一般来说会和cross-validation一起使用，对于给定的所有超参数，组合出所有可能的超参数组合，使用每组超参数训练模型并挑选验证集误差最小的超参数组合。网格搜索适用于三四个（或者更少）的超参数（当超参数的数量增长时，网格搜索的计算复杂度会呈现指数增长，这时要换用随机搜索）。
   - 优点
     - 全面找所有的参数组合，一般都可以找到最优的超参数组合在给定的范围内。
   - 缺点
     - 太消耗resource了，需要计算量大，时间长，对于超参数多的模型或者训练耗时的模型，一般不建议使用。实际环境中超过3个以上的超参数或者训练模型达到1M的以上的都不建议使用。
2. Random Search，对每个参数定义了一个分布函数并在该空间中sample并组合出候选超参数组合，候选组合个数由参数指定，之后和Grid Search一样，用cross-validation技术挑选验证集误差最小的超参数组合。当超参数组合多比如3-4个以上并且resource有限的时候优先考虑此方法。
   - 优点
     - 高效，快速找到可能的最优超参数组合。并且对比Grid Search对于连续地超参数空间，随机搜索反而可以找到更多的参数组合空间而不是固定的参数组合空间。
   - 缺点
     - 因为是随机取样的超参数组合，可能会miss最优解或者某一部分超参数搜索空间。
3. Bayesian Optimization，在探索参数空间的新区域之间进行权衡，可以根据前面的训练结果，来指导后面训练超参数设置，利用历史信息来找到快速最大化函数的参数。像随机搜索一样，贝叶斯优化是随机的。适用于和随机搜索差不多的搜索空间但是希望更准确的找到更好的参数组合。
   - 优点
     - 考虑之前的参数信息，不断地更新先验，对比其它两个并未考虑之前的参数信息。
     - 对比网格搜索，贝叶斯迭代次数少，速度快；网格搜索速度慢,参数多时易导致维度爆炸。
     - 不依赖人为猜测所需的样本量为多少，优化技术基于随机性，概率分布。
     - 在目标函数未知且计算复杂度高的情况下极其强大，比如深度学习，树类型的model。
     - 通常适用于连续值的超参，例如 learning rate，regularization coefficient。
     - 不易陷入局部最优，根据EI函数的设计。
   - 缺点
     - 可能会在local min/max 卡住当目标函数过多的山峰波谷的时候。需要更多的时间对比随机搜索。
4. AutoML，自动化调参工具，很费时间和resource。Bayesian Optimization就在其中使用。
5. 一般情况如果时间和resource都允许的情况下，贝叶斯SMBO是首选，但是也应该考虑建立一个随机搜索的base搜索参数范围。一般情况下超参数的搜索主要是要提高调参的时间，我们可以配early stopping策略和采样策略，来达到快速搜索出初步参数范围的效果。
6. 实际工作中先用Gird Search的方法，得到所有候选参数，然后每次从中随机选择进行训练，或者再用贝叶斯是比较常见的策略。

  
### Over-fitting & Under-fitting (过拟合 & 欠拟合)
#### 参考链接：
- [过拟合和欠拟合的总结及案例](https://zhuanlan.zhihu.com/p/670881437)
- [过拟合和欠拟合的总结](https://zhuanlan.zhihu.com/p/72038532)
- [过拟合和欠拟合的总结带实例分析](https://ljwsummer.github.io/posts/advice-for-applying-machine-learning.html)
#### 知识点提炼:
1. over-fitting的解决方法，参考下面的面试问题总结第一题。
2. under-fitting怎么解决?
   - 欠拟合表示模型太过简单，无法捕获数据中的关键特征和模式。模型在训练数据和测试数据上的性能都较差。
   - 数据角度：
     - 训练数据量太少了，不足以表达数据分布的情况。模型无法拟合到数据分布。方法是增加训练数据，参考data augmentation。
     - 训练数据的特征太少了或者不太能表达数据分布，方法是添加更多的特征或进行特征工程，以捕获更多数据的信息。
   - model角度：
     - model结构：
       - 核心思想是model太简单了，或者是不合适当前的数据分布
       - 使用更复杂的模型，例如增加神经网络的层数或增加决策树的深度，或者NLP用Transformer框架。
     - 参数的控制：
       - 核心思想是参数拟合分布不确定。
       - 同一个模型但是调整模型的超参数，如学习率、批量大小等，以改善模型的性能。一般情况下default状态下的模型足以不会太欠拟合。
       - 如果使用了正则化，可以降低正则化的强度，使模型更灵活。
     - 训练控制：
       - 同一个模型，适当的增加训练epoch，比如100 -> 200，如果我们在每一次的迭代中能清晰的看出performance在逐渐递增。
       - 如果使用了early stopping，先取消提前终止，多训练几轮看看情况，如果还是欠拟合，则考虑是否是上面的情况，比如模型本身太简单，或者数据太少。



### 面试问题总结
1. over-fitting怎么解决？
   - 什么是过拟合，随着训练过程的进行，模型复杂度增加，在training data上的error渐渐减小，但是在验证集上的error却反而渐渐增大——因为训练出来的网络过拟合了训练集, 对训练集外的数据却不work, 这称之为泛化(generalization)性能不好。泛化性能是训练的效果评价中的首要目标，没有良好的泛化，就等于南辕北辙, 一切都是无用功。简单来说就是，只学习到了训练数据集的数据分布，对实际测试数据集分布无法拟合。
   - 数据角度：
     - 训练数据集样本太少无法代表实际情况中的样本分布。解决方法很直接，增加训练样本，参考data augmentation。
     - 训练数据集和测试数据集的样本分布不一致。解决方法，重新构造训练数据集，重新做sampling，参考如何正确的分train/evaluation/test数据集。
     - 训练数据集太单一，不抗噪，如果前面两个方法都受限不能实现的情况下，解决方法，在原始数据中增加人工生成的噪音，或者重采样，来重新构建新的训练数据集。
     - 训练数据太多噪音，导致模型拟合这些噪声，增加了模型复杂度，本质上和第一条是一样的，数据不能代表实际情况中的样本分布。
     - 训练数据的特征太多某些特征之间相关性太高，导致模型过度拟合数据，方法就是减少相关性高的特征数量。
   - model角度：
     - model结构：核心思想是model太复杂了要简化model结构。
       - dropout，在DeepLearning里面，在训练的运行的时候，让神经元以超参数p的概率被激活(也就是1-p的概率被设置为0)，每个w因此随机参与，使得任意w都不是不可或缺的，效果类似于数量巨大的模型集成。等价于ensemble learning。
       - 对于树类型的model或者neural-network，简化model的设计比如降低控制树嗯对生长深度，比如神经网络不要设计太多层。具体参考每个model的章节。
     - 参数的控制：核心思想是参数的变化幅度太大了，需要加以控制，即可避免过拟合又可以加速converge。
       - BN/LN，这个方法给每层的输出都做一次归一化(网络上相当于加了一个线性变换层)，使得下一层的输入接近高斯分布。这个方法相当于下一层的w训练时避免了其输入以偏概全，因而泛化效果非常好。
       - L2正则化，目标函数中增加所有权重w参数的平方之和, 逼迫所有w尽可能趋向零但不为零。因为过拟合的时候, 拟合函数需要顾忌每一个点，最终形成的拟合函数波动很大，在某些很小的区间里，函数值的变化很剧烈，也就是某些w非常大。为此，L2正则化的加入就惩罚了权重变大的趋势。
       - L1正则化，目标函数中增加所有权重w参数的绝对值之和，逼迫更多w为零(也就是变稀疏。L2因为其导数也趋0，奔向零的速度不如L1给力了)。大家对稀疏规则化趋之若鹜的一个关键原因在于它能实现特征的自动选择。一般来说，xi的大部分元素（也就是特征）都是和最终地输出yi没有关系或者不提供任何信息的，在最小化目标函数的时候考虑xi这些额外的特征，虽然可以获得更小的训练误差，但在预测新的样本时，这些没用的特征权重反而会被考虑，从而干扰了对正确yi的预测。稀疏规则化算子的引入就是为了完成特征自动选择的光荣使命，它会学习地去掉这些无用的特征，也就是把这些特征对应的权重置为0。
     - 训练控制：核心思想是model已经学习到一定程度了，不需要在过度学习了，其实还是参数控制，过度学习会导致参数过于拟合。
       - 提前终止(early stopping)，理论上可能的局部极小值数量随参数的数量呈指数增长，到达某个精确的最小值是不良泛化的一个来源。实践表明，追求细粒度极小值具有较高的泛化误差。这是直观的，因为我们通常会希望我们的误差函数是平滑的，精确的最小值处所见相应误差曲面具有高度不规则性，而我们的泛化要求减少精确度去获得平滑最小值，所以很多训练方法都提出了提前终止策略。典型的方法是根据交叉叉验证提前终止: 若每次训练前，将训练数据划分为若干份，取一份为测试集，其他为训练集，每次训练完立即拿此次选中的测试集自测。因为每份都有一次机会当测试集，所以此方法称之为交叉验证。交叉验证的错误率最小时可以认为泛化性能最好，这时候训练错误率虽然还在继续下降，但也得终止继续训练了。
       - 使用交叉验证的方式来调参，每次验证记录验证数据集的误差，选出一组参数让验证eval error最接近 train error。
2. under-fitting怎么解决？
   - 参考知识点提炼第二条
3. 介绍一下L1和L2范数，L1和L2的差别，为什么一个让绝对值最小，一个让平方最小，会有那么大的差别呢？ 
   - L1范数: 为x向量各个元素绝对值之和。
   - L2范数: 为x向量各个元素平方和的1/2次方，L2范数又称Euclidean范数或者Frobenius范数。
   - Lp范数: 为x向量各个元素绝对值p次方和的1/p次方。
   - L1范数可以使权值稀疏，方便特征提取。 
   - L2范数可以防止过拟合，提升模型的泛化能力。
   - 看导数L1导数是1，L2导数是x，在靠进零附近，L1以匀速下降到零，而L2则完全停下来了。这说明L1是将不重要的特征(或者说, 重要性不在一个数量级上)尽快剔除, L2则是把特征贡献尽量压缩最小但不至于为零。两者一起作用，就是把重要性在一个数量级(重要性最高的)的那些特征一起平等共事，及剔除不重要的特征，平滑重要的特征。
4. L1和L2正则先验分别服从什么分布？
   - L1是拉普拉斯分布，L2是高斯分布。
   - 从函数分布来看：
     - L2，![高斯分布](/pics/normal_distribution.png)
     - L1，![拉普拉斯分布](/pics/laplace_distribution.png)
     - 从上面两个图可以看出来，L2先验的值趋向零周围，L1先验的值趋向零本身。
   - 从数学推导来看：
     - 数学推导[链接](https://blog.csdn.net/weixin_43786241/article/details/109605265)
5. 什么是正则化？
   - 正则化是针对过拟合而提出的，以为在求解模型最优的是一般优化最小的经验风险，现在在该经验风险上加入模型复杂度这一项（正则化项是模型参数向量的范数），并使用一个rate比率来权衡模型复杂度与以往经验风险的权重，如果模型复杂度越高，结构化的经验风险会越大，现在的目标就变为了结构经验风险的最优化，可以防止模型训练过度复杂，有效的降低过拟合的风险。
6. 总结一些DeepLearning的调参方式?
   - 参考[链接](https://www.zhihu.com/question/41631631)

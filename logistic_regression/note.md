
### 基本原理
参考链接：
- [中文原理讲解及并行运算解释](https://zhuanlan.zhihu.com/p/74874291)
- [英文链接损失函数推导](https://medium.com/analytics-vidhya/logistic-regression-with-gradient-descent-explained-machine-learning-a9a12b38d710)
- [英文链接sigmoid函数推导](https://towardsdatascience.com/derivative-of-the-sigmoid-function-536880cf918e)
- [sklearn中相关应用文档](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
### 优点
- 非常容易理解和解释结果和算法过程。
- LR可以产生概率，并且很容易扩展到多分类。
- 当数据是线性可分的时候效果很好。
- 运行速度很快，对于大规模数据。
- feature的coefficient可以提供含义，代表log odds增加或减少了单位coefficient，可预判feature重要性的方向。
### 缺点
- 当维度很高的时候，维度最后不能超过样本数量，否则容易over-fitting，需要引入regularization。
- LR受数据分布影响，尤其是样本不均衡时影响很大，需要先做平衡。
- 存在线性假设在dependent variable和independent variable之间，对非线性问题很难分类。
- 各个feature之间最好没有correlation，assume特征之间相互独立。
### 知识点提炼
- 参数解释
  - learning_rate，梯度下降幅度。一般设置0.1就行。
  - n_iterations，最大迭代次数，跑gradient descent时候最大的轮数。
  - penalty，{‘l1’, ‘l2’, ‘elasticnet’, None}, default=’l2’，正则化的类型，一般选用l2。
  - C，正则化强度的倒数，必须是正数。与SVM里面一样，值越小说明penalty越大。
- 损失函数解释
  - 损失函数由MLE推导出来，cross-entropy推导出来的也是一样。![loss function](/pics/logistic_regression_loss_function.jpg)。
- 正则化解释
  - l1，加上L1范数容易得到稀疏解，LASSO回归，相当于为模型添加了这样一个先验知识：w服从零均值拉普拉斯分布。
  - l2，范数的平方，加上L2正则相比于L1正则来说，得到的解比较平滑（不是稀疏），但是同样能够保证解中接近于0（但不是等于0，所以相对平滑的维度比较多，降低模型的复杂度。Ridge回归，相当于为模型添加了这样一个先验知识：w 服从零均值正态分布。
  - elasticnet，混合l1，l2，l1_ratio是控制l1，l2比例的参数，取值0-1区间。
- 其它要点 
  - 一般来说正则化之所以能够降低过拟合的原因在于，正则化是结构风险最小化的一种策略实现。结构风险最小化是指在经验风险最小化的基础上（也就是训练误差最小化），尽可能采用简单的模型，以此提高泛化预测精度。
  - 加入正则化同时具备feature selection和判断feature importance的能力。
  - 优化加速参考spark版本的model。
  - LR适合对数据做transfer成离散数据，比如age > 30 转化成0，1。因为离散后速度更快计算，简化模型，模型更稳定和增加泛化能力。
  - 不用MSE做损失函数，因为MSE平方差后 损失函数是非凸的，很难找到全局最优解。
### Engineer Work
- 一般来说直接调用sklearn中的logistic包来实现。实际生产中普通版本目前来说在(150, 4)的数据集上需要10ms左右在8cores/16Memory上。
- 如需要再1M以上的数据跑production，推荐用spark的版本来实现。
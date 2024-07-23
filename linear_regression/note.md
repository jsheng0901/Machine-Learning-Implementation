
### 基本原理
参考链接：
- [中文原理详解](https://blog.csdn.net/iqdutao/article/details/109402570)
- [sklearn中相关应用文档](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
### 优点
- 非常容易理解和解释结果和算法过程。
- 非常快，对于小数据集有时候效果很好，对于大数据集可以作为base model。
- 是很多非线性模型的基础。
- feature的coefficient可以提供含义，直观的表示了每个特征单位增长对于最后结果的影响。
### 缺点
- 当维度很高的时候，维度最后不能超过样本数量，否则容易over-fitting，需要引入regularization。
- 存在线性假设在dependent variable和independent variable之间，对非线性问题很难拟合。
- 很难表达高纬度复杂关系的大数据。
### 知识点提炼
- 参数解释
  - learning_rate，梯度下降幅度。一般设置0.1就行。
  - n_iterations，最大迭代次数，跑gradient descent时候最大的轮数。
  - penalty，{‘l1’, ‘l2’, ‘elasticnet’, None}, default=’l2’，正则化的类型，一般选用l2。
- 损失函数解释
  - 损失函数一般是回归问题常见的MSE。
- 正则化解释
  - l1，加上L1范数容易得到稀疏解，LASSO回归，相当于为模型添加了这样一个先验知识：w服从零均值拉普拉斯分布。
  - l2，范数的平方，加上L2正则相比于L1正则来说，得到的解比较平滑（不是稀疏），但是同样能够保证解中接近于0（但不是等于0，所以相对平滑的维度比较多，降低模型的复杂度。Ridge回归，相当于为模型添加了这样一个先验知识：w 服从零均值正态分布。
  - elasticnet，混合l1，l2，l1_ratio是控制l1，l2比例的参数，取值0-1区间。
- 其它要点
  - 线性回归遵从的强假设
    - Y对X而言是线性关系
    - 误差服从均值为0的正态分布
    - 不同特征之间相互独立，也就是不能有共线性，否则容易使某些特征的权重过高，导致模型不稳定，也就是容易过拟合
    - 样本长度远大于特征个数
  - 共线性可以通过VIF来判断，一般来说VIF值大于10就表示有严重的共线性
  - 移除共线性的方法
    - 手动移除，如果相关系数大于0.7一般，手动删除其中一个特征
    - 增加样本容量，也就是增加训练数据
    - 利用L1回归，限制参数的大小，来达到特征选择，移除共线性的特征权重
  - 常见的评估指标包括，MSE，MAE，R^2。具体计算参考loss function章节。
  - 求最优解的过程可以采用最小二乘法直接得到最优值，也可以采用gradient descent，但是实际工程中最小二乘法计算inverse矩阵特别费时和空间，所以对于大数据一般采用SGD来求解。
### Engineer Work
- 一般来说直接调用sklearn中的linear regression包来实现。实际生产中普通版本目前来说在(150, 4)的数据集上需要10ms左右在8cores/16Memory上。
- 如需要再1M以上的数据跑production，推荐用spark的版本来实现。


### 面试问题总结

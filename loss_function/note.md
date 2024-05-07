## supervised learning (有监督学习)

### classification (分类)
#### 参考链接：
- [常见的损失函数带复现](https://zhuanlan.zhihu.com/p/692647626)
- [深度学习中常见的损失函数及pytorch中的应用](https://zhuanlan.zhihu.com/p/561589376)
- [深度学习中损失函数总结应用及trick](https://zhuanlan.zhihu.com/p/104273926)
#### 知识点提炼:
1. LogLoss or LogisticsLoss，对数损失，直观地理解为预测概率与真实标签之间的“距离”。如果预测的概率完全准确（即，对于真实标签为1的样本，预测概率也为1；对于真实标签为0的样本，预测概率也为0），那么对数损失为0。反之，如果预测的概率与真实标签完全不符，那么对数损失将会无限大。常见的情况适用于二分类问题，输出是一个数，再sigmoid，对应BCELoss，如果变成多分类问题，输出是C个class的数，再softmax，对应CELoss。
   - 优点：
     - 直接衡量预测概率的准确性，而不仅仅是预测标签的准确性，可以生成概率值的预测。
   - 缺点：
     - 对于预测概率的小变化非常敏感。例如，改变一个预测概率从0.01到0.02，和改变另一个预测概率从0.99到1.0，对于对数损失的影响是一样的，尽管后者的预测已经非常接近真实标签。
2. Cross Entropy Loss，交叉熵损失，同上面的对数损失含义，区别是一般用于多分类问题，因为它可以很好地处理概率输出。对于二分类问题，使用对数损失，它是交叉熵损失的一个特例。此loss有带权重的版本，主要针对样本类别不平衡的情况。少样本的loss权重较小，变化较慢，这样梯度下降的时候会更多的关注到小样本的loss。
   - 优点
     - 同对数损失
   - 缺点
     - 同对数损失
3. Hinge Loss，常用于支持向量机（SVM）和一些类型的感知器算法的损失函数，如果模型的预测值与真实标签的乘积大于1，那么损失为0，得到稀疏解，使得少量的支持向量就能确定最终的超平面；否则，损失为 1-y_true*y_pred。这意味着，即使模型的预测是正确的，只要它的置信度不够（即，y_true*y_pred < 1），就会产生一定的损失。常用于二分类问题也可以变成多分类问题的loss，区别就是one-many或者many-many的方法对待样本。
   - 优点
     - 它对于正确分类的样本产生的损失较小，这使得它在处理大规模且高维度的数据集时非常有效。
   - 缺点
     - 它不是可微的，这使得它不能直接用于需要损失函数可微的优化算法，如梯度下降。比如SVM需要转化成对偶问题来进行凸优化。
4. Focal Loss，主要用于解决多分类任务中样本不平衡的问题，还有比如多任务中，不同任务的样本比例不平衡的情况。

### regression (回归)
#### 参考链接：
- [中文翻译AAAMLP中的metric章节](https://zhuanlan.zhihu.com/p/476927099)
- [常见的损失函数带复现](https://zhuanlan.zhihu.com/p/692647626)
- [一些实际操作中回归问题的损失函数选择方法](https://zhuanlan.zhihu.com/p/378822530)
#### 知识点提炼:
1. MAE，平均绝对误差，这是所有绝对误差的平均值，它找到预测值和真实值之间的平均绝对距离。
   - 优点
     - 它对于异常值不那么敏感。这是因为在计算平均绝对误差时，我们不会对差异进行平方，所以差异较大的样本不会有过大的权重。收敛速度快。
   - 缺点
     - 同样和MSE一样，优点也可以是缺点，取决于适用的场景。如果我们的数据中包含很多噪声，那么平均绝对误差可能是一个更好的选择。还有个问题是在y_true = y_pred的拐点时候，loss为0，不连续可导，不方便优化。
2. MSE，均方误差，可能是用于回归问题的最流行的评估指标，它本质上是找到预测值和真实值之间的平均平方误差。MAE 比 MSE 对异常值更稳健，主要原因是在 MSE 中，通过平方误差，更大的波动对于整个loss，异常值在误差中得到更多的关注，并影响模型参数。
   - 优点
     - 均方误差的一个主要优点是它对于异常值非常敏感。这是因为在计算均方误差时，我们会对差异进行平方，所以差异较大的样本会有更大的权重。
   - 缺点
     - 缺点和优点是一样的，具体的影响取决于具体的应用场景。如果我们希望模型对异常值更敏感，那么均方误差是一个好的选择。但是，如果我们的数据中包含很多噪声，那么均方误差可能会导致模型过于复杂，从而过拟合。模型学习速度慢。
3. RMSE，均方根误差，RMSE = SQRT(MSE)，常见的指标之一，用于降低MSE误差对于异常值的敏感度。
   - 优点
     - 降低MSE里面对于异常值的敏感度。
   - 缺点
     - 同MSE
4. MAPE，平均绝对百分比误差，MAPE = Mean(np.abs(y_t - y_p) / y_t)。
5. Huber Loss，直观地理解为均方误差和平均绝对误差的结合。当预测值与真实值之间的差异较小（小于 ）时，Huber损失就变成了均方误差；而当差异较大时，它就变成了平均绝对误差。这使得Huber损失在处理有噪声的数据时比均方误差更稳健，因为它对于异常值不那么敏感。
   - 优点
     - 在保持对数据中的正常值敏感的同时，对异常值具有鲁棒性。这使得它在处理有噪声的数据时非常有效。误差大的时候是MAE，降低异常值影响，误差小的时候是MSE，减少异常值敏感度，同时加速梯度下降。这样同时兼顾了MAE和MSE的两个优点，又照顾到了两个的缺点。
   - 缺点
     - 它需要选择一个合适的delta值这个参数来决定什么时候是均方误差，什么时候是平均绝对误差，这需要根据具体的问题和数据来决定。不过这个参数一般可以通过cross-validation找到最佳值。
6. R^2，决定系数，表示模型对数据的拟合程度。接近1.0的R方表示模型与数据吻合得很好，而接近0表示模型不太好。当模型只是做出荒谬的预测时，也可能是负的。


## clustering or others (聚类学习或者其它类型任务)

#### 参考链接：
- [常见的损失函数带复现](https://zhuanlan.zhihu.com/p/692647626)
#### 知识点提炼:
1. KL 散度损失，也被称为相对熵，直观地理解为真实概率分布和预测概率分布之间的“距离”。如果预测的概率分布完全准确（即，与真实的概率分布完全一致），那么KL散度损失为0。反之，如果预测的概率分布与真实的概率分布完全不符，那么KL散度损失将会无限大。常用于概率分布的预测问题，比如聚类问题中最常见的T-sne模型。
   - 优点
     - 直接衡量预测概率分布的准确性，这使得它在需要考虑不确定性的问题中非常有用。
   - 缺点
     - 同log loss
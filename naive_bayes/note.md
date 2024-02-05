
### 基本原理
参考链接：
- [中文原理讲解及例子解释](https://zhuanlan.zhihu.com/p/37575364)
- [中文原理解释及离散和连续API例子解释](https://www.ngui.cc/51cto/show-708719.html?action=onClick)
- [中文解释及其例子](https://zhuanlan.zhihu.com/p/624002501?utm_id=0)
- [中文拉普拉斯平滑解释](https://www.python100.com/html/94252.html)
- [英文原理理解](https://www.analyticsvidhya.com/blog/2017/09/naive-bayes-explained/)
- [sklearn中相关应用文档](https://scikit-learn.org/stable/modules/naive_bayes.html)
### 优点
- 非常容易理解和解释结果和算法过程。
- 朴素贝叶斯算法简单容易实现，对异常值、缺失值不敏感。
- 当数据分布接近相互独立时，分类准确率高，可实现多分类。
- 运行速度很快，对于大规模数据。
- 对于生成式模型，模型可通过增量学习得到。
- 对于生成式模型，模型能够应付存在隐变量的情况，比如混合高斯模型就是含有隐变量的生成方法。
- 当feature是categorical时候一般比numerical情况下performance更好，因为numerical下有强假设normal distribution。
### 缺点
- 对数据的依赖性高，如果训练集误差大，最终效果就不好。
- 需要知道先验概率，收到先验概率准确性的影响。
- 如果各个特征之间依赖性较高，会降低分类效果。
- 对于categorical feature，没有在train过程中看见的feature会assign概率为0，此时会出现zero frequency的情况，需要平滑处理，例如Laplace estimation。
### 知识点提炼
- 参数解释
  - priors，每个class的先验概率。可以不设置，default按照class的频率来assign。
  - type，此参数并不是常规参数，主要是表示选择那种likelihood分布，比如连续数据选择高斯，离散数据选择Multinomial或者Bernoulli。
- 损失函数解释
  - 生成式模型，没有损失函数，此处贴出贝叶斯核心公式。![bayes rule](/pics/bayes_rule.png)
- 正则化解释
  - 无
- 其它要点 
  - 朴素贝叶斯的基本假设是条件独立性，即在类确定时，假设其各个特征相互独立。
  - 如果输入feature不符合高斯，则可以适当的应用数据转化，变成normal distribution。
  - 遇到条件概率相乘为0的时候，使用拉普拉斯平滑，拉普拉斯平滑的参数alpha值选择一般为1，alpha * k代表特征取值的个数，这个值越大，相应的拉普拉斯平滑所增加的概率值也就越小。
  - 朴素贝叶斯的模型比较简单，对异常值和缺失值有较高的容错，属于低方差模型。
  - 对于异常值和缺失值，由于朴素贝叶斯判别公式中，采用的是连乘的方式，因此个别的异常值，对于整体的结果影响不大。此外，最终的分类结果，是根据各个类别概率的排序来判别的结果，因此，个别异常值影响也不大。
  - 和LR对比，LR更适合大数据，NB更适合小数据。LR可以处理co-linearity用正则化，但NB不行。
### Engineer Work
- 一般来说直接调用sklearn中的NB包来实现。实际生产中普通版本目前来说在(150, 4)的数据集上需要2ms左右在8cores/16Memory上。
- 如需要再1M以上的数据跑production，推荐用spark的版本来实现。[参考例子](https://spark.apache.org/docs/latest/mllib-naive-bayes.html)

### 面试问题总结
1. 为什么朴素贝叶斯如此“朴素”？
   - 因为它假定所有的特征在数据集中的作用是同样重要和独立的。这个假设现实中基本上不存在，但特征相关性很小的实际情况还是很多的，所以这个模型仍然能够工作得很好。
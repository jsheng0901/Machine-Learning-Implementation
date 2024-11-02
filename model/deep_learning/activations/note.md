## 面试问题总结
1. 简单说下sigmoid激活函数？
   - sigmoid函数表达式如下![图片](/pics/sigmoid_formula.png)其中z是一个线性组合，比如z可以等于：b + w1*x1 + w2*x2。通过代入很大的正数或很小的负数到g(z)函数中可知，其结果趋近于0或1。
   - sigmoid函数g(z)的图形表示如下（ 横轴表示定义域z，纵轴表示值域g(z) ）![图片](/pics/sigmoid_plot.png)
   - sigmoid函数的功能是相当于把一个实数压缩至0到1之间。当z是非常大的正数时，g(z) 会趋近于1，而z是非常小的负数时，则g(z)会趋近于0。压缩至0到1的用处是这样一来便可以把激活函数看作一种“分类的概率”，比如激活函数的输出为0.9的话便可以解释为90%的概率为正样本。
   - sigmoid函数，是LR回归的压缩函数，它的性质是可以把分隔平面压缩到[0,1]区间一个数（向量），在线性分割平面值为0时候正好对应sigmoid值为0.5，大于0对应sigmoid值大于0.5、小于0对应sigmoid值小于0.5；0.5可以作为分类的阀值；exp的形式最值求解时候比较方便，用相乘形式作为logistic损失函数，使得损失函数是凸函数；不足之处是sigmoid函数在y趋于0或1时候有死区，控制不好在bp形式传递loss时候容易造成梯度消失。
2. 对比一下常见的几个激活函数比如Relu, Sigmoid, Tanh?
   - 参考资料，可视化26种激活函数，并且包含介绍，[链接](https://dashee87.github.io/deep%20learning/visualising-activation-functions-in-neural-networks/)
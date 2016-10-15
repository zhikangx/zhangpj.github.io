---
layout: post
title: Learning From Data Pt.2
date: 2016-10-14 21:09
categories: ML
summary: 我好想吃绝味鸭脖啊。
---

## Intro

这一篇主要是第一章最后一节的内容，另外介绍第二个机器学习模型：线性回归。

## Error Measure

上一篇中我们提到，机器学习的目标就是找到一个尽可能接近目标函数的 Hypothesis，那么为了评价 Hypothesis 与目标函数的相似程度，我们就需要定义一个 Error Measure：$E(h, f)$。在实用中，这个 Error Measure 通常是按数据点来定义的：$e(h(x), f(x))$，比如：

- Squared error：$e(h(x), f(x)) = (h(x) - f(x))^2$
- Binary error: $e(h(x), f(x)) = \left[ h(x) \neq f(x) \right]$

(我用 $[\cdot]$ 表示 “如果参数为真，则返回 1，否则返回 0”。)

整体的 $E(h, f)$ 就是逐点错误的平均值。

现在，我们就可以正式定义 $E_{in}$ 和 $E_{out}$ 了。

$$
E_{in}(h) = \frac{1}{N}\sum_{n = 1}^{N}e(h(\mathbf{x}_n), f(\mathbf{x}_n))
$$

$$
E_{out}(h) = \mathbb{E}_{\mathbf{x}}[e(h(\mathbf{x}), f(\mathbf{x}))]
$$

式中，$\mathbb{E}$ 表示期望。

理想情况下，这个错误函数 $e$ 应该是根据问题的需要来选择的。书中给出了一个指纹识别系统的例子。

{: .center}
![Fingerprint Verification System]({{ site.baseurl }}/images/finger.png)

如果一个超市来使用这个系统作为折扣发放的检验系统，那么这个错误函数应该是：

{: .center}
![Supermarket]({{ site.baseurl }}/images/supermarket.png)

因为对于超市来说，多给一个折扣并不是什么严重的问题，相比之下让应该得到折扣的客户无法拿到折扣要严重的多，所以错误函数中 _False Negative_ （$f(\mathbf{x}) = 1$，$h(\mathbf{x}) = -1$）的惩罚要比 _False Positive_（$f(\mathbf{x}) = -1$，$h(\mathbf{x}) = 1$）大得多。

但如果是 CIA 来使用这个系统作为其办公场所的安保系统，那么错误函数应该是：

{: .center}
![CIA]({{ site.baseurl }}/images/cia.png)

所以说错误函数应当是根据问题的需要来选择的，但有的时候无法这样做，所以也有许多的替代方案，比如前述的 Squared Error。

## Noise

现在我们来考虑一下数据中的噪声问题。首先回顾一下我们的 Learning Diagram，我们在上一篇的基础上加上了刚刚介绍的错误度量部分：

{: .center}
![Learning Diagram 3]({{ site.baseurl }}/images/learning-diagram-3.png)

在上图中，我们认为“目标函数”是一个函数，然而，在实际问题中，由于各种问题的影响（如数据收集时的错误），有可能出现同样的 $\mathbf{x}$，不同的 $y$ 的情况，所以，“目标函数”有时并不是一个函数。我们现在引入“目标分布”的概念，来代替之前的目标函数：

$$
P(y\vert\mathbf{x})
$$

现在，我们的数据点是通过 $\mathbf{x}$ 和 $y$ 的联合分布来产生的：

$$
P(\mathbf{x})P(y\vert\mathbf{x})
$$

对于上述有噪声的目标，我们可以认为其由两部分组成：

1. Deterministic Target Function：$f:\mathcal{X} \to \mathcal{Y}$；
2. Noise: $y - f(\mathbf{x})$。

现在，我们介绍了机器学习（有监督式机器学习）的所有基本概念，所以这里给出 Learning Diagram 的最终版：

{: .center}
![Learning Diagram 4]({{ site.baseurl }}/images/learning-diagram-4.png)

## Linear Regression

在进入理论部分之前，这里先介绍第二个机器学习模型：线性回归。

在机器学习领域，“回归”的含义就是实数输出。以之前的信用卡发放问题为例，回归模型的输出是“信用卡的额度”，而分类模型的则是输出是“是否发放信用卡”。

线性回归的模型形式十分简单：

$$
h(\mathbf{x}) = \mathbf{w}^T\mathbf{x}.
$$

在线性回归问题中，我们使用的错误度量是 squared error：

$$
\text{in-sample error: } E_{in}(h) = \frac{1}{N}\sum_{n = 1}^{N}(h(\mathbf{x}_n) - y_n)^2
$$

代入线性回归的模型：

\begin{align}
E_{in}(\mathbf{w}) =& \frac{1}{N}\sum_{n = 1}^N(\mathbf{w}^T\mathbf{x}_n - y_n)^2 \newline
=& \frac{1}{N}\parallel \mathbf{X}\mathbf{w} - \mathbf{y}\parallel{}^2
\end{align}

现在我们要最小化 $E_{in}$：

$$
E_{in}(\mathbf{w}) = \frac{1}{N}\parallel \mathbf{X}\mathbf{w} - \mathbf{y} \parallel{}^2 \\
\nabla E_{in}(\mathbf{w}) = \frac{2}{N}\mathbf{X}^T(\mathbf{X}\mathbf{w} - \mathbf{y}) = 0 \\
\mathbf{X}^T\mathbf{X}\mathbf{w} = \mathbf{X}^T\mathbf{y} \\
\mathbf{w} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y} \\
$$

这样我们就得到了 $\mathbf{w}$，也就是获得了最佳的 $h$，这就是线性回归模型。线性回归的 python 实现：[LinearReg](https://github.com/zhangpj/learning-from-data/blob/master/pymodels/LinearReg.py)（核心就一行……）。

现在我们获得了 $\mathbf{w}$，可以放入回归模型中来获得实数输出。事实上，我们也可以使用 $sign(\mathbf{w}^T\mathbf{x_n})$来产生二值输出，或者把线性回归得到的 $\mathbf{w}$ 作为分类模型的初始值，来降低模型的训练时间，所以线性回归也是可以用于分类问题的。

## Feature Transform

现在我们有了线性回归这个模型，但线性模型的表达能力较差，比如：

{: .center}
![Non-linear Hypothesis]({{ site.baseurl }}/images/nonlinear.png)

我们想要的边界很明显不是一条直线，这时候还能用线性回归吗？

我们首先考虑一下“线性模型”中“线性”的含义。这个“线性”显然指的不是 $\mathbf{x}_i$。因为如果式中有 $\mathbf{x}_i^2$，我们可将其记为一个新的变量。所以这里的“线性”指的是系数$\mathbf{w}$是线性的。

既然 $\mathbf{x}_i$ 是不是线性的对我们的模型形式没有任何影响，那么我们就可以将 $\mathbf{x}_i$ 进行转换，从而获得非线性的分类边界。比如，我们可以令 $\mathbf{z} = (x_1^2, x_2^2)$，然后在 $\mathcal{Z}$ 上做线性回归，这样得到的结果在 $\mathcal{X}$ 上就是非线性的：

{: .center}
![Feature Transformation]({{ site.baseurl }}/images/feature-transformation.png)

## Outro

本篇内容较散，首先补完了机器学习的基本概念，然后又介绍了一个新的模型，并通过这个模型引出了特征转换的概念。接下来几篇将会进入机器学习的理论基础部分，将我们在第一篇剩下的那个问题“机器学习是否可行？”彻底解决。

另外，这里还想说一下关于统计与机器学习的区别。在实用层面的区别我不太了解，但在理论层面，我认为两者最重要的区别就是统计学希望得到数据的真实分布，而机器学习只求能找到一个接近目标函数的 Hypothesis，所以相较于机器学习，统计学中的模型会有更多的限制。（想想 Gauss--Markov Theorem 你就明白我在说什么了。）

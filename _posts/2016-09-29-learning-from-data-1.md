---
layout: post
title: Learning From Data Pt.1
date: 2016-09-29 03:21
categories: ML
summary: 坂本真绫声音真好听。
---

最近我买了 Yaser S. Abu-Mostafa，Malik Magdon-Ismail 和林轩田三人写的那本 *Learning From Data*。这本书虽然不厚但是内容丰富，主要从理论角度介绍机器学习的相关概念。配合 youtube 上 Abu-Mostafa 教授在 Caltech 上课时的录像，可以作为不错的机器学习自学入门课程。这里把我的读书笔记发出来，大概到1月份发完。另外我可能会根据课程录像进行调整，一些地方会与书上内容不一致。

## Intro

第一章主要介绍机器学习的基本概念：

1. 什么是机器学习？

2. 机器学习是否可行？

## Learning Problem

对人来说，可以很容易判断一个物体是什么，但却很难给这个物体下一个准确的定义。人类是通过大量的观察来学习如何识别一个物体，机器学习也是一样。也就是说，对于那种有内在的规律，但是很难给出解析解的问题，机器学习是最合适的方法之一；但是如果一个问题有明显的解析解，比如知道物体的体积和密度求物体质量，机器学习算法虽然可用，但绝对不会是最好的方法，另外，如果一个问题根本没有内在规律<del>（如股市）</del>，那么机器学习算法是不可用的。

### Components of Learning

这里以一个简单的例子说明机器学习的组成部分。

一家银行希望通过某种方法可以自动化其信用卡发放的流程。这家银行每天发放大量的信用卡，所以有大量申请人的历史信息。现在来了一位新的申请人，其个人信息如下：

| Variable           | Value   |
| :----------------- | :------ |
| age                | 23      |
| gender             | male    |
| annual salary      | $30,000 |
| years in residence | 1 year  |
| years in job       | 1 year  |
| current debt       | $15,000 |
| ...                | ...     |

那么这家银行是否应该发放信用卡给他？

首先来规范化这个问题：

- 输入：$\mathbf{x}$ （这里就是客户的信息）
- 输出：$y$ （是否应该发信用卡？）
- 目标函数：$f: \mathcal{X} \rightarrow \mathcal{Y}$ （理想的信用卡发放决策方式）
- 数据：$(\mathbf{x}_1, y_1)$, $(\mathbf{x}_2, y_2)$, ..., $(\mathbf{x}_N, y_N)$ （人工审查时获得的历史记录）
- Hypothesis: $g:\mathcal{X} \rightarrow \mathcal{Y}$ （机器学习的结果，也是银行将要使用的模型，我们要使其尽可能地接近目标函数。）

把这些部分组合起来，就得到了如下的 Learning Diagram：

{: .center}
![Learning Diagram]({{ site.baseurl }}/images/learning-diagram.png)

我们从未知的目标函数中获得了一定量的数据，将这些数据放入机器学习算法中（图中的 Learning Algorithm），算法从 Hypothesis Set 里选出一个最好的作为 Final Hypothesis，使得我们可以对新的数据获得最接近真实情况的预测，这就是机器学习的大体上的流程。其中，Learning Algorithm 和 Hypothesis Set 合称机器学习模型。

### A Toy Model

继续上述的信用卡发放问题，对于一个客户$\mathbf{x} = (x_1, x_2, ..., x_d)$，我们使用这样一个公式来判断是否发放信用卡：

$$
\text{Approve if} \sum_{i = 1}^{d}w_ix_i > threshold,\\
\text{Deny if} \sum_{i = 1}^{d}w_ix_i < threshold.
$$

这就是一个简单的 Hypothesis Set，感知机 Perceptron：

$$
h(\mathbf{x}) = sign\left(\left(\sum_{i = 1}^{d}w_ix_i\right) - threshold\right)
$$

为了简化公式，我们在 $\mathbf{x}$ 中加入新的一维 $x_0$，同时将 threshold 记为 $w_0$，得到如下的形式：

$$
h(\mathbf{x}) = sign(\mathbf{w}^T\mathbf{x})
$$

同时介绍一个非常 Naive 的 Learning Algorithm：

- 我们有一个数据集：$(\mathbf{x}_1, y_1)$, $(\mathbf{x}_2, y_2)$, ..., $(\mathbf{x}_N, y_N)$;
- 从中选出一个分错了的数据：$sign(\mathbf{w}^T\mathbf{x}_n) \neq y_n$;
- 更新 $\mathbf{w}$: $\mathbf{w} \leftarrow \mathbf{w} + y_n\mathbf{x}_n$

{: .center}
![Perceptron Learning Algorithm]({{ site.baseurl }}/images/pla.png)

这个算法除了写作业和上课之外貌似真的没什么用……

### Types of Learning

机器学习的类型有好多种，比较常见的两种（我只知道这两种）是 Supervised Learning 和 Unsupervised Learning。Supervised Learning 的特点就是数据集有确定的 $y$，比如之前的感知机学习算法就是这种。Unsupervised Learning 的数据集没有 $y$，比如一些聚类算法就是这一种。此外还有 Semi-supervised Learning, Reinforcement Learning，但我都不了解。 

## Feasibility of Learning

现在我们知道了什么是机器学习，以及机器学习的目的就是获得一个尽可能接近目标函数的 Hypothesis，那么一个很自然的问题就是，我们只能知道一部分数据，目标函数是未知的，如何判断我们的 Hypothesis 是否接近目标函数？

答案是：大数定律。大数定律有一坨不等式，这里我们使用的是 Hoeffding 不等式：

$$
\mathbb{P}[\vert \nu - \mu \vert \gt \epsilon] \leq 2e^{-2\epsilon^2N} 
$$

式中 $\nu$ 是样本中某一类型的出现的频率，$\mu$ 则是该类型出现的概率，$\epsilon$ 是我们设定的错误阈值，$N$是样本容量。那么这个和机器学习有什么关系？

首先我们要对之前的 Learning Diagram 加上一部分：

{: .center}
![Learning Diagram - Updated]({{ site.baseurl }}/images/learning-diagram-2.png)

我们的样本 $\mathbf{X}$ 是从一个概率分布中产生的，但是我们不需要知道这个分布是什么。所以这是一个非常宽松、合理的假设，不会影响结论的一般性。

现在我们考虑一个类比实验：

- 我们有一个罐子，里面装了许多石头；
- 石头有红色的，有绿色的；
- 我们从里面挑出了 N 个石头，计算出了红色石头出现的频率 $\nu$；
- 根据 Hoeffding 不等式，我们知道 $\nu$ = $\mu$ 是 P.A.C.（Probably Approximately Correct） 的。

放到机器学习的情境下：

- 罐子里的石头就是我们数据的总体，里面石头的分布就是数据的概率分布；
- 我们有一个 Hypothesis，如果它把一条数据分对了的话，这条数据就标成绿色的，错了的话就标成红的，这样对于所有的数据，我们都可以给它们涂色，但是我们并不知道罐子里的颜色的情况；
- 我们从里面选出了 N 个数据，作为我们的一个样本，样本中两种颜色的数量我们可以知道；
- 根据 Hoeffding 不等式，我们的 Hypothesis 在这个样本中的错误率（红色数据的频率）等于其在总体中的错误率是 P.A.C.的。

{: .center}
![Marble Bin Experiment]({{ site.baseurl }}/images/bin.png)

也就是说，如果我们有一个 Hypothesis，它在我们的样本中表现很好，那么我们就可以信任它，认为它是接近真实的目标函数的。

但是，上述方法只能用来做检验，如果我们想从 Hypothesis Set 里选出一个好的 Hypothesis，可以使用同样的思路吗？

从 Hypothesis Set 中选一个出来，可以类比为我们有多个完全一样的罐子（里面的石头也是完全一样的），每个罐子分一个我们 Hypothesis Set 里的 Hypothesis，对里面的石头涂色，并选出一个样本:

{: .center}
![Multiple Marble Bins Experiment]({{ site.baseurl }}/images/multibin.png)

虽然对于每一个罐子来说，我们选出的样本中红色石头出现的频率 $E_{in}$ 与罐子里所有的红石头的频率 $E_{out}$ （或者可以说是概率，因为罐子里是石头的总体）相差很大是一个小概率事件（样本量足够的情况下），但如果我们有好多好多 Hypothesis，“其中至少一个 $E_{in}$ 与 $E_{out}$ 差别很大”就不再是一个小概率事件了，如果我们的算法正好选到了这样一个 Hypothesis，可以说我们的机器学习就失败了。

不过，虽原始的 Hoeffding 不等式不适应于多个 Hypothesis 的情况，但我们可以对其进行简单的推广：

$$
\mathbb{P}[\vert E_{in}(g) - E_{out}(g) \vert \gt \epsilon] \leq 2Me^{-2\epsilon^2N}
$$

具体的步骤我就不写了，一个简单的 Upper Bound，而且式子太长渲染出来不太好看。式中新增加的 $M$ 为 Hypothesis Set 中包含的 Hypothesis 的数量。

现在我们又有 Hoeffding 不等式了，但是这依然不能保证我们的机器学习会成功。原因就在于新增加的 $M$，对于绝大多数 Hypothesis Set 来说，$M$ 都是 $\infty$（包括我们之前说的 Perceptron，在二维的条件下，其 Hypothesis Set 是整个平面）。所以上式在大多数情况下都没意义。那么机器学习到底可不可行？这本书用了大概一章来讲这个问题……所以剩下的分析会在之后的笔记中。

## Outro

第一篇笔记讨论了什么是机器学习，以及分析了一半机器学习的可行性。下一篇会换一个话题，介绍一下线性模型和噪声的概念。

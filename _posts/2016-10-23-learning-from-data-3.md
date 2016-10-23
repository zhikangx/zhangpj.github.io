---
layout: post
title: Learning From Data Pt. 3
date: 2016-10-23 00:50
categories: ML
enable_mermaid:
summary: 看到 ‘punchline’ 这个词忽然想起我 Remember 11 还没有通关。
---

## Intro

我们在第一章讨论的机器学习的可行性这个问题，当时我们得到了这样一个不等式：

$$
\mathbb{P}[\vert E_{in}(g) - E_{out}(g) \vert \gt \epsilon] \leq 2Me^{-2\epsilon^2N}
$$

但由于对大多数 Hypothesis Set 而言，Hypothesis 的数量 $M$ 都是无限大的，所以上面这个不等式并不能保证机器学习一定能成功。在这一章，我们将用一个新的量来代替式中的 $M$，为 $E_{in}$ 和 $E_{out}$ 之间的差距找到一个有意义的上界，从而证明机器学习是可行的。

## Learning and Testing

首先，我们用一个简单的例子来回顾一下第一篇最后的内容：如果一个学生在期末考试中取得了不错的成绩，那么我们可以认为其较好地掌握了课程内容，也就是：

$$
\mathbb{P}[\vert E_{in}(g) - E_{out}(g) \vert \gt \epsilon] \leq 2 e^{-2\epsilon^2N}
$$

但如果我们给了其 $M$ 份试卷，然后这个学生给了我们其中一份得分最高的，这时候我们就不能肯定其完全掌握了课程内容，有可能这份得分高只是因为这份试卷中的问题恰好是他已经掌握的问题，因为他做了大量的试卷，所以这种情况的可能性也提高了，也就是：

$$
\mathbb{P}[\vert E_{in}(g) - E_{out}(g) \vert \gt \epsilon] \leq 2Me^{-2\epsilon^2N}
$$

现在，我们来考虑一下这个 $M$ 是如何产生的。对于一个 Hypothesis，我们将使得 $\vert E_{in}(h_m) - E_{out}(h_m)\vert \gt \epsilon$ 的事件记为 $\mathcal{B}_{m}$，所以对于一个有 M 个 Hypothesis 的集合，这种情况发生的概率就是：

$$
\mathbb{P}[\mathcal{B}_1 or \mathcal{B}_2 or ... or \mathcal{B}_M]\leq \mathbb{P}[\mathcal{B}_1] + \mathbb{P}[\mathcal{B}_2] + ... + \mathbb{P}[\mathcal{B}_M]
$$

在上面的式子中，我们的上界实际上是基于所有的 $\mathcal{B}$ 都是互斥的这样一个假设得到的，然而事实上这些事件通常有很大部分的“重叠”：

{: .center}
![Overlap]({{ site.baseurl }}/images/overlap.png)

如上图中，蓝线和绿线的差别只有黄色的那一小部分而已，所以“事件互斥”这样一个假设并不合理。

## Growth Function and Break Point

之前我们的分析中，区分不同 Hypothesis 的方法是从参数的角度来看的，也就是从所有数据的方面来区别不同的 Hypothesis，既然这样会出现较大的“重叠”，那么我们就用 Hypothesis 对我们手中的训练集的分类结果来区别不同的 Hypothesis 如何？

这种不同的分类结果可以称为 ‘Dichotomy’，不同的 Hypothesis，只要对我们的训练数据集有相同的分类结果，就属于同一个 ‘Dichotomy’。对于一个有 N 个数据点的数据集，Hypothesis 的数量可能是无限的，但 Dichotomy 的数量最多就是 $2^N$，所以如果能用 Dichotomy 的数量来代替 $M$ 的话，我们的 Bound 应该会好很多。

现在，我们来考虑一下这个 Dichotomy 的数量问题。首先定义增长函数 Growth Function。这个就是在任何 $N$ 个数据点中一个 Hypothesis Set 能够得到的 Dichotomy 的最大数量：

$$
m_{\mathcal{H}}(N) = \max_{\mathbf{x}_1, ..., \mathbf{x}_N \in \mathcal{X}}\vert\mathcal{H}(\mathbf{x}_1, ..., \mathbf{x}_N)\vert
$$

显然，增长函数必须满足：

$$
m_{\mathcal{H}}(N) \leq 2^N
$$

Growth Function 的概念稍微有点绕，我们举个例子来看一下。我们考虑一个 2D Perceptron 的 Growth Function。

{: .center}
![Growth Function of 2D Perceptron.]({{ site.baseurl }}/images/growthfunc-perceptron.png)

在 $N = 3$ 的时候，我们将数据点按照左图那种摆放方法是可以得到所有8个 Dichotomy 的，但这不代表 Perceptron 可以在任何 $N=3$ 的数据集中得到全部的 Dichotomy，如中间这幅图。但是对于 $N=4$ 的情况，无论怎么摆放数据点，我们都无法得到全部16个Dichotomy，最多就是14个，所以$m_{\mathcal{H}}(4) = 14$。这个 4 也被称作 Perceptron 的一个 ‘Break Point’。

Break Point 就是使得：

$$
m_{\mathcal{H}}(k) \lt 2^k
$$

的这个 $k$。如果 $k$ 是这个 Hypothesis Set 的Break Point 的话，任何比 $k$ 大的值也都会是 Break Point。

## VC Bound

我们再来回顾一下最初的那个 Bound。

$$
\mathbb{P}[\vert E_{in}(g) - E_{out}(g) \vert \gt \epsilon] \leq 2Me^{-2\epsilon^2N}
$$

现在，我们想用 Growth Function $m_{\mathcal{H}}(N)$ 来代替式中的 $M$，但是：

$$
m_{\mathcal{H}}(N) \leq 2^N
$$

也是指数级，如果直接把 $m_{\mathcal{H}}(N)$ 也没有什么太大的作用。所以，我们得先给它找一个新的 Upper Bound。首先给出结论：只要一个 $\mathcal{H}$ 有 Break Point，那么其 $m_{\mathcal{H}}(N)$ 就是 Polynomial 的。这样，我们就可以为我们的 Generalization Error 找到一个有意义的上界了。

接下来是证明。首先定义一个新的量 $B(N, k)$：在 Break Point 是 $k$ 条件下，我们在 $N$ 个数据点上能够得到的 Dichotomy 数量的最大值。这是一个与 $\mathcal{H}$ 无关的值，也是所有 Break Point 是 $k$ 的 $\mathcal{H}$ 所可能做到的最好结果（另外如果 $m_{\mathcal{H}(N)} \gt B(N, k)$ 的话，也就表示 $k$ 不是 $\mathcal{H}$ 的 Break Point），我们只要证明这个值是 Polynomial，就可以证明所有有 Break Point 的 $\mathcal{H}$ 的 Growth Function 都是 Polynomial。

我们的证明是采用数学归纳法。首先得给 $B(N, k)$ 找到一个递归边界。

{: .center}
![Recursive Bound 1]({{ site.baseurl }}/images/recursive-bound-1.png)

首先我们把我们能够获得的 Dichotomy 分成3类：前 $N-1$ 个点不成对的，成对且最后一个点都是$+1$的和最后一个点都是$-1$的。也就是上图中的 $S_1, S_2^+, S_2^-$。这样，我们就有：

$$
B(N, k) = \alpha + 2\beta
$$

首先看上边两类中的前 $N-1$ 个点。

{: .center}
![Recursive Bound 2]({{ site.baseurl }}/images/recursive-bound-1.png)

这里的 Dichotomy 都是不重复的，所以根据定义：

$$
\alpha + \beta \leq B(N-1, k)
$$

再来看下面两部分：

{: .center}
![Recursive Bound 3]({{ site.baseurl }}/images/recursive-bound-3.png)

这里，所有的 Dichotoy 都是成对的，所以：

$$
\beta \leq B(N-1, k-1)
$$

如果 $\beta \gt B(N-1, k-1)$ 的话，也就是 $\beta = 2^{k-1}$，这样，我们把最后一列加上，就能得到 $2^k$ 个不同的 Dichotomy 了，这就与 Break Point 是 $k$ 矛盾了。

结合以上几个式子，我们就得到了一个 Recursive Bound：

$$
\begin{align}
B(N, k) &= (\alpha + \beta) + \beta \newline
        &\leq B(N-1, k) + B(N-1, k-1)
\end{align}
$$

有了这个递归边界，我们就能用数学归纳法了。首先给出我们希望得到的结论：

$$
B(N, k) \leq \sum_{i = 0}^{k - 1}{n \choose i}
$$

边界条件很简单，选几个小的数算一下就可以了，归纳的步骤如下：

$$
\begin{align}
B(N, k) &\leq B(N-1, k) + B(N-1, k-1) \newline
&\leq \sum_{i = 0}^{k-1}{N-1 \choose i} + \sum_{i = 1}^{k-1}{N-1 \choose i-1} \newline
&= 1 + \sum_{i = 1}^{k-1}{N-1 \choose i} + \sum_{i=1}^{k-1}{N-1 \choose i-1} \newline
&= 1 + \sum_{i = 1}^{k - 1}\left[{N-1 \choose i} + {N-1 \choose i - 1}\right] \newline
&= 1 + \sum_{i = 1}^{k - 1}{N \choose i} = \sum_{i = 0}^{k - 1}{N \choose i}
\end{align}
$$

现在，我们证明了 $m_{\mathcal{H}}(N) \leq \sum{i = 0}^{k - 1}{N \choose i}$，接下来我们就想用这个 $m$ 来代替 $M$，从而获得一个更为合适的边界。

事实上：

$$
\mathbb{P}[\vert E_{in}(g) - E_{out}(g) \gt \epsilon \vert] \leq 4 m_{\mathcal{H}}(2N)e^{-\frac{1}{8}\epsilon{}^2N}
$$

这个就是 Vapnik-Chervonenkis 不等式，具体证明很长，有兴趣的可以找这篇文章来看...（别打我...书上也是把这个证明放到了附录里...）

## Outro

这一篇里我们主要解决了机器学习的可行性问题。下一篇我们将会介绍一个新的，在实用中常常见到的量，并从另外一个角度分析一下 Generalization Error 的问题。

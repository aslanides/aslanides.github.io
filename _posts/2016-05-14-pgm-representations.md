---
layout: post
title:  "Conditional independence and representations"
date:   2016-05-14 16:58:00 +1100
categories: machine_learning
excerpt: test
---
$$
	\newcommand{\CI}{\mathrel{\perp\mspace{-13mu}\perp}}
	\newcommand{\nCI}{\cancel{\CI}}
	\require{cancel}
$$

This post should be read as a prelude to my previous post about marginalization in simple graphical models. I talk very briefly about representing discrete probability distributions, and about conditional indepedence.

## Representations

An important and often overlooked detail in introductory discussions of probabilistic graphical models (PGMs) is the representations used. Recall that PGMs are factorized representations of joint distributions over a number of discrete random variables. Consider that we have $$n$$ discrete random variables $$X_1,\dots,X_n$$ with $$\Omega_{X_k} = \mathcal{X}_k$$ for each of the $$X_k$$. Now, mathematically, the joint distribution over $$(x_1,\dots,x_n)$$ is just a function $$p\ :\ \mathcal{X}_1\times\dots\times\mathcal{X}_N \to [0,1]$$. To represent this function, we  need a table of values $$P\in [0,1]^{\lvert\mathcal{X}_1\rvert\times\dots\times\lvert\mathcal{X}_n\rvert}$$. In fact, when we think about conditioning and marginalization, it often makes sense to think of $$P$$ as a matrix.

For example, consider the matrix $$P_{ab}$$ representing a joint distribution over two random variables $$A$$ and $$B$$ [1], and assume that $$A$$ is independent of $$B$$, so that the joint factorizes as the product of marginals

$$
p(a,b)=p(a)p(b)\ \forall a\in\mathcal{\Omega_A},b\in\mathcal{\Omega_B}.
$$

In terms of our representation, this corresponds to the vector outer product $$P_{ab}=Q_aR_b$$, where $$Q$$ and $$R$$ are the vectors representing the respective marginal distributions. Using the shorthand $$p(x_i)\equiv p(X=x_i)$$:

$$
\begin{aligned}
\underbrace{\begin{pmatrix}
p(a_1,b_1) & \dots & p(a_n,b_1)\\
\vdots & \ddots & \vdots\\
p(a_1,b_n) & \dots & p(a_n,b_n)
\end{pmatrix}}_{P}
\quad
&=
\quad
\underbrace{\begin{pmatrix}
p(a_1)\\
\dots\\
p(a_n)
\end{pmatrix}}_{Q}

\underbrace{\begin{pmatrix}
p(b_1) & \dots & p(b_n)
\end{pmatrix}}_{R}
\end{aligned}.
$$

Recall that the sum rule of probability corresponds to marginalization, and the product rule corresponds to conditioning. We can represent both of these fundamental operations in the matrix representation. Conditioning corresponds to element-wise multiplication:

$$
p(a,b) = p(a\lvert b)p(b) \iff P_{ab} = C_{ab}R_b,
$$

where $$C\in[0,1]^{\Omega_A\times\Omega_B}$$ is the matrix representation of the conditional distribution $$p(a\lvert b)$$. Marginalization corresponds to matrix multiplication:

$$
p(a) = \sum_{b\in\Omega_b}p(a,b) \iff Q_a = P_{a}^{b}\mathbb{1}_b,
$$

where $$\mathbb{1}$$ is the vector of ones.

## Conditional independence

The well-known graphical rules for reasoning about conditional independence are explained in Bishop [2]. I thought I'd riff a bit more about the consequences of the asymmetry in Bayesian networks, and the implications of the directed nature of the graph. Consider the familiar situation with three random variables $$A$$, $$B$$ and $$C$$ and the two graphical models below, which differ only by the direction of the edges:

![](/figures/pgms.png)
*Two graphical models representing different conditional independences. **(1)**: $$p(a,b,c) = p(c|a,b)p(a)p(b)$$. **(2)**: $$p(a,b,c) = p(a,b|c)p(c)$$.*

It can easily be shown two different factorizations induce different conditional independence relationships:

$$
(1)\quad A\nCI B|C, \quad A\CI B \\
(2)\quad A\CI B|C, \quad A\nCI B.
$$

Of the two cases, the first is the more interesting. The [standard](http://lesswrong.com/lw/ev3/causal_diagrams_and_causal_models/){:target="\_blank"}, common-sense example of this is the case where $$A,B,C\in\{0,1\}$$ are boolean variables that represent the events

$$
\begin{aligned}
A &\to \text{"The sprinkler was on"}\\
B &\to \text{"It rained"}\\
C &\to \text{"The grass is wet"}.
\end{aligned}
$$

Without observing the state of the grass (variable $$C$$), $$A$$ and $$B$$ are _a priori_ independent variables, but once we condition on (observe) a value for $$C$$, we lose this conditional independence. Now, learning about $$B$$ tells me about the state of $$A$$. The intuition in our example goes like this: after initially observing that the grass is wet ($$C=1$$), if I later discover that it had in fact rained earlier ($$B=1$$), then that lowers my credence in the sprinkler having been on. This clearly shows a lack of independence when conditioned on $$C$$, since $$A$$ and $$B$$ are now correlated. This intuition is fine, but let's _show_ it properly:

_Note_: After two hours of this, I'm stumped. I thought I'd be able to show this easily with the two weak assumptions $$p(C=1\lvert A=1)$ > \frac{1}{2}$$ and $$p(C=1\lvert B=1)$ > \frac{1}{2}$$, and maybe relying on the rather stronger assumption $$P(b)=p(a)$$. I've been bashing the sum and product rules with no success. What do?

[1] For convenience and clarity we use the Einstein notation explained in my previous blog post.

[2] C. M. Bishop. _Pattern Recognition and Machine Learning_. Springer, 2006

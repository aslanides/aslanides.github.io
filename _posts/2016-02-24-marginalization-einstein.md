---
layout: post
title:  "Marginalization tricks (WIP)"
date:   2016-02-24 17:37:00 +1100
categories: machine_learning
excerpt: In this post we explore some tricks for convenient marginalization in directed acyclic graphs (DAGs) using numpy's einsum API.
---

I discovered this useful trick while I was recently working on an assignment question for Christfried Webers' [excellent Introduction to Statistical Machine Learning course](https://sml.forge.nicta.com.au/isml15/). The idea is to simplify implementations of the [belief propagation](https://en.wikipedia.org/wiki/Belief_propagation) algorithm on acyclic factor graphs.

# Background: Einstein notation

The [Einstein summation notation](https://en.wikipedia.org/wiki/Einstein_notation) is a really convenient way to represent operations on multidimensional matrices. It's primarily used in differential geometry and in relativistic field theories in physics, but we'll be using it to do operations on discrete joint distributions. The rules are simple:

Expressions are in terms of the elements of objects. If an object is $$ N$$-dimensional, it must have $$ N$$ distinct indices.
If an index occurs both as a subscript and as a superscript in an expression, then it is summed over.
Following these rules, we can look at some simple examples. Matrix-vector multiplication looks like

$$

\begin{aligned}

\left[Ax\right]_{jk} &= \sum_{i}A_{ijk} x_i\\

&= A_{ijk} x^i

\end{aligned}

$$

Another example: compute the trace by contracting with the Kronecker delta:

$$

\begin{aligned}

\text{tr}\{A\} &=\delta^{i}_{j} A_{i}^{j}\\

&=A_{i}^{i}.

\end{aligned}

$$

Note that the dual vectors  $$ V^{i}$$ and $$ V_{i}$$ are related by the [metric](https://en.wikipedia.org/wiki/Metric_tensor) tensor: $$ V^{i} = g^{ij}V_j$$. In our case, the metric is the identity, so effectively we draw no distinction between them. The permissible operations with Einstein indices (Also known as Ricci calculus) generalise matrix algebra. For example, if we wanted to multiply two matrices $$ A$$ and $$ B$$, we require that the dimensions match up, i.e. if $$ A\in\mathbb{R}^{M\times N}$$ and $$ B \in\mathbb{R}^{P\times N}$$, then $$ AB^T$$ is well-defined but $$ AB$$ is not. We obviate these issues by letting the indices tell us which dimensions to sum over:

$$ A_{ij}B^{i}_{k}.$$

Numpy provides a pretty simple-to-use implementation of Einstein sums. Simply specify what indices to sum over. Brute-force marginalisation of a small joint distribution in two variables:

$$

\begin{aligned}

p\left(x\right) &= \sum_{y} p(x,y)\\

&= P\mathbb{1}\\

&= P_{ij}\mathbb{1}^{i},

\end{aligned}

$$

where $$ \mathbb{1}$$ is a vector of ones of the appropriate length. This procedure can work with higher dimensional matrices. Example:

```python
import numpy as np
A = np.random.rand(3,5,4,3,9,7,2) # make a random 7-dimensional array
A /= np.sum(A) # normalise it
np.einsum('abcdefg->d',A) # marginalise!

```
Note that repetition of indices does not necessarily imply a summation, if they're both super/subscripts. This lets us easily define element-wise products. For example:

$$ A_{ijk} = B_{ij} C_{ik}.$$

```python
B = np.random.rand(3,2)
C = np.random.rand(3,5)
A = np.einsum('ij,ik->ijk',B,C)
# A has shape (3,2,5)
```
As long as the shapes match up, then we have combined two rank-2 tensors to make a rank 3 tensor without contracting, i.e. without summation. This is key when we want to compute the product of messages in belief propagation. More tips and tricks can be found at the [einsum documentation page](http://docs.scipy.org/doc/numpy/reference/generated/numpy.einsum.html).

# Background: belief propagation

We have a graphical model of discrete variables $$ X=\left(x_1,\dots,x_N\right)$$ that induces some factorisation of the joint distribution $$ p(X)=\prod_{s}f_s\left(X_s\right),$$ where the factors $$ f_s$$ are functions of the variable subsets $$ X_s \subset X$$. When it comes to marginalisation:

$$ \begin{aligned} p(x) &= \sum_{X\backslash x} p(X)\\ &= \sum_{X\backslash x} \prod_{s\in\text{ne}(x)} F_s\left(x,X_s\right)\\ &= \prod_{s\in\text{ne}(x)}\sum_{X_s}F_s\left(x,X_s\right)\\ &:= \prod_{s\in\text{ne}(x)} \mu_{f_s \to x}(x), \end{aligned} $$

where $$ F_s(x,X_s)$$ is the product of factors in the subtree annexed by $$ x$$, and we interpret the subtree marginals $$ \mu_{f_s\to x}(x) $$ as "messages", which satisfy the mutually recursive relations:

$$ \begin{aligned} \mu_{f_s\to x}(x) &= \sum_{x_1}\cdots\sum_{x_M}f_s\left(x,x_1,\dots,x_M\right)\prod_{m\in\text{ne}\left(f_s\right)\backslash x}\mu_{x_m \to f_s}\left(x_m\right)&&(1)\\ \mu_{x_m\to f_s}\left(x_m\right) &= \prod_{l\in\text{ne}\left(x_m\right)\backslash\left(f_s\right)}\mu_{f_l\to x_m}\left(x_m\right),&&(2) \end{aligned} $$

where for all leaf factors and nodes we set $$ \mu_{f_l \to x} = f_l(x)$$ and $$ \mu_{x_l \to f} = 1$$, respectively. Given a joint distribution $$ p$$ corresponding to a graphical model, we can efficiently marginalise by evaluating the messages in equations (1) and (2). The subject of this post is the details of the implementation of these messages and their evaluation. Since we are dealing with discrete variables each factor $$ f_s\left(x_1,\dots,x_M\right)$$ is represented by a $$ K_1 \times \dots K_M$$ array, where $$ K_m$$ is the size of the domain of $$ x_m$$.

# Putting it together

Note that each of the messages $$ \mu(x_m)$$ is a marginal distribution with respect to $$ x_m$$, and so is a vector of size $$ K_m$$. The multiplication in equations $$ (1)$$ and $$ (2)$$ then corresponds to a bunch of elementwise matrix-vector products. This makes them amenable to the Einstein treatment. Rewriting equation $$ (1)$$ with the Einstein notation, we have

$$ \left[\mu_{f \to x}\right]_{j} = f_{j,i_1,\dots,i_M}\left[\mu_{x_1 \to f}\right]^{i_1}\dots\left[\mu_{x_M \to f}\right]^{i_M}.$$

Similarly for equation $$ (2)$$:

$$ \left[\mu_{x \to f}\right]_{i} = \left[\mu_{f_1 \to x}\right]_{i} \dots \left[\mu_{f_L \to x}\right]_{i}.$$

Demo can be found on my [GitHub](https://github.com/aslanides/DAGInference).

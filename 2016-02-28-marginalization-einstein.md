---
layout: post
title:  "Marginalization tricks (WIP)"
date:   2016-02-28 19:58:00 +1100
categories: machine_learning
excerpt: In this post we explore a convenient trick for marginalizing discrete distributions in directed acyclic graphs using numpy's einsum API.
---

I discovered this useful trick while I was recently working on an assignment question for Christfried Webers' [excellent Introduction to Statistical Machine Learning course](https://sml.forge.nicta.com.au/isml15/). The idea is to simplify implementations of the [belief propagation](https://en.wikipedia.org/wiki/Belief_propagation) algorithm on acyclic factor graphs.

# Einstein notation

The [Einstein summation notation](https://en.wikipedia.org/wiki/Einstein_notation) is a really convenient way to represent operations on multidimensional matrices. It's primarily used when manipulating tensors in differential geometry and in relativistic field theories in physics, but we'll be using it to do operations on discrete joint distributions, which are basically big normalized matrices. The distinction between a tensor and a matrix is that the [tensor](https://en.wikipedia.org/wiki/Tensor) has to behave in a certain way under coordinate transformations; for our purposes, this (physically motivated) constraint is lifted.

- All expressions are in terms of the elements of multidimensional objects. If an object is $$ K$$-dimensional, it must have $$K$$ distinct indices. For example, if $$\boldsymbol{A}$$ is an $$N \times N$$ matrix of real numbers, it is a $$2$$-dimensional object (which we will call rank-2 from now on), and we write it down as $$A^{ij}$$.

- If an index occurs both as a subscript and as a superscript in an expression, then it is summed over.

Following these rules, we can look at some simple examples. The inner product of two vectors $$\boldsymbol{u},\boldsymbol{v}\in\mathbb{R}^N$$ is given by the shorthand

$$
u^iv_i=\sum_{i}^{N}u_iv_i.
$$

This kind of operation is called a _contraction_, since the result has lower rank than its inputs (in the case of the inner product, these are 0 and 1, respectively). Notice that we can construct objects of higher rank out of lower rank ones quite easily. The outer product

$$
A^{i}_{j}=u^iv_j
$$

is an $$N\times N$$ matrix whose $$(i,j)^{\text{th}}$$ element is given by the product $$u_iv_j$$. Note that the dual vectors  $$ v^{i}$$ and $$ v_{i}$$ are in general related by the [metric](https://en.wikipedia.org/wiki/Metric_tensor) tensor: $$ v^{i} = g^{ij}v_j$$. In our case, the metric is the identity, so they are element-wise equal, and the only distinction is that one is a column vector and the other is a row vector. Hence, in this context, the matrices $$A_{ij}$$ and $$A^{i}_{j}$$ are equal, since to get from one to the other we left-multiply by the identity.

Matrix-vector multiplication looks like

$$

\begin{aligned}

\left[Ax\right]_{jk} &= \sum_{i}A_{ijk} x_i\\

&= A_{ijk} x^i.

\end{aligned}

$$

Here's where the power of the notation starts to become apparent: note that in the above example, $$\boldsymbol{A}\in\mathbb{R}^{L,M,N}$$ is a rank-3 matrix. By summing over the $$i$$ index, we are rotating $$\boldsymbol{A}$$ in such a way that it is being multiplied with $$x$$ in the first dimension. To illustrate this further, consider the multiplication of two matrices $$A\in\mathbb{R}^{M\times N}$$ and $$B \in\mathbb{R}^{P\times N}$$; clearly the product $$AB^T$$ is well-defined but $$AB$$ is not. With the Einstein notation, the indices tell us explicitly which dimensions to sum over. Hence

$$A_{ji}B^{i}_{k}$$

is well defined, since it sums over the $$\mathbb{R}^N$$ 'slot', whereas

$$ A_{ij}B^{i}_{k}.$$

is not well-defined. Note that flipping the order of two indices amounts to taking the transpose. As we will see, this feature is really helpful when it comes to marginalizing a multidimensional distribution.

Some final examples: we can compute the trace of a matrix by contracting with the Kronecker delta:

$$

\begin{aligned}

\text{tr}\{A\} &=\delta^{i}_{j} A_{i}^{j}\\

&=A_{i}^{i}.

\end{aligned}

$$

Given a vector $$\boldsymbol{f}$$ whose entries are functions on some basis set $$\mathbb{x}$$, we can write down the Jacobian simply as

$$
J^i_j = \partial_j\ f^i,
$$

where we identify $$\partial_j \equiv \frac{\partial}{\partial_{x_j}}$$. Bonus: if you're a statistician or a computer scientist, you now have all the tools you need to parse [quantum field-theoretic Lagrangians](https://en.wikipedia.org/wiki/Standard_Model):

$$
\mathcal{L}_{\text{EW}} =
\sum_\psi\bar\psi\gamma^\mu
\left(\imath\partial_\mu-g^\prime{1\over2}Y_\mathrm{W}B_\mu-g{1\over2}\vec\tau_\mathrm{L}\vec W_\mu\right)\psi.
$$

Ok, back to the task at hand. I hope I've convinced you that the set of permissible operations with with these rules (more formally known as the [Ricci calculus](https://en.wikipedia.org/wiki/Ricci_calculus)) generalize matrix algebra. Let's see how we can use these to make marginalization cleaner when doing belief propagation.

# Discrete distributions and marginalization in numpy

Before we get to belief propagation, let's talk about the standard set-up: we have a discrete multi-dimensional probability mass function $$p$$ over a bunch of random variables $$X_1,\dots,X_K$$, where each of the $$X_k$$ has its own finite sample space $$\Omega_k$$, and in general $$\Omega_i \neq \Omega_j$$ if $$i\neq j$$. For example, we could have $$\Omega_1=\{0,1,2,3\},\ \Omega_2=\{0,\dots,255\},\ \Omega_3=\{"blue","green","red"\}$$, etc.

The two most basic operations on $$p$$ are conditioning and marginalization. To marginalize, we wish to compute, for example,

$$
p(x_1)=\sum_{x_2}\cdots\sum_{x_K}p(x_1,\dots,x_K)
$$

In [NumPy](http://www.numpy.org), we represent the probability mass function $$p$$ as a $$\Omega_1 \times \Omega_2 \times \dots \times \Omega_K$$ matrix $$P$$ of real numbers in the interval $$[0,1]$$. To compute the sums above 'by brute force', we would sum over all the dimensions except the first. Similar to Matlab, this operation is most efficient in NumPy if it is vectorized. The best way to accomplish a simple sum like this is with `numpy.sum(...)`, specifying the dimensions to sum over. We can also do this with `numpy.einsum(...)`:

```python
import numpy as np
A = np.random.rand(3,5,4,3,9,7,2) # make a random 7-dimensional array
A /= np.sum(A) # normalize
np.einsum('abcdefg->d',A) # marginalize :)
```
Note that repetition of indices does not necessarily imply a summation if they're both super/subscripts. This lets us easily define element-wise products. For example:

$$ A_{ijk} = B_{ij} C_{ik}.$$

```python
B = np.random.rand(3,2)
C = np.random.rand(3,5)
A = np.einsum('ij,ik->ijk',B,C)
# A has shape (3,2,5)
```
As long as the shapes match up, then we have combined two rank-2 tensors to make a rank 3 tensor without contracting, i.e. without summation. This is key when we want to compute the product of messages in belief propagation. More tips and tricks can be found at the [einsum documentation page](http://docs.scipy.org/doc/numpy/reference/generated/numpy.einsum.html).

# Belief propagation

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

A short demo on a simple Bayesian network can be found [here](https://github.com/aslanides/DAGInference).

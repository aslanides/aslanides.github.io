---
layout: post
title:  "Linear regression yooo"
date:   2016-02-24 13:37:00 +1100
categories: machine_learning
---

Baby steps guys, so let's make the first blog post about something simple. Here's my take on linear regression :). We start with the standard (frequentist) maximum likelihood approach, then show the correspondence between isotropic Gaussian priors and $$l_2$$ regularizers, before hinting at the full Bayesian treatment. All of these results are derived and explored in more detail in Bishop (2006).

# Setup

So in the standard regression task we are given labelled data

$$\mathcal{D} \stackrel{\cdot}{=} \left\{\left(x_{i},y_{i}\right)\right\}_{i=1}^{M},$$

with each $$ x_{i}\in\mathbb{R}^K$$ drawn i.i.d. from some unknown process $$ p(x)$$, and $$ y_i\in \mathbb{R},$$

and our objective is to learn a linear mapping $$ f:\mathbb{R}^K\to\mathbb{R} $$

$$f(x,w;\phi)=w^T\phi(x)$$

where the basis functions 
$$\phi : \mathbb{R}^K \to \mathbb{R}^N$$
are user-defined and fixed, and we place no restriction on the relative sizes of $$ K $$ and $$ N$$. Now, our data will be noisy, so

$$y(x)=f(x)+\epsilon \qquad \epsilon\sim\mathcal{N}(0,\beta^{-1}),$$

and so the distribution of the targets conditioned on the inputs is

$$p(y|x) = \mathcal{N}\left(w^T\phi(x),\beta^{-1}\right)$$

# Max Likelihood Solution

Given that our dataset is drawn i.i.d., the likelihood is given by

$$
	\begin{aligned}

	p(\mathcal{D}|w) &= \prod_{i=1}^{N}p(y_i|x_i,w)p(x_i)\\

		&\propto \prod_{i=1}^{N}p(y_i|x_i)\\

		&= \prod_{i=1}^{N}\mathcal{N}\left(w^T\phi(x_i),\beta^{-1}\right)\\

	\end{aligned}
$$

Our objective now is to learn the weights $$ w$$ that maximise this likelihood, taking the variance of the noise $$ \beta^{-1}$$ to be fixed.That is, we wish to find

$$
\hat{w}_{\text{ML}} = \arg\max_{w}p(\mathcal{D}|w)
$$

Now, $$ \log(\cdot) $$ is a monotonically increasing function, so

$$
\begin{aligned}

\hat{w}_{\text{ML}}&=\arg\max_{w}\log p(\mathcal{D}|w)\\

&=\arg\max_{w}\log\prod_{i=1}^{N}\mathcal{N}\left(w^T\phi(x_i),\beta^{-1}\right)\\

&=\arg\max_{w}\sum_{i=1}^{N}\log\mathcal{N}\left(w^T\phi(x_i),\beta^{-1}\right)\\

&=\arg\max_{w}\sum_{i=1}^{N}\log\left(\frac{1}{\sqrt{2\pi\beta^{-1}}}\exp\left\{-\frac{\left(y_i-w^T\phi(x_i)\right)^2}{2\beta^{-1}}\right\}\right)\\

&=\arg\min_{w}\sum_{i=1}^{N}\left(y_i-w^T\phi(x_i)\right)^2,\\

\end{aligned}
$$

which shows the correspondence between maximum likelihood under Gaussian noise with minimizing the square error.

# Bayes v0.1

Now, maximum likelihood is notorious for overfitting; we typically introduce an $$ l_2$$ regularizer. Interestingly, a certain class of prior over $$ w$$ effectively induces this regularizer, which gives a stronger intuition for introducing this term into the objective function. Consider the isotropic Gaussian prior

$$
p(w) = \mathcal{N}\left(0,\alpha^{-1}I_{N}\right)
$$

If we now do the 'Bayes v0.1' thing and get a maximum a posteriori (MAP) point estimate of $$ w $$ using this prior, we see that

$$
\begin{aligned}

\hat{w}_{\text{MAP}}&=\arg\max_{w}p(w|\mathcal{D})\\

&=\arg\max_{w}\frac{p(\mathcal{D}|w)p(w)}{p(\mathcal{D})}\\

&=\arg\max_{w}p(\mathcal{D}|w)p(w)\\

&=\arg\max_{w}\log\left(p(\mathcal{D}|w)p(w)\right)\\

&=\arg\max_{w}\left(\log p(\mathcal{D}|w) + \log p(w)\right)\\

&=\arg\min_{w}\left(\sum_{i=1}^{N}\left(y_i-w^T\phi(x_i)|\right)^2+C\alpha^{-1}\|w\|^2\right)\\

\end{aligned}
$$

# Bayes v1.0

With MAP inference we start with a relatively ignorant prior (it is not entropy-maximizing, and certainly not Solomonoff) and end up with a point estimate. This clearly is not useful if we want to use a sequential updating scheme, in which we use the posterior as the prior for the subsequent iteration. We can maintain a distribution over the weights, representing our subjective beliefs. Or, we can use the predictive distribution, representing our uncertainty in the value of $$ y$$, given some $$ x$$ and a bunch of experience $$ \mathcal{D}$$.

$$
p(y|x,\mathcal{D}) = \int_\mathcal{W}\text{d}wp(y|w,x,\mathcal{D})p(w)
$$

# Bayes v2.0

Of course, the central issue with the Bayesian scheme (neglecting the computational/analytic difficulties arising from maintaining and updating a large distribution) is choosing the prior in a sensible and principled way. Note that the prior we chose for our MAP estimator was essentially chosen for convenience; though it is simple and thus reasonable a la Ockham, it is still kind arbitrary. Enter Solomonoff's universal prior, which I'll discuss more in a later post.

$$
M(x) = \sum_{q:U(q)=x*}2^{-l(q)}
$$

# Scientific Importance

...
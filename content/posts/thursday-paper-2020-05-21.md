---
title: "Thursday Paper: Your Classifier Is Secretly An Energy Model"
date: 2020-05-21T09:08:17+01:00
---

I'm going to start a regularly series where I try to read a couple of academic papers per week. The plan will be that on Tuesdays I read classic computer science papers (of which I have read embarrassingly few) and on Thursdays I will read more recent machine learning papers (of which I have also read embarrassingly few). 

My Tuesday paper will lean towards distrbuted systems and databases, probably with a lot drawn from [this list](https://dancres.github.io/Pages/) initially.

My Thursday paper will mostly be directed by what is being covered at the [London ML Paper Club Meetup](https://www.meetup.com/ML-Paper-Club/events/khjgrrybchbcc/).

Due to time constraints, I'm only going to spend an hour on each paper. If this feels extremely rushed, I'll split the post over a couple of weeks.

## Introduction

Up today is a paper called [You Classifier Is Secretly An Energy Model And You Should Treat It Like One, Grathwohl et al.](https://arxiv.org/pdf/1912.03263.pdf).

The paper starts by lamenting the drift between generative models and the downstream applications that they originally motivated. Recall that a generative model is something which is aimed at, given the data, learning the true distribution of the data. You might not achieve that, but if you have a good approximation to that distribution, you have a way to benefit downstream applications like semi-supervised learning (I imagine, by generating new data), imputation of data, and calibration of uncertainty. However, the recent trend has apparently been towards direct measurement of how good the generative model itself is (e.g. log-likelihood on held-out validation sets, or even just qualitively assessing if generated data looks like it came from the same distribution as the original data).

So the paper is obviously going to be about generative models with a downstream application in mind. To quote, it's going to use something called _Energy Based Models_ to "help realize the potential of generative models on downstream discriminative problems".

The authors claim that using the energy based models along with the classifiers allows the resulting classifiers to perform better with respect to calibration, out-of-distribution detection, and adverserial robustness.

## What Is An Energy Model?

The motivation comes from writing a probability density function over $x \in \mathbb{R}^d$ as $p\_{\mathbf{\theta}}(\mathbf{x}) = \frac{\exp(-E\_{\mathbf{\theta}}(\mathbf{x}))}{Z\_{\mathbf{\theta}}}$. The denominator is just the normalizing constant $Z\_\mathbf{\theta} = \int\_{\mathbf{x}} \exp(-E\_{\mathbf{\theta}}(\mathbf{x}))$. The function $E\_{\mathbf{\theta}} : \mathbb{R}^d \to \mathbb{R}$ is called the _energy function_. The paper is extremely terse on why you might bother reparameterising like this. In the meetup, we discussed that learning with an energy function instead of the probability distribution directly is smoothing around the data manifold, so it is like inserting a prior.

Writing your model in this way has downsides, for example computing the denominator is intractible. This makes training difficult. To make some headway, you can express $\frac{\partial \log p\_\theta{x}}{\partial \theta}$ in terms of an expectation over the model distribution. You can therefore approximate this using MCMC techniques. More recent approachs use techniques based on something called Stochastic Gradient Langevin Dynamics (SGLD).

## What Does This Have To Do With Classifiers?

A classification problem with $K$ output classes normally involves a function $f:\mathbb{R}^d \to \mathbb{R}^K$ to produce _logits_, then an application of a softmax function $p\_(y|\mathbf{x}) \propto \exp(f\_{\theta}(\mathbf{x})[y])$. In this notation, $f\_{\theta}(\mathbf{x})[y]$ denotes the $y$th index of $f\_{\theta}(\mathbf{x})$.

We just converted the logits to probabilities for classes using the softmax function, but another thing we can do is to define an energy based model from these. More specifically, we define $p\_{\theta}(\mathbf{x}, y) \propto \exp(f\_{\theta}(\mathbf{x})[y])$, and $E\_{\theta}(\mathbf{x}, y) = -f\_{\theta}(\mathbf{x})[y]$. We can marginalise out $y$ to get a distribution over $\mathbf{x}$, which gives us an energy based model with $E\_{\theta}(\mathbf{x}) = -\log \sum\_{y} \exp(f\_{\theta}(\mathbf{x})[y])$.

As the authors put it, this as "a generative model hidden within every standard discriminative model". They call the output of this interpretation a Joint Energy based Model (JEM).

## How Can We Use This?

Having identified that there is a generative model within your standard discriminative model, you might want to try to train both of them together. There are a few ways that this can be done, but the authors say that the factorisation $\log p\_\theta(\mathbf{x}, y) = \log p\_\theta(\mathbf{x}) + \log p\_\theta(y|\mathbf{x})$, coming straight from the definitions, works best.  They train the classifier part with standard cross-entropy and the energy part with SGLD.

Going back to the introduction of the paper, we should expect that if we train in this way then we get more robust models. This is starting to look plausible.

## Results

They first compare a Wide Residual Network architecture trained using their method and compare it to classifiers and generators for the same datasets, and find it performs well as both.

Next, they talk about calibration. A classifier is calibrated if its predictive confidence in a class and its misclassification rate line up as you would hope. This is obviously a desirable feature for real-world applications. Apparently, in recent years classifiers have got better but their calibration has generally got worse. Here, the authors train the same architecture and compare EBM with non-EBM (and no Platt scaling): they find that accuracy is slightly worse but calibration is significantly better.

For out-of-distribution detection, you want to have a function $s\_\theta : \mathbb{R}^d \to \mathbb{R}$ which will give you a score for how likely the input came from your distribution. You would expect that the probability distribution itself would be a good function to use here, but somewhat surprisingly there are arguments against this: apparently there are tractible deep models that score poorly in this sense. They instead choose the maximum predicted probability $s\_\theta(\mathbf{x}) = \max\_y p\_\theta(y|\mathbf{x})$. Intuitively, if the classifier is unsure, then maybe that's because the input isn't something from the distribution it's modelling, so in this sense it's similar to calibration. Again, their method performs well on this score. (They also propose a new score for OODD, which I skipped).

For robustness, the authors start by discussing some similarities in the training between their approach and adversarial training, so you would hope that JEMs would have some robustness to adversarial examples. I'm going to skip over this part though due to lack of time and knowledge on adversarial attacks. 

## Rounding Off

They mention a number of challenges they encountered during training, with lack of feedback on progress and instability. There is quite a lot in the appendix, at least part of which covers this. They close off by saying that these difficulties could likely be overcome with researchers examining the training more, and they think there is value in incorporating the energy model in to the classifier.

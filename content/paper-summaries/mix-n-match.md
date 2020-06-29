---
title: "Mix-n-Match: Ensemble and Compositional Methods for Uncertainty Calibration in Deep Learning"
date: 2020-06-18T08:49:39+01:00
---

Another [paper](https://arxiv.org/pdf/2003.07329.pdf) about calibration today. We have the usual motivation for calibration: Deep learning models are making predictions in the real world, and we would like it if they were well-calibrated (in the sense that the probability they predict for a class matches aligns with the confidence the model has in that prediction). Off the shelf, many models don't have this property, but there are established _post-hoc_ methods for making a classifier calibrated (e.g., Platt scaling). This seems to involve holding out a _calibration set_ and learning a transformation on the model's predictions to a calibrated version of those. The authors reference a few papers for post-hoc calibration techniques; it seems to be a fairly active area. They also reference a few papers for training a well-calibrated model _ab initio_.

The authors define three desiderata for any uncertainty calibration method:

* Accuary-preserving: The calibration method should not degrade the classification accuracy of the original classifier
* Data-efficiency: We shouldn't need a huge amount of calibration data to make the model calibrated
* High expressive power: The ability to approximate the true calibration function given sufficient calibration data

They will argue that none of the existing techniques achieve all three of these. However, they will also argue that certain ensemble methods are able to do so.

Another contribution of this paper is to propose a new metric for measuring calibration performance. The reason they give is that the usual approach (expected calibration error) is data inefficient and unreliable. More details on that later.

## What Do We Want?

First up, a nice precise definition of calibration. Let $X \in \mathcal{X}$ be an input feature, $Y = (Y\_1,...,Y\_L) \in \mathcal{Y}$ be the one-hot encoded label, and let $f : \mathcal{X} \ni x \mapsto z \in \mathcal{Z} \subset \Delta^L$ be a function that outputs a prediction probability. (Here $\Delta^L$ is the $L$ dimensional probability simplex.) Write $\mathbb{P}(Z, Y)$ be the joint distribution of the prediction and the label. The canonical calibration function is $\pi(z) = (\mathbb{P}(Y\_1=1|f(X)=z), ..., \mathbb{P}(Y\_L=1|f(X)=z))$. We say $f$ is _perfectly calibrated_ if for any $x \in \mathcal{X}$ we have $z = \pi(z)$.

An example would probably help. Suppose we have some point $x$ that our model is certain should have the label $1$, so it outputs $f(x) = z = (1, 0, .... 0)$. Now we feed it through $\pi$, which is going to ask what are the actual class probabilities conditioned on the prediction. Well, it could the case that our model has a problem in that it confidently predicts label $1$ in situations where label $2$ is also a strong candiation, say $\pi(z) = (0.6, 0.4, 0, ..., 0)$. In this case our model would not be perfectly calibrated, because for this particular input it output an over-confident score.

This paper will focus on post-hoc calibration, which is to find a map $T : \Delta^L \to \Delta^L$ that adjusts the output of an existing classifer to make it better calibrated. There will be a held out set $\{(z^{(i)}, y^{(i)})\}$ of $n\_c$ samples to perform this post-hoc calibration training, and a set of $n\_e$ samples used to evaluate it.

Touching back on the intuition again: if you had a lot of data, you might be able to _learn_ that your model confidently predicts label $1$ in situations where label $2$ is also a strong candidate. This means you could learn the transformation $T$ that "undoes" this miscalibration.

## Existing Post-hoc Calibration Methods

They split these up in to two groups. First up there are parameteric approaches to finding a function $T$. This includes Platt scaling, which uses a logistic regression approach $T(z; a, b) = (1 + \exp(-az-b))^{-1}$, and finds the best $a, b$ by minimises the negative log-likelihood on the calibration set.

There are also non-parametric methods. There are a few approaches mentioned here. One is _histogram binning_, which sets up some histograms and uses these to estimate $\pi(z)$ (then uses what was estimated as the output of the post-hoc calibration).

They do some studies of performance with increasing calibration dataset size, measuring both expected calibration error (ECE) and accuracy. In terms of the desiderata, the impact of of applying a post-hoc calibration should show have the following effects on the metrics:

* An accuracy-preserving method would not have any impact on the accuracy at all
* A data-efficient method would show decent improvements in ECE even for small calibration datasets
* An expressive method would show continued improvements in ECE as the size of the calibration dataset increasing

What they find, generally speaking, is that parameteric methods are data-efficient but not expressive, where-as non-parametric methods are expressive but not data-efficient. They also have a graph of accuracy, in this temperature scaling (parametric) has no effect on accuracy, but isotonic regression (non-parametric) has a pretty significant drop in accuracy, which I'm surprised they don't make more of.

## Measurement Issues

We care about how far $z$ is from $\pi(z)$, so naturally we will look at some sort of error. There are various norms you can use here, but the most common seems to be $L\_2$, so we look at $\text{ECE}(f) = \int ||z-\pi(z)||\_2^2 p(z) dz$.

Of course this can't be computed from this definition in practice, as $z$ is a continuous random variable, so we need some estimator. A popular approach is to partition the data points in to $b$ bins based on the prediction $z$, and approximate with the sum $\overline{\text{ECE}}(f) = \frac{1}{n\_e} \sum\_{i=1}^b |B\_i| ||\overline{f}(B\_i) - \overline{\pi}(B\_i)||\_2^2$, where we take the average of $f$ and $\pi$ inside the bins.

There's a bias-variance trade-off in the bin selection. Too few bins, and you get an under-estimate of the ECE. Too many, and your estimate is noisy due to sparsely populated bins. There are also issues with how quickly this converges. For this reason the authors are going to propose another method for measuring calibration error, but I'm not going to look in to it too much due to lack of time.

## Designing New Calibration Methods

They first point out that one way to easily achieve accuracy-preservation is to restrict transformations to those that apply the same, normalised, order-preserving map to all the predictions. In particular, this explains why Platt/temperature scaling are accuracy-preserving.

To improve a parametric model (like Platt scaling), we would like to improve its expressivity. For this they propose using an ensemble for accuracy-preserving calibration maps. They suggest a very specific ensemble that they call Ensemble Temperature Scaling.

For non-parametric calibration (they focus on isotonic regression) there is no normalised order-preserving map kicking around so they have to learn one and force it in. Modifying in this way also makes it more data-efficient, but unsurprisingly costs in expressivity.

In the end, they stack a non-parametric model on top of a parametric one. Roughly, the parametric one is a baseline way to reduce the variance, and then the lingering data-efficiency problems for the non-parametric one are less of an issue from this baseline. If we make both parts accuracy-preserving, then clearly the whole thing will be.

## Results

I skipped over the section on measurement, but basically it's still estimating expected calibration error but just using a kernel density estimator instead. The results will be based on this method of calculating.

They do a bunch of experiments with different image problems. The first results they give focus on just the ECE improvements. It looks to me like ETS is the clear winner here.

The next set of graphs are about increasing dataset sizes, in which non-parametric things start to become more useful. Recall that in this case accuracy-preservation is learned, so we'd expect a fairly large dataset before we start seeing this bearing out and that is indeed the case. The non-parametric methods generally improve in the expressivity sense more than ETS does. There is a pretty significant difference for CIFAR-10 where the non-parametric methods are significantly better (actually everything plateaus after a pretty small amount of data), but for CIFAR-100 and ImageNet it takes a lot longer for the non-parametric approach to edge ahead of ETS.

Overall, I thought this was a great, precise paper. I wasn't totally sold on the value of expressivity to be honest though. Sure, it's a good thing to have, but looking in the result sections the gains are marginal and there is a clearly a cost. What I took away from this paper is that ensemble temperature scaling is a nice simple approach that doesn't cost much and gets you a long way.

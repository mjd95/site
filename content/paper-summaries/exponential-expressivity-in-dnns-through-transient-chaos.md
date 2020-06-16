---
title: "Exponential Expressivity in Deep Neural Networks Through Transient Chaos"
date: 2020-06-11T08:34:29+01:00
---

I was supposed to be reading [Deep Information Propagation, Schoenholz et. al.](https://arxiv.org/pdf/1611.01232.pdf) today, but I wasn't able to get through it without reading a precursor paper [Exponential Expressivity in Deep Neural Networks Through Transient Chaos, Poole et. al.](https://arxiv.org/pdf/1606.05340.pdf), so here's a summary of that one instead.

The starting point for this paper is the observation that DFNNs (1) seem to be able to express complex functions with few parameters (a linearly shallower network expressing the same function requires exponentially wider layers) and (2) seem to be able to "disentangle" highly curved manifolds. However, neither of these are actually theorems, they are just what people believe based on examples.

(Actually that's not fair. There has been prior work for some sort of complexity theory for the functions expressed by variously architected DFNNs. However, Poole et. al. argue that these have only been considered in limited scope, and they want something general (arbitrary nonlinearities, generic networks).)

The framework set up by Poole et. al. and picked up by Schoenholz et. al. considers a certain ensemble of random deep neural networks. Roughly, you will look at neural networks of a fixed shape (fixed depth, fixed layer width) and draw weights and biases for all the nodes from i.i.d. Gaussians. (Perhaps surprisingly, the activiation functions or loss functions chosen don't really seem to matter.) Within this framework, you can ask questions about the evolution of various signals as they propagate through the network.

## Mean Field Theory For Signal Propagation in DFNNs

So what exactly is this framework?

Suppose we fix a number of layers $L$, and sequence of widths $N\_l$, and an activation function $\phi : \mathbb{R} \to \mathbb{R}$. Choosing i.i.d. weights $W\_{i,j}^{l} \sim N(0, \sigma\_w^2/N\_l)$ and biases $b\_i^l \sim N(0, \sigma\_b^2)$ we get a neural network described by the pair of equations $z\_i^l = \sum\_j W\_{i,j}^l y\_j^l + b\_i^l$ and $y\_i^{l+1} = \phi(z\_i)$. (We write $y\_i^0 = x\_i$) for the input.) This is some sort of "random DFNN ensemble". What can we say about it?

In the limit of large layer widths ($N\_l \rightarrow \infty$) the authors say that "certain aspects of signal propogation take on an essentially deterministic character". I'm not sure what they mean here, but one obvious thing to note is that since $W\_{i,j}^{l} \sim N(0, \sigma\_w^2/N\_l)$ the weights are not very random at all in this limit.

## The Dynamics of Length Vectors

Define $q\_l = \frac{1}{N\_l} \sum\_{i=1}^{N\_l} (z\_i^l)^2$, the lengths of $x\_i$ as it journeys through the network. Or, if you prefer, the second moments of the empirical distribution across the various layers. Now each of these $z\_i^l$ are a sum of uncorrelated random variables, so as we increase $N\_l$ they are going to look like some mean zero Gaussian, which the $q\_l$ are the second moments of. You can write down an integral for how the Guassians evolve at each layer (basically the limit of the sum coming from the definition), which gives you an iterative definition $q\_l = \mathcal{V}(q^{l-1} | \sigma\_w, \sigma\_b) = \sigma\_w^2 \int \phi(\sqrt{q^{l-1}z})^2 d\mu(x) + \sigma\_b^2$, where $d\mu(z)$ is the Gaussian measure.

They study the dynamics of the map $\mathcal{V}$ in the case when the activation function is $\phi(h) = \text{tanh}(h)$. The exact dynamics obviously depends on $\sigma\_w$ and $\sigma\_b$. They highlight a few cases:

* $\sigma\_b=0$, $\sigma\_w < 1$ (bias-free, little weight-variance). Intuitively, you'd probably guess in this case that the map isn't doing much so has little chance of being chaotic. And indeed, it's obvious frmo the definition that there is a single fixed point ($q^\*=0$ and that all vectors lengths are shrinked to zero (in particular, the zero vector is a stable fixed point).
* $\sigma\_b=0$, $\sigma\_w > 1$ (bias-free, large weight-variance). Again from the definition, we see $q^\*=0$ is still a fixed point, but it is now unstable (the map expands vectors of short length). There is also a non-zero fixed point, which is stable (the map shrinks vectors of long length).
* $\sigma\_b \neq 0$. Here there is a single (non-zero) fixed point. It is stable, and actually the convergence to the fixed point is fast (often with only four iterations).

## The Dynamics of Covariance

In the paper this section talks about "transient chaos". Reading around, this is a property of dynamical systems in which certain regions of phase space behave chaotically for a while, before espacing to an external attractor. In this case the "transient" seems to just be referring to the fact that even deep networks have a very finite depth, so all dynamics are truncated.

We'll now consider two inputs $\mathbf{x}^{0,1}$ and $\mathbf{x}^{0,2}$ and how they evolve together by looking at the sequence of $2 \times 2$ matrices of inner products. We already know what's happening on the diagonal, so our focus is now on the crossterms. In much the same way as before, we can write this inner product as a convolution of the two previous measures, and get an iterative definition $q\_{12}^l = \mathcal{C}(q\_{12}^l, q\_{11}^l, q\_{22}^l|\sigma\_w, \sigma\_b)$, though it's a fair bit more complicated. They express their results in terms of the correlation coefficient $c\_{12}^l = q\_{12}^l (q\_{11}^l q\_{22}^l)^{-1/2}$.


There are again three qualitative regions, cut out by the derivative $\chi\_1 = \frac{\partial c\_{12}^l}{\partial c\_{12}^{l-1}}|\_{c=1}$. Fix some finite non-zero $\sigma\_b$.

* Small $\sigma\_w$. The only fixed point is $c^\*=1$, and it is stable. Any two inputs are moved closer together.
* Increasing $\sigma\_w$. As $\chi\_1$ climbs over one, the $c^\*=1$ point becomes unstable. Another, stable, fixed point appears. The authors describe this middle regime as being a competition between weights and non-linearities (which decorrelate inputs) and biases (which correlate inputs).
* Large $\sigma\_w$. Everything becomes maximally decorrelated, which results in there being a single stable fixed point at $c^\*=0$.

## The Dynamics of Manifolds

You can define a one-dimensional manifold $x^{0}(\theta)$ and think about how this evolves. You can get some sort of idea of what's going on by taking pairs of vectors from the manifold and studying the quantities of the previous sections.

Intuitively, you would expect that in a chaotic phase the manifold should become decorrelated/complex, whereas in the stable phase it should just shrink to a point. And that is indeed what their simulations show. With $\sigma\_b=0.3$ and $\sigma\_w=4.0$ a circle is evolved to something completely unrecognisable after 15 layers.

They go on to have a bit of fun talking about this in terms of basic Riemannian geometry, but I'm going to skip over this part.

## Shallow Networks Cannot Achieve Exponential Expressivity

So far we saw that length vectors display whatever behaviour they're going to behave with only a few iterations, but the convergence is a fair bit slower. This suggests that we really need to be talking about _deep_ networks to see the unfolding of complicated manifolds.

Alongside the positive (empirical) results they've presented for deep networks being able to unfold complicated manifolds, they present a negative (theoretical) result. Specifically, they consider a one-layer NN of width $N\_1$ and a one-dimensional curve in its input space, and they give an upper bound for its Euclidean length in terms of $N\_1$, the maximum number of sign changes of the derivative of the input curve along any axis, and a measure of the "dynamical range" of the non-linearity in the NN. The point is that it's linear in $N\_1$, so you're never going to get the "exponential expressivity" in this aspect.

## Discussion

There's one more section on what happens to classification boundaries, but I'm going to skip that due to lack of time. Overall the paper concludes with a kind of vague discussion of what was shown here and what potential tools from Riemannian geometry could bring to understanding learning dynamics of DNNs.

This paper didn't really say all that much to me but to be fair it was introducing concepts which are probably not that widely used in the deep learning community. Perhaps I should defer my judgement until I see some of the follow-up work that came out of this.

---
title: "Thursday Paper: Deep Information Propogation"
date: 2020-06-11T08:34:29+01:00
---

## Mean Field Theory For Signal Propagation in DFNNs

The paper is extremely terse on this topic so I'm going back to the Poole paper for more motivation and details.

That starts from the observation that people like DFNNs because (1) they seem to be able to express complex functions with few parameters (a linearly shallower network expressing the same function requires exponentially wider layers) and (2) they seem to be able to "disentangle" highly curved manifolds. But neither of these are actually theorems, they are just what people believe based on examples. Actually that's not fair, because there has been prior work for some sort of complexity theory for the functions expressed by variously architected DFNNS. However, the authors argue that these have only been consideredin limited scope, and they want something general (arbitrary nonlinearities, generic networks). 

They're going to use some tools from Riemannian geometry and mean field theory to quantify a network's expressivity. They'll quantitatively define how well something "disentangles a manifold" and show that DFNNs perform well with respect to their definition (what a surprise!). They will also pursue something that doesn't make sense.

Suppose we fix a number of layers $L$, and sequence of widths $N\_l$, and an activation function $\phi : \mathbb{R} \to \mathbb{R}$. Choosing i.i.d. weights $W\_{i,j}^{l} \sim N(0, \sigma\_w^2/N\_l)$ and biases $b\_i^l \sim N(0, \sigma\_b^2)$ we get a neural network described by the pair of equations $z\_i^l = \sum\_j W\_{i,j}^l y\_j^l + b\_i^l$ and $y\_i^{l+1} = \phi(z\_i)$. (We write $y\_i^0 = x\_i$) for the input.) This is some sort of "random DFNN ensemble". What can we say about it?

In the limit of large layer widths ($N\_l \rightarrow \infty$) the authors say that "certain aspects of signal propogation take on an essentially deterministic character". I'm not sure what they mean here, but one obvious thing to note is that since $W\_{i,j}^{l} \sim N(0, \sigma\_w^2/N\_l)$ the weights are not very random at all in this limit.

Define $q\_l = \frac{1}{N\_l} \sum\_{i=1}^{N\_l} (z\_i^l)^2$, the lengths of $x\_i$ as it journeys through the network. Alternatively, you can think of these as the second moments of the empirical distribution across the various layers. Now each of these $z\_i^l$ are a sum of uncorrelated random variables, so as we increase $N\_l$ they are going to look like some mean zero Gaussian, which the $q\_l$ are the second moments of. You can write down an integral for how the Guassians evolve at each layer (basically the limit of the sum coming from the definition), which gives you an iterative definition $q\_l = \mathcal{V}(q^{l-1} | \sigma\_w, \sigma\_b)$.

They plot some pictures of the dynamics of $q\_l$ in various cases. They look at $\phi(h) = \text{tanh}(h)$ with various choices of $\sigma\_w$ and $\sigma\_b$. For $\sigma\_b=0$ and $\sigma\_w < 1$ (bias-free, low weight) the network shrinks lengths, and the only fixed length vector is the zero vector. Still bias-free but with larger weights, the network expands small inputs and shrinks large inputs, and acquires another (non-zero) fixed point. As soon as the bias is non-zero, there is a single, stable, non-zero fixed point.

Suppose now we had two inputs $\mathbf{x}^{0,1}$ and $\mathbf{x}^{0,2}$ and we look at how they evolve together by looking at the sequence of $2 \times 2$ matrices of inner products. We already know what's happening on the diagonal, so our focus is now on the crossterms. In much the same way as before, we can write this inner product as a convolution of the two previous measures, and get an iterative definition $q\_{12}^l = \mathcal{C}(q\_{12}^l, q\_{11}^l, q\_{22}^l|\sigma\_w, \sigma\_b)$.

We now have some good tools to study the natural question of how two points that start close together evolve, and what effect varying $\sigma\_b$ and $\sigma\_w$ has on this. They express their results in terms of the correlation coefficient $c\_{12}^l = q\_{12}^l (q\_{11}^l q\_{22}^l)^{-1/2}$.

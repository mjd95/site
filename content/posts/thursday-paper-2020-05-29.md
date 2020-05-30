---
title: "Thursday Paper: A Simple Neural Network Model for Relational Reasoning"
date: 2020-05-29T08:34:17+01:00
---

[Today's paper](https://arxiv.org/pdf/1706.01427.pdf) is a DeepMind paper from 2017 NeurIPS. It's about relational reasoning: how we can build systems that understand entities and the relationships between them, and make inferences based on those entities and relationships. A purely symbolic approach to AI is going to be happy approaching things from this perspective, but such systems do not deal with noise and uncertainty well. At the opposite end of the spectrum we have statistical learning techniques, which deal with noise well but struggle to learn complex relations between entities, especially if there is not a plethora of data about the relations to train on.

This is all pretty abstract, so discussing an actual task will help ground things. One of the main things this paper focusses on is performance on [CLEVR](https://cs.stanford.edu/people/jcjohns/clevr/). This is a dataset which presents an image and a set of questions about that image. To answer them, you will need to detect objects in that image, understand their properties, and perform a number of "reasoning" style tasks (attribute identification, counting, comparison, spatial relationships, and logical operations) based on those. This seems like a hard task. I suppose we'll see what the SOTA and this paper's score is. (Recall also that this paper is a few years old as well now, so these could have improved since then).

How are the authors going to do better at CLEVR? They're going to introduce a new neural network "module" called a Relation Network (RN) that is focussed on relational reasoning. They will combine this with CNNs and LSTMs so they can extract entities to feed in to their RN.

## What is a Relation Network?

One might say that CNNs have a built-in capacity to reason about spatial, translation invariant properties. By way of analogy, we're going to try to construct a type of network that has a built-in capacity to compute relations.

In the simplest form a relation network is a function $\text{RN}(O) = f\_\phi(\sum\_{i,j} g\_\theta(o\_i, o\_j))$. Here $O = (o\_i)\_{1 \leq i \leq n}$ is a set of objects (things you want to learn relations between), $g\_\theta$ is an MLP (which is learning "relations" between pairs of objects) and $f\_\phi$ is another MLP (which is learning what to do with those relations).

By default RNs work on a full mesh of object pairs, but if you wanted to restrict to considering only relations between certain objects you could easily insert that in to the definition. In general, though, since the default position is no prior on what the relations between objects are, it's the job of the RN to learn what relations exist.

It's important to note that there is one (as opposed to $n^2$) function(s) $g\_\theta$ that is supposed to know about the relationship between all $n$ of your object. This is sharing parameters and data efficient, so if it works then you're going to be happy. (And I'm assuming this paper will say it works).

Note also that the summation over $i$ and $j$ ensures that the RN is invariant to the ordering of the objects presented to it. This is again a nice property to have.

## Tasks

The authors discuss a few tasks that they applied their models to, but due to lack of time I'm only go to look at CLEVR. This is a visual question-answering task. 

In general, visual QA tasks require scene understanding, relational reasoning (spatial and otherwise). To properly answer the questions, one requires knowledge about the world which is not present in the data set. However, there are linguistic biases in how the questions are sometimes asked that some systems have exploited to perform quite well on these tasks without actually having a deep understanding.

CLEVR is an attempt to distill the core challenges of visual QA. An interesting aspect is that it certainly seems to require some reasoning capabilities. In the original CLEVR paper it was noted that ResNet-101 image embeddings with LSTM question processing and augmented with stacked attention models got an overall score of 68.5% (52.3% was next best, 92.6% is human level). But the interesting part is that even this ResNET+LSTM model performed at about the baseline level for the questions which required the most relational reasoning.

The authors used two versions of the CLEVR input. One was the standard pixel version of the image, another was where the objects had already been extracted and listed as a series of state vectors (each object's state being position, color, shape, size, etc.)

## Models

In the authors' parlance, RNs operate on "objects", which is a pretty general notion. The authors get good mileage out of passing in CNN and LSTM embeddings as the input objects to the RNs.

The objects output by the CNN are pretty much what you would expect. They do not care which particular image feature constitutes an object, simply trusting that the CNN has discovered at least some useful objects and trusting that the RN will be able to infer what's important here.

They actually condition the function $g\_\theta$ on the question it's supposed to be processing (that is, on the LSTM embedding of the question). This is different to and less data-efficient than what they said in the introduction, but I guess it is reasonable.

In the case of state descriptions (rather than pixels) as input you can ignore the CNN part and just feed the state description straight in to the RN. You condition on the LSTM embedding of the question in the same way though.

I'm just going to directly quote the final model configuration:

> For the CLEVR-from-pixels task we used: 4 convolutional layers each with 24 kernels, ReLU non-linearities, and batch normalization; 128 unit LSTM for question processing; 32 unit word-lookup embeddings; four-layer MLP consisting of 256 units per layer with ReLU non-linearities for $g\_\theta$; and a three-layer MLP consisting of 256, 256 (with 50% dropout), and 29 units with ReLU non-linearities for $f\_\phi$.  The final layer was a linear layer that produced logits for a softmax over the answer vocabulary.  The softmax output was optimized with a cross-entropy loss function using the Adam optimizer with a learning rate of $2.5e^{−4}$.  We used size 64 mini-batches and distributed training with 10 workers synchronously updating a central parameter server.

## Results

They get SOTA performance on CLEVR from pixels of 95.5%, which is significantly better than the previous model and even better than human. (They mention other work from the same time that gets an even better result but uses a completely different approach with some sort of privileged training information, so it's not directly comparable.) They also do slightly better still (96.4%) on CLEVR from the state description.

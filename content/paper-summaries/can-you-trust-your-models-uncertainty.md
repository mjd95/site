---
title: "Can You Trust Your Model's Uncertainty?"
date: 2020-06-04T08:55:24+01:00
---

Another NeurIPS 2019 paper today, [Can You Trust Your Model's Uncertainty](https://arxiv.org/pdf/1906.02530.pdf). Given that deep learning are being used to make predictions in the real world, it would be nice if we knew how _certain_ these models were in the predictions that they make. An interpretable score for the certainty could then be used as input for whether the model should make the decision, or whether it should be defered to a human.

Recall that a model is calibrated if the probability that it predicts for a class aligns with its confidence in that prediction. If you are repeatedly sampling from the same fixed distribution then there isn't really any concern here: you can adjust your model during training to ensure that these line up. (The paper refers to this as "calibration in the i.i.d. setting"). In the real world, the samples coming in to your model are not necessarily from that same fixed distribution. One obvious example is simply that your model can be fed a sample from a completely different distribution. Another is that over time there could be drift between what you trained with and what your being fed. In both these cases, your model should display some amount of uncertainty, but there's no guarantees that it will.

So how do you measure uncertainty? There have been a bunch of ways proposed, and the goal of this paper is to do an empirical evaluation of the different methods. The authors argue that dataset shift is crucial for real-world applications and hence they focus their evaluation on that. They are interested in the following questions:

* How trustworthy are the uncertainty estimates of different evaluation methods under dataset shift?
* Does calibration in the i.i.d. setting translate to calibration under dataset shift?
* How do uncertainty and accuracy of different methods co-vary under dataset shift? Does anything do consistently well?

## Evaluation Process

Write $p^\*(\mathbf{x}, y)$ for the (unknown) true distribution over the training dataset $\mathcal{D}$ and $p^\*(y|\mathbf{x})$ for the true conditional distribution. Write $p\_{\theta}(y|\mathbf{x})$ for the distribution modeled by the learned neural network. This will be evaluated by providing it samples from different distributions $q(\mathbf{x}, y)$, specifically shifted versions (corruptions and perturbations of samples in $\mathcal{D}$ that still land in one of the classes - they refer to another paper which presumably has a more precise definition) and also completely different datasets (where the ground truth is a different class).

They evaluate a number of methods for constructing a classifier model, including:

* _Vanilla_: Maximum softmax probability
* _Temperature Scaling_: Use a validation set to do post-hoc calibration. This is "calibration in the i.i.d. setting" - ensuring that your model is well-calibrated on the original dataset
* _Enesembles_: Just an ensemble model of $M=10$ independently trained models
* _Stochastic Variational Bayesian Inference_: I haven't come across this before, so I'll treat it as a black box

They evaluate using a few metrics, including _Negative Log Likelihood_ (the usual, low is good), _Brier score_ (squared error of the predicted probability vector and the one-hot encoded true response, low is good), _Expected Calibration Error_ (the average gap between within-bucket-accuracy and within-bucket-predict-probability, low is good) and entropy.

## Experiments and Results

They evaluate the predictive uncertainty in the images, text, and categorial data (ad data - this is a Google paper after all). Whatever the domain, they follow a standard approach: do normal training/validation/testing with a fixed dataset, then evaluate results on increasingly shifted and OOD datasets. There's a detailed discussion on the MNIST case so I'll focus on that.

They train a LeNet architecture on MNIST in the normal way. To explain the predictive accuracy of the shifted versions of the dataset, they have a few graphs:

* Rotate the images from 0 to 180 degrees, plot the accuracy and Brier score as a function of the angle.
* Translate the images from 0 to 14px, plot the accuracy and Brier score as a a function of the shift.
* Rotate images by 60 degrees. Consider a threshold $\tau$ varying from 0 to 1, plot the accuracy on samples for which the prediction has probability $\geq \tau$.
* Again rotate images by 60 degrees. Same increasing threshold, plot the number of examples for which the probability is larger than that threshold.
* Use a genuinely different distribution (Not-MNIST), plot the entropy against the number of samples
* Use a genuinely different distribution (Not-MNSIT), consider increasing thresholds and plot the number of samples for which the probability exceeds that threshold

Each of these are plotted for the various methods (Vanilla, Temp Scaling, Ensemble, LL-Dropout, SVI, LL-SVI, Dropout).

For the first two (increasingly rotated and increasingly translated), the accuracy for each method follows a similar pattern (high accuracy with little shift, (very) low accuracy at 90 degrees, kicking back a little bit towards 180 degrees. (Of course, MNIST is not rotation invariant, but some symbols look the same rotated 180 degrees. This size of the kick looks about the same as the proportion for which this is the case). The Brier score for different methods is a little bit more different. Temparature Scaling is the SOTA post-hoc calibration method, but this does not translate in to good calibration under shift. For both rotation and translation, SVI has the best Brier score. SVI has a slightly worse accuracy, but you would trade it off for the additional robustness to OOD data.

In fact, for the MNIST case, SVI does pretty well across the board as a model that is not necessarily the best but not over-confident in its bad predictions.

If you stopped reading at this point (which I did, before the meetup!) you might take away the message that SVI is the best. But actually, for the other image experiments (ImageNet) ensemble methods are the best performers. Arguably the experiment is a bit small-scale to fully support this conclusion though (Figure 2 is a box-plot where each of the boxes seems to contain only eight samples).

The authors go on to discuss other examples, but I'm skipping those due to lack of time.

## Conclusion

The authors point out that measuring calibration under dataset shift is a good thing to do, and point out that well-calibrated models do not generally retain this property under shift. In fact, post-hoc calibration does pretty bad under shift. SVI seems to do well on MNIST; they suggest its poorer performance on ImageNet is due to the size of the dataset. Overall, ensembel methods seem to do the best.

They offer up their test suite as a benchmark. It seems worthwhile to me as a first step, but to be honest the only one I looked at in detail (MNIST) seems like a set of fairly trivial modifications to make the images and I would want something more representative of the real world in my benchmark.


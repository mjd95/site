---
title: "Notes on Deep Learning (Goodfellow et. al)"
date: 2019-09-24T08:08:00+01:00
---
I've been reading the book [Deep Learning (Goodfellow et. al)](https://www.deeplearningbook.org/) recently.  I'm doing this with a view to filling in gaps about my deep learning knowledge, as most of my learning in the past has been from random blogs posts and the occassional research paper.

Since this is a fairly long book and I am constrained in the amount of time I can dedicate to reading it, I've tried an experiment with keeping quick markdown notes of what I've read.  This means that when I come back to reading after a short break, I can remember the salient points of where I was by a quick skim over my notes.

In the unlikely situation that my notes could be useful to anyone else, I though I'd post them here.  Due to my background, I skipped the mathematical prelude before Chapter 5.

## Chapter 5 - Machine Learning Basics

### Learning Algorithms

  * A computer program is said to learn from experience $E$ to do task $T$, measured by performance measure $P$, if its performance at $T$, as measured by $P$, is improved with $E$
  * Some examples of tasks you might want to perform:
    - Classification: learn a function $f: \mathbb{R}^n \to \{1,...,k\}$, or output a probability distribution over the classes, 
    - Regression: predict $f: \mathbb{R}^n \to \mathbb{R}$, such as predicting a house price
    - Transcription (OCR, speech recognition), machine translation, anomaly detection, estimating pdfs, ...
  * E.g. performance measure: for classification, could measure performance by accuracy (ratio of correct/all) or error rate (incorrect/all).  Can think of error rate as expected 0-1 loss (lose 0 if correct, lose 1 if incorrect).
  * Supervised vs unsupervised is a differentation between the kind of experience the process is allowed
    - In a technical sense, there isn't a huge difference.  In supervised learning, you're definitely learning $p(y|\mathbf{x})$.  In unsupervised, you're arguably learning $p(\mathbf{x})$.  By ordering the features of $\mathbf{x}$ and succcessively conditioning on them, the latter reduces to the former.
    - In practice, what is computationally tractable is presumably very different
  * Reinforcement learning is out of scope for this book
  * Details of simplest example of machine learning (linear regression):
    - We are learning a linear function $f: \mathbb{R}^n \to \mathbb{R}$
        - Task $T$ is to predict $y$ from $\mathbf{x}$ by outputting $y = \mathbf{w}^{\intercal} \mathbf{x}$
        - Experience $E$ will be whatever dataset we give it, i.e. the pairs $(\mathbf{x}, y)$
        - Performance measure $P$ is to what extent it minimises the MSE loss function on the test data set
    - Intuitively, it makes sense to choose the weight $\mathbb{w}$ that minimise the MSE on the training dataset.  This can be found with basic calculus.  (These are the _normal equations_)
    - Now we have a $\mathbb{w}$ and can make predictions.  They're probably reasonably good.

### Capacity, Overfitting and Underfitting

  * Minimising the training error in itself (like linear regression above) is just an optimisation problem
  * Generalisation/test error is the expected error on a new input, with expectation being taken across possible inputs.  This is what we really care about
    - Obviously this is hard to compute in practice, but it can be estimated by averaging the error on a reasonable number of test examples
  * Statistical learning theory comes in to play:
    - For the algorithm to have any success, obviously there has to be some relation between the training set and the test set
    - Typically, examples are assumed iid, and training and test examples are drawn from the same population
    - The underlying distribution is called the data-generating distribution, $p\_{\text{data}}$
    - Since they are drawn from the same data-generating distribution, clearly if we fix the parameters and calculate the expected training and test errors, they are the same
  * In practice, you set the parameters only after examining the training set, then compute the expected test error.  So the expected test error is $>=$ to expected training error
  * _Underfitting_: when the model can't get a low error on the training set
  * _Overfitting_: when the test error is much larger than training error
  * Since expected test error $>=$ expected training error, both underfitting and overfitting mean bad (expected) test errors
  * _Model capacity_: think of large capacity = high degree polynomial fit.  Large capacity tends to overfit, low capacity tends to underfit
  * _Representational capacity_: the family of functions the learning algorithm can choose (e.g. in my setup, maybe I'm fitting a polynomial and any polynomial of deg $<=10$ is fair game)
  * _Effective capacity_: what the learning algorithm can actually reach inside the representational capacity, given the shortcomings of the optimisation algorithm (e.g., walking in the space of polynomials of degree $<=10$ with discrete step sizes => not all real coefficients are possible => only a subset of polynomials are actually reachable)
  * _Vapnik-Chervonenkis dimension_: measures the capacity of a binary classifier.
    - Defined as largest $m$ s.t. there exists a training set of size $m$ different points that the classifier can label arbitrarily.
    - So for example a classifier which just assigns the same label to all samples has a VC dimension of $0$.
  * _Statistical learning theory_: proves theorems like $(\text{test error}) <= (\text{training error}) \times f(\text{capacity}, \text{# training samples})$, where $f$ increases as capacity does, but decreases as # training samples does.
    - In practice, have no idea what the capacity of deep learning algorithms are.
    - The optimisation algorithm affects the capacity (effective capacity vs. representational capacity), and we don't understand the optimisation algorithm
  * _Parametric models_: learn a function described by a parameter vector whose size is finite and fixed before any data is observed
    - E.g. linear regression is a parametric model - there is a fixed number of weights before starting
  * Non-parametrics models are everything else
    - E.g. nearest neighbour regression: given a test sample $\mathbf{x}$, I search through all my training samples $\mathbf{X}$ and find the one closest to $\mathbf{x}$, and return the output $y$ for that closest training sample.
    - Could also allow the parameter list to grow inside a parametric model, e.g. allow unbounded polynomial expansions of input to linear regression
  * _Bayes error_: error of a classifier which knows the true joint probability distrubtion of $(\mathbf{x}, y)$.  If $\mathbf{x} \to y$ is deterministic then Bayes error is zero, but otherwise it will be non-zero.
  * No free lunch theorem: averaged over all possible data-generating distributions, every classification algorithm has the same error rate when classifying previously unobserved points
    - I.e. there is no "universally good" classifier
    - I.e. need to choose the right classifer for the right problem
  * Regularisation: roughly, making the learning algorithm prefer one solution over another
    - E.g. weight decay in linear regression.  Instead of just minimising the MSE on the training set, we now minimise $J(\mathbf{w}) = \text{MSE}\_{\text{train}} + \lambda \mathbf{w}^{\intercal} \mathbf{w}$.  When $\lambda = 0$ this is the same as before, but when $\lambda$ is large it forces us to prefer $\mathbf{w}$ whose coefficients are small
    - Previously we flat-out excluded certain functions from the hypothesis space of functions.  In a sense, this is expressing an infinitely strong preference against that function.  Regularisation more generally gives a more nuanced approach
  * Regularisation: actual definition given, any modification we make to a learning algorithm that is intended to reduce its generalisation error but not its training error

### Hyperparameters and Validation sets

  * Hyperparameters: settings we use to control an algorithm's behaviour
    - E.g. degree of polynomial in linear regression, value of the weight decay parameter $\lambda$
  * Could deem any parameter a hyperparameter (and hence stop the learning algorithm from to optimise it)
    - Maybe because some parameter is difficult to optimise
    - Maybe because it doesn't make sense to attempt to learn that parameter in the current problem.  E.g. don't try to learn $\lambda$ in your linear regression problem with large capacity, because it's definitely going to overfit by picking $\lambda=0$ and a high degree polynomial
  * Use a validation set for checking hyperparameters.  We still keep the test set separate (model, including its hyerparameters, should know nothing about the test set).  Instead, we split the training set in to two parts.  Fix the hyperparameters and use the first part to learn the parameters.  Then use the second part to estimate the generalisation error.  If it looks terrible, go back and tweak the hyperparameters.  If it looks good, proceed to the test set.
    - Somewhat confusing, the "first part of the training set" is called the training set as well.
    - Rule of thumb: roughly 80% of the training data goes to actual training, roughly 20% goes to validation
  * If the dataset available to you is small, dividing in to a fixed training and test set is problematic.  With little data, there is lots of statistical uncertainty.
  * Cross-validation gets round this, at cost of increased computation.
    - E.g. $k$-fold cross-validation.  Split the dataset in to $k$ parts.  Do $k$ passes through the data.  On the $i$th pass, declare the $i$th part to be the test set and the rest to be the training set.
    - Can do the same thing with training/validation/test splits
    - Cross-validation is potentially problematic, [Bengio and Grandvalet (2004)](http://www.jmlr.org/papers/volume5/grandvalet04a/grandvalet04a.pdf) probably worth reading

### Estimators, Bias, and Variance

  * Basic stats things like parameter estimation, bias, and variance are used in ML to talk about generalization, underfitting, and overfitting
  * Point estimation: supposed to capture notion of providing a single "best" prediction of a quantity of interest
  * Given $m$ iid data points $\mathbf{x}^{(1)}, ..., \mathbf{x}^{(m)}$, define a _point estimate_ (or _statistic_) as any function of the data $\widehat{\theta}\_{m} = g(\mathbf{x}^{(1)}, ..., \mathbf{x}^{(m)})$.
    - This definition is literally any function of the data, doesn't have to be related to anything
    - Obviously in practice we are interested in functions for which $\widehat{\theta}\_m$ is close to some quantity of interest $\theta$
  * Take the frequentist approach: assume the true value of $\theta$ is fixed but unknown.  Note that since $\widehat{\theta}$ is a function of the data, which is drawn from a probability distribution, $\widehat{\theta}$ is itself a random variable
  * Function estimation: estimating a function of the data.  This is just a point estimate in a function space, e.g. linear regression is estimating a point in the space spanned by the coefficients of the weight vector
  * Bias of an estimator: $\text{bias}(\widehat{\theta}\_m) = \mathbb{E}(\widehat{\theta}\_m) - \theta$. Here the expectation is over the data $\mathbf{x}^{(1)},...\mathbf{x}^{(m)}$, and $\theta$ is the true value
    - Say $\widehat{\theta}\_m$ is unbiased if $\text{bias}(\widehat{\theta}\_m) = 0$
    - Say $(\widehat{\theta}\_m)\_{m \geq 1}$ is asymptotically unbiased if $\lim\_{m \rightarrow \infty} \text{bias}(\widehat{\theta}\_m) = 0$
  * E.g. Bernoulli distribution with mean $\theta$ (so $P(x; \theta) = \theta^{x} (1-\theta)^{1-x}$).
    - Given observations $x^{(1)},...,x^{(m)}$, we might make the point estimate $\widehat{\theta}\_m = \frac{1}{m} \sum\_{i=1}^m x^{(i)}$
    - Simple calculation shows this is an unbiased estimate of $\theta$
  * E.g. Gaussian $\mathcal{N}(x; \mu, \sigma)$
    - Again with $m$ observations, try the sample mean $\widehat{\mu}\_m = \frac{1}{m} \sum\_{i=1}^m x^{(i)}$ as an estimator for the mean
    - Very similar computation shows that this is unbiased
    - Now try sample variance $\widehat{\sigma}\_m^2 = \frac{1}{m} \sum\_{i=1}^m \left(x^{(i)} - \widehat{\mu}\_m \right)^2$ as an estimator for the variance
    - Standard calculation shows $\text{bias}(\widehat{\sigma}\_m^2) = -\sigma^2/m$, so this is a biased estimator
    - Unbiased sample variance is $\tilde{\sigma}\_m^2 = \frac{1}{m-1} \sum\_{i=1}^m \left(x^{(i)} - \widehat{\mu}\_m \right)^2$
    - This one is unbiased
  * We computed the expectation of an estimator to measure its bias, we can also compute the variance of an estimator.  Intuitively, we'd like this variance to be low
  * Write $\text{Var}(\widehat{\theta})$ for the variance (variance over the empirical distribution again), and write $\text{SE}(\widehat{\theta})$ for the standard error (positive square root of the variance)
  * E.g. Standard error of sample mean is $\text{SE}(\widehat{\mu}\_m) = \sqrt{\text{Var}(\frac{1}{m}\sum\_{i=1}^m x^{(i)})} = \sigma/\sqrt{m}$, where $\sigma$ is the true standard error.
    - We don't _know_ $\sigma$ though.  We can attempt to get an _estimate_ for the standard error of the sample mean by using an estimate for $\sigma$
    - The two we've seen (square root of sample variance, square root of unbiased sample variance) both result in biased estimates
    - The square root of the unbiased sample variance is used in practice, it works okay for large $m$
  * Use of standard error in ML:  
    - Have some model capable of making predictions.  Look at some of the errors it makes on the test set, and compute the sample mean.  
    - This is an estimate of the true error on the test set, i.e. the generalisation error.  How close this is to the true error depends on size of test set
    - By the central limit theorem, our sample mean is normally distributed around the true error
    - We don't know the standard error of this normal, but we can use an _estimate_ for the standard error, as above.  Hence we can compute confidence intervals for the true expected error
    - E.g. 95% confidence interval centred on $\widehat{\mu}\_m$ is $(\widehat{\mu}\_m - 1.96\text{SE}(\widehat{\mu}\_m), \widehat{\mu}\_m  + 1.96\text{SE}(\widehat{\mu}\_m))$ under the normal distribution with mean $\widehat{\mu}\_m$ and variance $\text{SE}(\widehat{\mu}\_m)^2$
  * E.g. Bernoulli distribution
    - Have sample mean $\widehat{\theta}\_m$, compute its variance $\text{Var}(\widehat{\theta}\_m)$
    - Turns out $\text{Var}(\widehat{\theta}\_m) = \frac{\theta(1-\theta)}{m}$
    - As $m \rightarrow \infty$, $\text{Var}(\widehat{\theta}\_m) \rightarrow 0$.  This is quite common
  * Intuitively, if you are learning a function and you overfit, you would expect your estimate for that function to have low bias but high variance
  * Similarly, if you underfit, you would expect your estimate for the function to have low variance but potentially high bias
  * Overall quality of an estimate takes both into account.  E.g. the MSE satisfies $\mathbb{E}[(\widehat{\theta}\_m - \theta)^2] = \text{bias}(\theta\_m) + \text{Var}(\theta\_m)$.  There is probably a sweet spot where both bias and variance are reasonably small that gives the best error in the MSE sense
  * Taking an increasing number $m$ of samples from the training set, our estimate $\widehat{\theta}$ is actually a sequence $(\widehat{\theta})\_m$.  We say $\widehat{\theta}$ is _weakly consistent_ if the sequence converges in probability to $\theta$, and is _strongly consistent_ if the sequence converges almost surely to $\theta$.
    - I guess we can construct many sequences from one estimator, but they'll converge in probability to each other anyway so it doesn't matter?  Might need some reasonability assumptions on $\widehat{\theta}$
  * Clearly (weakly) consistent estimators are asymptotically unbiased
  * Converse is false, e.g. estimate the mean of a Gaussian by always saying the mean is always the first value you saw, no matter how many datapoints are thrown at you.  This fits the definition of an estimator, is unbiased (hence asymptotically unbiased), but is not consistent

### Maximum Likelihood Estimation

  * Random guessing is probably not a good way to build estimators in general.  MLE is the most common systematic way to do so
  * Have $m$ samples $\mathbf{x}^{(i)}$ drawn independently from some (unknown) distribution $p\_{\text{data}}$.  Have some parameter $\theta$ and a family of distributions $p(\mathbf{x}; \theta)$ s.t. for each concrete value of $\theta$ we get some distribution $p(\mathbf{x})$ which may or may not be related to $p\_\text{data}(\mathbf{x})$.  The MLE for $\theta$ is $\theta\_{\text{ML}} = \text{argmax}\_{\theta} \prod\_{i=1}^m p(\mathbf{x}^{(i)}; \theta)$, i.e. the choice of $\theta$ that maximises the likelihood of the observed data
  * More numerically convenient to work with log-likelihood, $\theta\_\text{ML} = \text{argmax}\_\theta \sum\_{i=1}^m \log p(\mathbf{x}^{(i)}; \theta)$.  Equivalently, $\theta\_\text{ML} = \text{argmax}\_\theta \mathbb{E}\_{\mathbf{x} \sim \widehat{p}\_{\text{data}}} [\log p(\mathbf{x}; \theta)]$, where $\widehat{p}\_\text{data}$ is the empirical distribution coming from our sampling from $p\_\text{data}$
  * E.g. suppose I have $m$ samples from a Bernoulli distribution with parameter $\theta$, but I don't know $\theta$ and I want to estimate it instead.  For a particular value of $\theta$, I know that $p(x; \theta) = \theta^{x}(1-\theta)^{1-x}$.  So the thing I'm trying to argmax is $\sum\_{i=1}^m \log (\theta^{x\_m}(1-\theta)^{1-x\_m}) = \sum\_{i=1}^m x\_m \log(\theta) + (1-x\_m)\log(1-\theta)$.  The derivative of this is $\sum\_{i=1}^m \frac{x\_m}{\theta} - \frac{1-x\_m}{1-\theta}$ which is extremal when $\sum\_{i=1}^m(1-\theta)x\_m - (1-x\_m)\theta = 0$, i.e. $m\theta\_{\text{ML}} = \sum\_{i=1}^m x\_m$, i.e. at the sample mean.  This can be checked to be a maximum.
  * Equivalent point of view: MLE is the estimate that moves $p(\mathbf{x}; \theta)$ closest to the empirical distribution $\widehat{p}\_{\text{data}}$ in the KL sense.  (A short calculation shows that attempting to minimise the KL divergence boils down to maximising the above expectation over the empirical distribution.)  
  * In a supervised setting, can do the same thing with conditional probabilities.  Here $\theta\_{\text{ML}} = \text{argmax}\_{\theta} \sum\_{i=1}^m \log p(\mathbf{y}^{(i)} | \mathbf{x}^{(i)}; \theta)$
  * E.g. linear regression.  Previously we just searched for functions of the form $y = \mathbf{w}^{\intercal} \mathbf{x}$, and decided to pick $\mathbf{w}$ that minimised the mean square error on the training set.  Now think of the linear regression as outputting a conditional distribution $p(y | \mathbf{x})$ with Gaussian noise, i.e. we define $p(y|\mathbf{x}) = \mathcal{N}(y; \widehat{y}(\mathbf{x}; \mathbf{w}), \sigma^2)$.  A short calculation shows that maximising the log-likelihood of this is equivalent to minimising the the MSE on the training set as before
  * MLE is consistent, under the condition that there is precisely one value of $\theta$ s.t. $p\_{\text{data}} = p(.; \theta)$
  * Consistency says nothing about the rate of convergence.  That is measured by statistical efficiency, which is quantified for parameter estimates by the MSE (over the data-generating distribution) between the estimated and the true parameter.  Cramer--Rao says that the MLE has the lowest MSE amongst all consistent estimators, so in this sense MLE is the asymptotically best estimator
  * In practice, with small amounts of training data, maybe use regularization strategies to get biased versions of MLE which have lower variance

### Bayesian Statistics

  * Frequentist approach: true value of $\theta$ is fixed but unknown, the point estimate $\widehat{\theta}$ is a random variable (since it is a function of the dataset, which is regarded as random)
  * Bayesian approach: dataset is directly observed (so not random), and $\theta$ is unknown _and uncertain_ so $\theta$ itself is a random variable
  * Before observing data, represent our knowledge of $\theta$ by specifying a prior distribution $p(\theta)$.  Initially it can just be some distribution with high entropy, e.g. uniformly distributed in some range, or Gaussian
  * Given data $x^{(1)},...,x^{(m)}$, we have the posterior distribution $p(\theta|x^{(1)},...,x^{(m)}) = \frac{p(x^{(1)},...,x^{(m)}|\theta)p(\theta)}{p(x^{(1)},...,x^{(m)})}$
  * E.g. Bayesian linear regression:
    - In standard linear regression, you have $\widehat{\mathbf{y}} = \mathbf{X}\mathbf{w}$, where $\mathbf{X}$ is the whole training set.  
    - First we found optimal $\mathbf{w}$ by minimising the MSE.  
    - Then we redid it by assuming $p(y|\mathbf{x})$ was Gaussian (say with $\sigma^2=1$, standard apparently) and using the MLE for $\mathbf{w}$.  This recovered the same value as MSE.
    - For Bayesian, we still work with $p(\mathbf{y}|\mathbf{X}, \mathbf{w}) = \mathcal{N}(\mathbf{y}|\mathbf{X}\mathbf{w}; \mathbf{I})$.  We choose a prior $p(\mathbf{w}) = \mathcal{N}(\mathbf{w}; \mathbf{\mu}\_0, \mathbf{\Lambda}\_0)$.  Computing the posterior distribution $p(\mathbf{w}|\mathbf{X},\mathbf{y})$, turns out to be Gaussian again, with the mean and covariance matrix a function of both the priors and the data (i.e. Gaussian family is _self conjugate_)
  * In the Bayesian approach, you can make predictions using the whole posterior distribution of $\theta$.  If it's computationally intractible, though, you might want a point estimate of $\theta$ anyway to use for your predictions.  This estimate will obviously take your prior into account, unlike a purely frequentist approach
  * This is where MAP estimates come in, $\theta\_{\text{MAP}} = \text{argmax}\_{\theta} p(\theta | \mathbf{x}) = \text{argmax}\_{\theta} \left(\log(p(x|\theta)) + \log(p(\theta))\right)$.  This is MLE with regularisation
  * E.g. Bayesian linear regression with $\mathcal{N}(\mathbf{w}; 0, \frac{1}{\lambda}\mathbf{I})$ prior.  The log prior term then corresponds to weight decay

### Supervised Learning Algorithms 

  * Logistic regression: force a linear function $\mathbf{x} \mapsto \mathbf{\theta}^T \mathbf{x}$ to take values in $(0,1)$ by composing with the logistic sigmoid $\sigma(x) = 1/(1+e^{-x})$, and interpret the result as a probability.  Now you can use a linear function to do classification: $p(y=1|\mathbf{x},\mathbf{\theta}) = \sigma(\mathbf{\theta}^T\mathbf{x})$.  No closed form solution for the optimal weights, instead search using gradient descent
  * Support vector machines:
    - Have $f(\mathbf{x}) = b + \sum\_i \alpha\_i k(\phi(\mathbf{x}), \phi(\mathbf{x}^{(i)}))$, where $\phi$ is some feature mapping and $k$ is some kernel.  Simplest case would be $\phi$ is the identity and $k$ is the dot product, but probably we're thinking of $\phi$ mapping in to some infinite-dimensional space
    - Most commonly used kernel is the Gaussian kernel, $k(\mathbf{u}, \mathbf{v}) = \mathcal{N}(\mathbf{u}-\mathbf{v}; 0, \sigma^2 \mathbf{I})$, corresponds to a dot-product in an infinite dimensional space
    - Think of Gaussian kernel as performing template matching.  Training example $(\mathbf{x}, y)$ becomes a template for class $y$.  When $\mathbf{x'}$ is near $\mathbf{x}$ in the Euclidean sense, the Gaussian kernel has a large response, so the contribution of $\mathbf{x}$ to the prediction for $\mathbf{x'}$ will be large (the corresponding $\alpha$ will count a lot).  We combine those for all training examples $\mathbf{x}$ to get our final prediction for $\mathbf{x'}$
    - Generally, algorithms that look like this are called _kernel machines_ or _kernel methods_
    - Since $f(\mathbf{x})$ involves each $\alpha\_i k(\mathbf{x}, \mathbf{x}^{(i)})$ this could be very expensive to evaluate (involves computing the kernel once for each training sample).  In SVMs, most the of the $\alpha\_i$ are zero, so it's not so bad.  Those $\alpha\_i$ that are non-zero are called _support vectors_
    - History note: current deep learning renaissance [began](https://www.cs.toronto.edu/~hinton/absps/fastnc.pdf) when a neural network was shown to outperform the radial basis function kernel SVM on MNIST
  * $k$-nearest-neighbours (kNN) is sort of a supervised learning algorithm.  But there's not really much learning, it's really just a function of the training data (given the thing you're predicting, find the $k$ nearest neighbours and average their votes)
    - This is a non-parametric learning algorithm, so it has a very high capacity.  So (at a high computational cost) it can obtain a high accuracy on a large training set.  It might generalize badly from a small training set (overfitting)
    - Disadvantage of naive kNN is that it treats all features equally, so if you had 100 features but only the first one actually mattered for the prediction, kNN wouldn't know that.  Of course you could alter the "nearest" metric to emphasise the first component, but I guess if you already knew that the first one was the only one that mattered you wouldn't be training a model with all 100 features
  * Decision trees, divide space into subregions, subregions are in bijection with leaf nodes.
    - Use specific algorithms to learn the tree, don't really care about any of that in this book
    - Standard setup has axis aligned splits (e.g. $x\_2 > 5$), so will struggle to approximate conditions like $x\_2 > x\_1$
    - Seems pretty down on decision trees :)

### Unsupervised Learning Algorithms

  * Line between supervised and unsupervised is blurry - subjective to say whether a value is a feature or is provided by a supervisor
  * In practice unsupervised normally means working with data that has not been manually annotated by a human.  Normally involves density estimation, learning to draw samples from a distribution, learning to denoise data from a distribution, finding a manifold that data lies near/on, or clustering the data
  * Classic unsupervised learning problem is finding a lower-dimensional representation of the data.  Or a sparse representation.  Or an independent representation (i.e. aim for representation where features are statistically independent)
  * Principle Component Analysis (PCA):
    - PCA provides a means of compressing data.  Can also view as an unsupervised learning algorithm learning a representation of the data, specifically a lower-dimensional representation whose dimensions have no linear correlation with each other
    - Recap of algorithm: have points $\mathbf{x} \in \mathbf{R}^n$, want lower dimensional representations $\mathbf{z} \in \mathbf{R}^l$.  Consider linear transformations $D:\mathbf{R}^l \to \mathbf{R}^n$ s.t. the columns of $D$ are orthonormal, i.e. $D$ is the decoding function.  The encoding function is just $\mathbf{x} \mapsto D^{\intercal} \mathbf{x}$.  (Given that we've already picked the decoding function, it can be checked this encoding function minimises the distance between a given point $\mathbf{x}$ and its reconstruction.)  $D$ is chosen by minimising the cumulative distance between _all_ points and their reconstruction.  After a fair amount of maths we find that $D$ is matrix whose columns are the $l$ eigenvectors of $\mathbf{X}^{\intercal} \mathbf{X}$ corresponding to the largest eigenvalues
    - How does PCA decorrelate the input $\mathbf{X}$?
      - WLOG assume the data has mean zero, $m$ samples of dimension $n$.  The unbiased sample covariance is $\text{Var}(\mathbf{x}) = \frac{1}{m-1}\mathbf{X}^T \mathbf{X}$.  PCA finds a representation $\mathbf{z} = \mathbf{W}^T \mathbf{x}$ where $\text{Var}(\mathbf{z})$ is diagonal
      - Basically, we find a rotation $\mathbf{W}$ of input space that aligns the principle axes of variance with the basis of the rotated input space
  * $k$-means clusters:
    - Divides the training set in to $k$ different clusters of examples that are near each other, so it's like a $k$-dimensional one-hot coding of the input space ($h\_i(\mathbf{x})=1$ iff $\mathbf{x}$ in the $i$th cluster)
    - $k$-means algorithm starts with $k$ centroids $\mathbf{\mu}^{(1)}, ..., \mathbf{\mu}^{(k)}$ chosen somewhat arbitrarily (probably make them evenly spread in some sense).  Then alternate steps A and B, where step A assigns each datapoint to its nearest centroid, then step B is to replace each centroid with the mean of all datapoints currently assigned to it.  Do this until it seems to be converging
    - Easy to come up with mathematical measures of how good a clustering is, hard to say if the clusterings represent anything useful in the real world
    - Shortcoming of this one-hot approach: think of clustering red cars, blue cars, red trucks, blue trucks.  Any one-hot coding is going to be unsatisfactory, so can be useful to have sparse representations that are not one-hot

### Stochastic Gradient Descent

  * Cost function used by a ML algorithm often decomposes as a sum over training examples of a per-example loss function, e.g. conditional log-likelihood $J(\theta) = \frac{1}{m} \sum\_{i=1}^m L(\mathbf{x}^{(i)}, y^{(i)}, \theta)$, where $L(\mathbf{x}, y, \theta) = -\log(p(y|\mathbf{x}, \theta))$
  * To minimise this, we need to compute the gradient $\nabla\_{\theta} J(\theta) = \frac{1}{m} \sum\_{i=1}^m \nabla\_{\theta} L(\mathbf{x}^{(i)}, y^{(i)}, \theta)$, which is going to be expensive if you have a large training set
  * From the above form, the gradient is an expectation, so you can approximate it by sampling.  At each step, sample a _minibatch_ $\mathbb{B} = (\mathbf{x}^{(1)}, ..., \mathbf{x}^{(m')})$ where $m'$ is small (say, between 1 and 100).  Approximate the gradient by computing with these samples only, then walk downhill in the direction your approximation suggested

### Building a Machine Learning Algorithm

  * ML algorithms typically involve a dataset, a cost function, an optimization procedure, and a model
  * Can tweak cost function and optimization procedure to get new algorithms, e.g. add weight decay to cost function in linear regression

### Challenges Motivating Deep Learning

  * Development of DL motivated by failure of traditional algorithms on speech recognition and object recognition
  * Curse of dimensionality: many problems become difficult when number of dimensions is high
    - Need lots of data to get a reasonable covering of possibly values of $\mathbf{x}$
    - But lots of data means lots of computation
  * We've talked about priors formally in terms of prior distribution $p(\theta)$.  More generally, though, prioirs are any prior assumptions that affect the output model, e.g. choice of model family is itself a prior.  Generally these are hard to quantify with probability
  * E.g. _smoothness prior_: we assume that the functions we learn should be smooth / locally constant.  This is implicit in many simpler algorithms (kNN, kernel methods, decision trees)
    - Smoothness prior pretty much ties you down to a slicing of the input space, where the resulting number of regions is at most the number of trainng samples.  That's somewhat restrictive
    - E.g. how to learn a chessboard?  With local constancy, need a training sample on each square.  (Obviously can preprocess the input space, but the spirit of the question is one where you don't know you're learning a chessboard ahead of time) 
  * This can be solved, e.g. you can define $O(2^k)$ regions with $O(k)$ samples, so long as you introduce some dependencies between the regions through additional assumptions about the underlying data generating distribution, e.g. for chessboard introduce the assumption that target function is periodic (which is equivalent to preprocessing the input space)
    - Doing things like specifying that the target function is periodic is a massive assumption to feed in to your model
    - You don't have to feed specific assumptions in like this for DL.  DL itself already makes an assumption that your data is generated by a _composition of factors_, so presumably at some layer it should learn the periodicity itself and take that into account for future layers
  * ML people use the term _manifold_ fairly loosely, roughly related to the mathematical one.  One difference is that the dimension of the manifold doesn't have to be constant (e.g. a figure 8 is a "manifold", mostly 1D except for one point where it is 2D)
  * _Manifold learning_ assume that most of $\mathbb{R}^n$ consists of invalid inputs, and that interesting inputs only occur along various manifolds embedded inside $\mathbb{R}^n$
    - _Manifold hypothesis_: the probability distribution over images, text strings, and sounds that occur in real life is highly concentrated.  An image generated by randomly selecting a black or white pixel at each point is probably not going to look like a human face.  
      - So "most of $\mathbb{R}^n$ is invalid input"
      - Also need to show that interesting input are on manifolds (rather than just sparse isolated points).  Can imagine this informally, e.g. small perturbations, dimming/brightening an image, rotating, etc.
    - Would be very useful if the ML algorithm understood these manifolds and could give coordinates on them.  Will apparently explain how to do this by the end of the book

## Chapter 6 - Deep Feedforward Networks

  * _Deep feedforward networks_ (DFNs) (or _feedforward neural networks_, or _multilayer perceptrons_) provide a particular means to learn $y = f^\*(\mathbf{x})$ by considering a family $f(\mathbf{x}; \mathbf{\theta})$ and learning the $\mathbf{\theta}$ that gives the best approximation
  * They are _feedforward_ because information flows completely from $\mathbf{x}$ to $y$ - there is no _feedback_.  (When there is feedback, this is a recurrent neural network (RNN))
  * Convolutional neural networks (CNNs), popular in object recognition, are a special kind of DFN.  DFNs are also a conceptual stepping stone towards RNNs (which are popular in natural language tasks)
  * Feedforwarded networks have an associated DAG, nodes in the DAG are functions to evaluate and edges represent composition
    - Have _layers_.  First one you encounter when traversing the DAG is the input layer, last one is the output layer.  Layers in the middle are called hidden layers
    - Overall length to the output layer is called the _depth_
    - Each hidden layer is typically vector valued, the dimensionality of these hidden layers is called the _width_ of the layer 
    - Can think of each layer as a vector-to-vector function, or as composed of a number of _units_ which are vector-to-scalar functions.  The latter point of view can be pushed in to a rough anology with brains
  * Motivation: Consider trying to use a linear model (e.g. linear regression, logistic regression) in a case where there is a non-linear relationship between the input and the target.  Previously we would try to do some preprocessing with a function $\phi$
    - This could be hand-crafted, but any attempt is going to be very specialized to the problem/domain
    - Could use a very generic feature mapping like the RBF kernel, but this often does not generalize well (this optimises for finding a locally smooth solution, rather than digging in to the problem at hand)
    - In DL you _learn_ $\phi$.  Have $y = f(\mathbf{x};\mathbf{\theta}, \mathbf{w}) = \phi(\mathbf{x};\mathbf{\theta})^{\intercal} \mathbf{w}$.  Have parameters $\mathbf{\theta}$ used to learn $\phi$ from a family of functions, and $\mathbf{w}$ for the actual linear regression part.  This is a feedforward neural network with one hidden layer.  This makes the optimisation problem harder (no longer convex) but we can use numerical methods

### Learning XOR

  * The target function is $y = f^\*(\mathbf{x})$ where $\mathbf{x}$ is a binary pair and $f^\*$ is `XOR`.  Our model provides a function $y = f(\mathbf{x}; \theta)$ and our learning algorithm will adapt the parameters to give a good approximation to $f^\*$.
    - There is no statistical generalisation here, we can just try to get a perfect score on the training set $X = ((0,0), (0,1), (1, 0), (1, 1))$
    - To make things simple, we model it as a regression problem with MSE loss function, so $J(\mathbf{\theta}) = \frac{1}{4} \sum\_{\mathbf{x} \in X} (f^\*(\mathbf{x})-f(\mathbf{x};\mathbf{\theta}))^2$
    - If we choose a linear model $f(\mathbf{x}; \mathbf{w}, b) = \mathbf{x}^T \mathbf{w} + b$ and minimise in the usual way with the normal equations, we end up with $\mathbf{w}=\mathbf{0}$ and $b=\frac{1}{2}$.  Clearly a linear function cannot learn `XOR`
    - Instead we want to transform feature space to a different space where a linear model becomes appropriate.  Introduce a single hidden layer $f^{(1)}(\mathbf{z}; \mathbf{W}, \mathbf{c})$, keep the linear regression output layer $y = f^{(2)}(\mathbf{h}; \mathbf{w}, b)$, so the whole model is $f(\mathbf{x}; \mathbf{W}, \mathbf{c}, \mathbf{w}, b) = f^{(2)}(f^{(1)}(\mathbf{x})))$
    - Clearly $f^{(1)}$ must be non-linear here.  Most neural networks use an affine transformation controlled by learning parameters, followed by a fixed non-linear activation function.  We do that here, with $\mathbf{h} = g(\mathbf{W}^T\mathbf{x}+\mathbf{c})$.  The activation function $g$ is typically chosen to be a function that is applied element-wise (i.e. $h\_i = g((\mathbf{W}^T\mathbf{x})\_{i} + c\_i)$).  The _rectified linear unit_ (ReLU) $g(z) = \max(0, z)$ is a common choice which we use here
    - The whole thing is now $f(\mathbf{x}; \mathbf{W}, \mathbf{c}, \mathbf{w}, b) = \mathbf{w}^T\max(\mathbf{0}, \mathbf{W}^T\mathbf{x}+\mathbf{c}) + b$.  At this point, can find a solution by inspection.  Of course, we generally wouldn't be able to see a solution immediately, so we would have to do the gradient optimisation algorithm and get some approximation

### Gradient-Based Learning

  * Biggest difference between feedforward neural networks and previous examples is the non-convexity of the loss function.  This means a numerical gradient method is a must
  * Moreover, size of the training set means that we have to use _stochastic_ gradient descent.  Stochastic gradient descent doesn't have any guarantees on convergence to minimum
  * Computing the gradient looks like it might be expensive, but we'll cover the _backpropagation algorithm_ for that
  * Use regularization techniques as well.  E.g. weight decay can be applied pretty much verbatim, there are also other ones we'll cover later
  * Cost function:
    - In most cases, we have a parametric set-up and can calculate $p(\mathbf{y}|\mathbf{x}, \mathbf{\theta})$, and we can use maximum likelihood
      - As usual, the equivalent cost function is $$J(\mathbf{\theta}) = -\mathbb{E}\_{(\mathbf{x}, \mathbf{y}) \sim \widehat{p}\_{\text{data}}} \left[\log p\_{\text{model}}(\mathbf{y}|\mathbf{x})\right]$$
      - The exact form depends on $p\_\text{model}(\mathbf{y}|\mathbf{x},\theta)$.  If we work with $p\_{\text{model}}(y|\mathbf{x},\mathbf{\theta}) = \mathcal{N}(y;f(\mathbf{x},\mathbf{\theta}),\mathbf{I})$ then we get mean squared error cost again
      - Maximum likelihood is good, because you don't have to ad hoc construct a cost function, it just comes out of your model
      - Gradient of cost function must be "large and predictable" enough.  There is a risk for some of the units in the network to flatten the gradient, and it's difficult to be sure you're optimising correctly if the gradient is flat.  Log likelihoods help as they unflatten by undoing the exponentation in some units.  This flattening phenomenom is called _saturation_
      - Usually the cross-entropy function in MLE does not actually have a minimum in these cases (think of the exponentials in ReLUs etc.) so some regularisation is needed to stop things getting carried away
    - In some cases, we only need to predict some statistic of $\mathbf{y}$ conditional on $\mathbf{x}$, e.g. have a predictor $f(\mathbf{x}; \mathbf{\theta})$ that we want to use to predict the mean of $\mathbf{y}$
      - Using the calculus of variations, we can find the solution to $$f^\* = \text{argmin}\_f \left[\mathbb{E}\_{(\mathbf{x}, \mathbf{y}) \sim \widehat{p}\_\text{data}} \left[\mathbf{y}-f(\mathbf{x})\right]^2\right]$$ is $$f^\*(x) = \mathbb{E}\_{\mathbf{y} \sim \widehat{p}\_{\text{data}}(\mathbf{y}|\mathbf{x})} [\mathbf{y}].$$  To make sense of this, think of having a really large dataset where you have a bunch of different predictions for $\mathbf{x}$; this is saying that your best bet prediction is to predict the mean of those different sample predictions
      - On the other hand, if we instead minimise the $L^1$ norm (_mean absolute error_), we recover a function that predicts the _median_ value $\mathbf{y}$ for each $\mathbf{x}$
      - These don't give great results apparently, seems to be common to predict the full distribution even if you just want the mean at the end of the day
  * Output units:
    - Choice of output unit is coupled to cost function, e.g. if you're using cross-entropy then the output units determine the form of the cross-entropy function
    - Assume hidden layers give $\mathbf{h} = f(\mathbf{x}; \mathbf{\theta})$
    - For a linear output, can use a linear unit $\widehat{\mathbf{y}} = \mathbf{W}^{\intercal}\mathbf{h}+\mathbf{b}$
      - Typically assume Gaussian noise as well, $p(\mathbf{y}|\mathbf{x}) = \mathcal{N}(\mathbf{y};\widehat{\mathbf{y}},\mathbf{I})$, and then we're back in the familiar maximise log-likelihood / minimise MSE situation
       - Could also try to learn the covariance matrix, but since it has to remain positive definite the optimisation is a bit more tricky.  We'll see other units used to parametrise covariance 
    - Now suppose you had a multinoulli output (including Bernoulli, when $n=2$)
      - Take a MLE approach again.  Want to produce $\mathbf{\widehat{y}}$, with $\widehat{y}\_i = P(y=i|\mathbf{x})$ the most likely given the data (i.e., minimising the cross-entropy with the empirical distribution)
      - We have a linear layer predicting $\mathbf{z} = \mathbf{W}^{\intercal}\mathbf{h}+\mathbf{b}$, and we interpret these as unnormalized log probabilities $z\_i = \log \tilde{P}(y=i|\mathbf{x})$
      - Then apply the softmax function to get our actual guesses for probabilities, $p(y=i|\mathbf{x}) = \text{softmax}(\mathbf{z})\_i = \frac{\exp(z\_i)}{\sum\_j \exp(z\_j)}$
      - The cost function is then $$J(\theta) = -\mathbb{E}\_{(\mathbf{x}, y) \sim \widehat{p}\_{\text{data}}} \left[z\_i - \log\left(\sum\_j \exp(z\_j)\right)\right].$$  Note that $\log\left(\sum\_j \exp(z\_j)\right)$ is roughly the max of the $z\_j$, so when we are tweaking the $z\_i$ to minimise the cost function, we are pushing $z\_i$ up and pushing the loudest incorrect prediction $z\_j$ down
      - Log-likelihood works well, but any objective function which does not use a $\log$ to undo the $\exp$ in the softmax will not work well
      - Softmax itself can also be translated to make it more numerically stable $\text{softmax}(\mathbf{z}) = \text{softmax}(\mathbf{z}-\text{max}\_i(z\_i))$
      - Softmax is a misleading name, it's more of a softargmax, i.e. it's a soft version of something that picks the _index_ which maximises, it does not pick the maxiumum _value_
    - In general, we can view the hidden layers of the neural network as outputting $\mathbf{\omega} = f(\mathbf{x}; \theta)$, and we regard this as the parameters for the distribution at our output layer.  Then we have $p(y|\mathbf{\omega})$, and MLE says we should use $-\log p(y|\mathbf{\omega}(\mathbf{x}))$ as our cost function
      - Consider learning variance of $\mathbf{y}$ given $\mathbf{x}$.  This is easy if $\sigma^2$ is constant (indeed, the MLE is just the empirical variance),  but if we don't make that assumption, we can just say that variance of $\mathbf{y}$ is something controlled by $\mathbf{\omega} = f(\mathbf{x};\mathbf{\theta})$.
        - If $\sigma$ does not depend on $\mathbf{x}$ (homoscedastic case), we can just include $\sigma$ as an additional term in $\mathbf{\omega}$, nothing to do with the previous layers going on
        - If $\sigma$ does depend on $\mathbf{x}$ (heteroscedastic case), then $\sigma$ is actually output by the hidden layers
      - When actually learning $\sigma$, it's probably helpful to rephrase it in terms of learning the precision matrix instead
      - We have a constraint that covariance (or precision) matrix be positive definite.  If we assume it's diagonal, then we just need the diagonal entries to be positive, which we can ensure by inserting a softplus function.  Non-diagonal covariance matrices are rare.  (Presumably you have tried to decorrelate features before starting any of this, but maybe this is more complicated in the heteroscedastic case?)
      - For multimodel regression (predict real values from a multimodal $p(\mathbf{y}|\mathbf{x})$), popular to use Gaussian mixtures for the output, $p(\mathbf{y}|\mathbf{x}) = \sum\_{i=1}^n p(c=i|\mathbf{x}) \mathcal{N}(\mathbf{y}; \mathbf{\mu}^{(i)}(\mathbf{x}), \mathbf{\Sigma}^{(i)}(\mathbf{x}))$.  A neural network with Gaussian mixture output is called a _mixture density network_
      - There is a bunch of stuff to learn here: a multinoulli distribution $p(c|\mathbf{x})$, a matrix of means, and a tensor of covariances (3D in theory, though only a 2D tensor if you impose diagonality).  Learning the latter two is complicated by having to consider which Gaussian your sample came from
      - Potential for numerical instability, _clip gradients_ might help
      - Gaussian mixture models are effective in e.g. generative models of speech

### Hidden Units

  * Hidden units are often not differentiable at a small number of points.  Inherent numerical approximation, the isolatedness, and the fact that they usually have a left and right derivative means that this isn't really a problem when working with the gradient
  * Use the general notation: hidden units take input $\mathbf{x}$, compute affine tranformations $\mathbf{z} = \mathbf{W}^T\mathbf{x} + \mathbf{b}$, and then apply some non-linear function elementwise to $\mathbf{z}$
  * ReLU use the activation function $g(z) = \max(0, z)$.  One drawback is they cannot learn (via gradient-based methdos) on examples for which their activation is zero
    - Generalize to $g(z, \mathbf{\alpha})\_i = \max(0, z\_i) + \alpha\_i \min(0, z\_i)$ to capture some this stuff
    - Setting $\alpha\_i = -1$ gives _absolute value retification_.  This is used in object recognition, where you want to reflect (lol) invariance to polarity reversal
    - _Leaky ReLU_ sets $\alpha\_i = 0.01$ or something else small
    - _Parametric ReLU_ learns $\mathbf{\alpha}$
    - _Maxout units_ are another generalisation, $g(z)\_i = \max\_{j \in \mathbb{G}^{(i)}} z\_j$, where $\mathbb{G}^{(i)}$ is the $i$th block when we split all the indices into blocks of $k$.  There's a fairly long discussion about this but I think it's just because the author invented them :)
  * Can use either the logistic sigmoid $g(z) = \sigma(z)$ or the hyperbolic tangent $g(z) = \tanh(z)$ as activation functions.  (Note $\tanh(z) = 2\sigma(2z)-1$).
    - Sigmoid units are probably not a good idea as a hidden unit.  The gradient saturates away from zero quite dramatically, and the only way we got away with using them as output layers was by using a log-likelihood cost function to undo the exponentiation
    - If you do want to use something like this as a hidden layer, $\tanh$ is better as it satisfies $\tanh(0)=0$ and generally looks a bit like the identity near $0$, so jamming it on top of a linear transformation doesn't do any weird translation
    - Sigmoidal activation functions are more useful in recurrent neural networks for giving feedback, guess we'll see some of that later
  * There is a lot of flexibility in the hidden units you choose.  E.g. they did an experiment training a feedforward network on MNIST with $\cos$ and got error rate less than one percent
  * Quote: "new hidden unit types that perform roughly comparably to known types are so common as to be uninteresting"
  * Some examples anyway:
    - Can just use the identity function.  Having a purely linear layer feed in to the next layer vs. just having one layer is basically having a matrix that factors vs. allowing general matrices.  Factorisation saves on parameters and often the factorisability assumption is reasonable
    - Softmax units can be used as hidden units as well.  These act as a switch and apparently are used in memory networks
    - Radial basis function: $h\_i = \exp(-\frac{1}{\sigma\_i^2} (\mathbf{W}\_{\cdot, i} - \mathbf{x})^2)$.  Becomes active as $\mathbf{x}$ approaches the template $\mathbf{W}\_{\cdot, i}$.  Difficult to optimize as it saturates to zero away from the template
    - Softplus: $g(a) = \zeta(a) = \log(1 + e^{a})$.  This is similar to ReLU, but smooth.  It has been proven to perform worse than ReLU, though, so might as well just use ReLU
    - Hard tanh: $g(a) = \max(-1, \min(1, a))$.  This is shaped similary to $\tanh$ and the ReLU, but it is bounded

### Architecture Design

  * Basic architectural decisions are: depth of the network?  width of each layer?  activation functions at each layer?
  * Universal approximation theorem: a feedforward network with a linear output layer and at least one hidden layer with any "squashing" activation function (e.g. logistic sigmoid) can approximate (to given accuracy $\epsilon>0$) any measurable function $\mathbb{R}^m \to \mathbb{R}^n$ (any $m$ and $n$), provided the network is given enough hidden units.  Can also approximate its derivative.  Can also approximate functions between finite dimensional discrete spaces
  * This says that there exists a network that is good, it does not mean that your chosen algorithm will be able to _learn_ a network that is good
  * Also it doesn't say how large the network has to be.  There are some bounds on the size of the approximating networks for specific classes of functions, but the bounds are very large
  * Interesting trade-off between depth of the network and width of the layers.  Technically you only need depth one to approximate well, but author seems to be saying that if you choose depth smaller than some $d$ (depending on the function you're approximating), then you need a number of units exponential in the number of features (i.e., few very wide layers).  If you choose depth larger than $d$, you need a smaller number of hidden units overall.  Refers to lots of research here
  * Also refers to lots of research showing the deep neural networks just seem to be empirically better
  * Most convincing theoretical argument is: you justify the depth by saying that you're making the prior assumption that you're trying to model something which is a composition of many different stages
  * What kind of neural network (CNN?  RNN?) is also an architecture choice
  * Don't necessarily have to connect the layers in a chain
  * Don't necessarily have to have a all units in one layer connected to all units in another layer, e.g. CNNs have a specific pattern of sparse connections between layers

### Backpropagation and Other Differentiation Algorithms

  * Giving an analytical expression for the gradient is straightforward, but actually computing it is expensive.  Backpropagation gives a cheaper way to evaluate the gradient numerically
  * Backpropagation only refers to how to compute the gradient, to get a learning algorithm on top of that you will need some gradient-based optimisation, like stochastic gradient descent
  * Fairly general, can compute $\nabla\_\mathbf{x} f(\mathbf{x}, \mathbf{y})$ for an arbitrary function $f$, where $\mathbf{x}$ is a set of variables that we want to differentiate along, and $\mathbf{y}$ are additional variables that we don't care about differentiating along
  * Talk about neural networks in terms of computation graphs:
    - A node in the graph represents a variable (could be scalar, vector, matrix, tensor, ...)
    - Have _operations_ which (WLOG) output a single variable (which, as above, could be a vector)
    - If $y$ is obtained by applying an operation to $x$, draw a directed edge from $x$ to $y$ (possibly annotating to say what the operation actually was)
  * Chain rule for vector-valued functions: if $\mathbf{x} \in \mathbb{R}^m$, $\mathbf{y} = g(\mathbf{x}) \in \mathbb{R}^n$, and $z = f(\mathbf{y})$, then $\nabla\_\mathbf{x} z = \left(\frac{\partial \mathbf{y}}{\partial \mathbf{x}} \right)^{\intercal} \nabla\_\mathbf{y} z$.  I.e., the gradient wrt $\mathbf{x}$ can be obtained by multiplying a Jacobian by the gradient wrt $\mathbf{y}$
  * Let $\mathbf{X}$ be a tensor, and write $\nabla\_\mathbf{X} z = (\nabla\_\mathbf{X} z)\_i$ for the tensor gradient, where $i$ is now a multi-index
  * Chain rule for tensor-valued functions: $\nabla\_\mathbf{X} z = \sum\_j (\nabla\_\mathbf{X} \mathbf{Y}\_j) \frac{\partial z}{\partial \mathbf{Y}\_j}$.  Here the variable $j$ will be range over all the ways you can index into $\mathbf{Y}$
  * This gives us the way to write down an algebraic expression for the gradient of a scalar wrt any node (=variable) in the computational graph, but it is expensive to evaluate
  * Can imagine that chained rule for multiply composed functions leads to lots of repeated terms - need to decide whether to recompute (costs CPU) or store (costs memory)
  * Given output scalar $u^{(n)}$, want to compute its derivative with wrt input nodes $u^{(1)},...,u^{(n\_i)}$.  Also have remaining nodes $u^{(n\_i+1)},...,u^{(n)}$.  Assume they're ordered in the obvious way.
  * Forward computation algorithm:
    - Put your input values $(x\_1,...,x\_{n\_i})$ into the nodes $u^{(1)},...,u^{(n\_i)}$.
    - Then for $n = n\_{i+1}, ..., n$, set $\mathbb{A}^{(i)}$ to be the set of $u^{(j)}$ s.t. $j \in \text{Pa}(u^{(i)})$, then calculate $u^{(i)} = f^{(i)}(\mathbb{A}^{(i)})$.  This notation is bad, but it's just trying to distinguish between a node and the value output from it
  * Back propagation, simple case:
    - Assume all variables scalars, want to compute derivative of $u^{(n)}$ wrt $u^{(1)},...,u^{(n\_i)}$.
    - Run forward propogation to compute the activations
    - Initialise $\text{gradtable}$, we will set $\text{gradtable}(u^{(i)})$ equal to the computed value of $\frac{\partial u^{(n)}}{\partial u^{(i)}}$
    - First set $\text{gradtable}(u^{(n)}) = 1$
    - Then for $j=n-1,...,1$, compute $\frac{\partial u^{(n)}}{\partial u^{(j)}} = \sum\_{i \text{ s.t. } j \in \text{Pa}(u^{(i)})} \frac{\partial u^{(n)}}{\partial u^{(i)}} \frac{\partial u^{(i)}}{\partial u^{(j)}} = \sum\_i \text{gradtable}(u^{(i)}) \frac{\partial u^{(i)}}{\partial u^{(j)}}$
    - When done, return the grad table
  * Forward propagation for a typical deep neural network (minibatch size one):
    - Have network of depth $l$, $W^{(1)},...,W^{(l)}$ the weights, $b^{(1)},...,b^{(l)}$ the biases, input $\mathbf{x}$, and target $\mathbf{y}$
    - Initialise $\mathbf{h^{(0)}} = \mathbf{x}$
    - For $k=1,...,l$, do $\mathbf{a}^{(k)} = \mathbf{b}^{(k)} + \mathbf{W}^{(k)} \mathbf{h}^{(k-1)}$, then $\mathbf{h}^{(k)} = f(\mathbf{a}^{(k)})$
    - Set $\widehat{\mathbf{y}} = \mathbf{h}^{(l)}$ and $J = L(\widehat{\mathbf{y}}, \mathbf{y}) + \lambda \Omega(\theta)$ (i.e. cost = loss + regularization)
  * Back propagation for a typical deep neural network:
    - Run the above forward propagation
    - Set $\mathbf{g} = \nabla\_\mathbf{\widehat{\mathbf{y}}} J = \nabla\_{\widehat{\mathbf{y}}} L(\widehat{\mathbf{y}}, \mathbf{y})$
    - For $k=l,...1$:
      - First replace the gradient $\mathbf{g}$ with the derivative of the pre-nonlinearity action $\nabla\_{\mathbf{a}^{(k)}} J = \mathbf{g} \cdot f'(\mathbf{a}^{(k)})$.
      - Compute the gradient on the weights and biases: $\nabla\_{\mathbf{b}^{(k)}} J = \mathbf{g} + \lambda \nabla\_{\mathbf{b}^{(k)}} \Omega(\theta)$, $\nabla\_{\mathbf{W}^{(k)}} J = \mathbf{g} \mathbf{h}^{(k-1)T} + \lambda \nabla\_{\mathbf{W}^{(k)}} \Omega(\theta)$
      - Backpropagate one stage: replace $\mathbf{g}$ with $\nabla\_{h^{(k-1)}} = \mathbf{W}^{(k)}{\intercal} \mathbf{g}$
    - At the end of the day, we have the gradients of $J$ wrt all the $\mathbf{W}^{(k)}$ and the $\mathbf{b}^{(k)}$, so we can do descent
  * One approach to backpropagation in general is to take a computation graph and a set of numerical inputs to the graph, and return numerical values giving the gradient at those input values.  Call this _symbol-to-number differentiation_.  Implementations in Torch and Caffe are like this
  * Alternative is to take the computational graph and add additional nodes explaining how to compute the derivatives.  This is the approach taken by Theano and Tensorflow.  To actually evaluate, you plug your concrete inputs into the new computational graph.  One advantage of this approach is you can now compute second order derivatives by feeding in the first order derivative graph, etc.
  * General set-up:
    - Each node in the graph corresponds to a tensor $\mathbf{V}$
    - For each operation (edge in the computational graph) $\text{op}$, we have a corresponding $\text{bprop}$, defined as follows:  Take any operation $\text{op}$ with inputs $\text{inputs}$, write $\text{op.f}$ for the corresponding mathematical function.  Let $\mathbf{G}$ be the gradient on the output of our operation, and let $\mathbf{X}$ be any input to the operation that we want to compute the gradient of (i.e., we want to step backwards to).  The corresponding backprop $\text{op.bprop}$ satisfies $\text{op.bprop}(\text{inputs}, \mathbf{X}, \mathcal{G}) = \sum\_i (\nabla\_\mathbf{X} \text{op.f}(\text{inputs})\_i)\mathbf{G}\_i$
    - C.f. the chain rule: if $\mathbf{X} \mapsto \mathbf{Y} \mapsto z$, then $\nabla\_\mathbf{X} z = \sum\_i (\nabla\_\mathbf{X} Y\_i) \frac{\partial z}{\partial Y\_i}$.  So the backprop calculates $\nabla\_\mathbf{X} z$, from which we obviously get each $\frac{\partial z}{\partial X\_i}$ and hence can recurse
    - Proceeds to write out the algorithm fully but I think it's clear now
  * If the forward graph has $n$ nodes, then the backpropagation is at worst $O(n^2)$ (as it's directed acyclic).  For neural networks, it's going to be $O(n)$ as the graph is a chain.  Of course, the complexity is in terms of number of operations as defined by your graph, if those operations are "multiply two very large matrices together" then it's still going to be slow
  * TL;DR of backpropagation is "use dynamic programming"
  * History lesson:
    - Initially people used neural network models but they were entirely linear (so they were just ways of representing linear models).  There was pushback (from e.g. Minsky) because these are not even capable of learning `XOR`
    - People added nonlinearity and worked out how to compute the gradient, including using dynamic programming methods, in the 60s and 70s
    - Some good applications and backpropagation meant continued popularity (with people like Hinton and Le Cun getting mentions) until the early 90s
     -  Other ML techniques took over for a while, until neural networks became fashionable again with deep neural networks in 2006.  Mentions larger datasets and better computers as major factors here
      - Algorithmically, mentions replacing MSE with cross-entropy as an improvement.  Also a good amount on how replacing sigmoids with ReLUs improved performance - people preferred sigmoids for a while as they are everywhere differentiable and work pretty well on small networks, but ReLUs on larger networks have shown notable performance improvements

## Chapter 7 - Regularization for Deep Learning

  * General statement: we often find that the best fitting model (in the sense of minimising generalisation error) is a large model (e.g. deep neural network) that has been regularised appropriately

### Parameter Norm Penalties

  * Parameter norm penalties: $\tilde{J}(\theta; \mathbf{X}, \mathbf{y}) = J(\theta; \mathbf{X}, \mathbf{y}) + \alpha \Omega(\theta)$ for some $\alpha \geq 0$
  * Typically in DL, $\Omega$ only penalises the weights of the affine transformation, and not the biases (regularising the biases can cause underfitting).  Write $\mathbf{w}$ for the weights, and $\mathbf{\theta}$ for all the parameters (which includes both the weights and the unregularized parameters)
  * Common parameter penalty is L2 norm, variously called weight decay / ridge regression / Tikhonov regularisation
  * Local effects of L2: For simplicity consider the case where there are no biases (so $\theta = \mathbf{w})$):
    - Then $\tilde{J}(\mathbf{w}; \mathbf{X}, \mathbf{y}) = \frac{\alpha}{2} \mathbf{w}^{\intercal} \mathbf{w} + J(\mathbf{w}; \mathbf{X}, \mathbf{y})$
    - And $\nabla\_\mathbf{w} \tilde{J}(\mathbf{w}; \mathbf{X}, \mathbf{y}) = \alpha \mathbf{w} + \nabla\_\mathbf{w} J(\mathbf{w}; \mathbf{X}, \mathbf{y})$
    - So we see that gradient descent at rate $\epsilon$ now combines two tasks: lowering the gradient of $J$, and decreasing the weights by a constant multiplicative factor $1-\epsilon \alpha$
  * Global effects of L2: take a quadratic approximation of the cost function around the point $\mathbf{w}^\*$ where $J(\mathbf{w}; \mathbf{X}, \mathbf{y})$ is minimised.  We find the minimum of the appromximation is $\widetilde{\mathbf{w}} = Q(\Lambda + \alpha \mathbf{I})^{-1}\Lambda Q^{\intercal} \mathbf{w}^\*$, where $H = Q\Lambda {}^tQ$ diagonalizes the Hessian ($Q$ being orthogonal, as $H$ is real and symmetric).  If $\alpha=0$ this recovers the orginal minimum $\mathbf{w}^\*$, but in general it scales the weights down by $\lambda\_i/(\lambda\_i + \alpha)$ along the eigenvectors of $H$
    - This makes sense: if an eigenvalue of the Hessian is small, then that means that the gradient does not vary much along the direction of the corrresponding eigenvector.  You want your model to assign a low weight to that direction, otherwise you are overfitting
  * Another option is L1 regularisation.  In the simplest no-biases case from above, $\widetilde{J}(\mathbf{w}; \mathbf{X}, \mathbf{y}) = \alpha ||\mathbf{w}||\_1 + J(\mathbf{w}; \mathbf{X}, \mathbf{y})$ and $\nabla\_\mathbf{w} \widetilde{J}(\mathbf{w}; \mathbf{X} \mathbf{y}) = \alpha \text{sgn}(\mathbf{w}) + \nabla\_\mathbf{w} J(\mathbf{w}; \mathbf{X}, \mathbf{y})$
    - Clearly this is quite different to L2 regularisation: now regularisation only depends on the sign of $w\_i$, not the actual size of $w\_i$
    - Since the analysis is harder here, further simplify by assuming the Hessian is diagonal (can achieve this by applying PCA to remove all correlation between input features as a preprocessing step)
    - Take a quadratic approximation again.  Still assuming the Hessian is diagonal, we now get $\widehat{J}(\mathbf{w}) = J(\mathbf{w}) + \sum\_i \left(\frac{1}{2} H\_{i, i} (w\_i - w\_i^\*)^2 + \alpha |w\_i|\right)$
      - This is minimised when $w\_i = \text{sgn}(w\_i^\*) \max\left(|w\_i^\* - \frac{\alpha}{H\_{i,i}}|, 0 \right)$
      - So if $|w\_i^\*| H\_{i,i}$ is small, then we simply replace that weight by zero.  If it is large, we shorten that weight by distance $\frac{\alpha}{H\_{i,i}}$
    - L1 regularisation therefore moves to a solution that is more _sparse_.  This can be used as a _feature selection_ mechanism, e.g. in LASSO (L1 penalty with least-squares cost function)
  * We previously saw that L2 regularisation was equivalent to MAP Bayesian inference with a Gaussian prior on the weights.  For L1 regularisation, there is a MAP perspective where the prior is an isotropic Laplacian distribution 

### Norm Penalities as Constrained Optimization

  * Suppose you wanted to impose $\Omega(\theta) \leq k$.  Then you could set up a Lagrangian $\mathcal{L}(\theta, \alpha) = J(\theta) + \alpha(\Omega(\theta)-k)$.  The solution is then $\theta^\* = \text{argmin}\_\theta \max\_{\alpha\geq 0} \mathcal{L}(\theta, \alpha)$
  * From this, if we can find $\alpha^\*$ minimising, we get the same form as before.  We could also use a different optimisation algorithm, e.g. do normal stochastic gradient descent then project back to the nearest point satisfying $\Omega(\theta) \leq k$

### Regularization and Under-Constrained Problems

  * Regularisation is sometimes necessary for the problem to be defined.  E.g. in linear regression and PCA, you need to invert $\mathbf{X}^{\intercal} \mathbf{X}$, which might not be possible.  But if you regularise, you are inverting $\mathbf{X}^{\intercal} \mathbf{X} + \alpha \mathbf{I}$, which will be fine for almost all $\alpha$.  This is basically the idea of the Moore--Penrose pseudoinverse
  * Another example: logistic regression, where we find $\mathbf{w}$ which gets a perfect score.  Then $2\mathbf{w}$ also gets a perfect score but has higher likelihood.  Regularisation at least stops this pattern continuing at some point

### Dataset Augmentation

  * Data augmentation can sometimes help reduce generalisation error as well
    - E.g. in object recognition, augment the data by appyling various translations etc. to the input but keep the class the same, makes the network more robust
    - Similarly injecting noise - neural networks can be sensitive to noise, and one workaround is train them on artificially noised data

### Noise Robustness

  * Noise can also be added to the hidden units, will cover this in more details when we look at the dropout algorithm
  * Consider a linear regression problem with training data $(\mathbf{x}\_1, y\_1),...,(\mathbf{x}\_n, y\_n)$ where we are minimising the MSE.  We use a neural network and include a random perturbation $\mathbf{\epsilon}\_\mathbf{W} \sim \mathcal{N}(\mathbf{\epsilon}; \mathbf{0}, \eta \mathbf{I})$ of the weights, and write $\mathbf{y}\_{\mathbf{\epsilon}\_\mathbf{W}}$ for the perturbed model.
    - **TODO** I don't understand this example, maybe reading this [paper](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=2&cad=rja&uact=8&ved=2ahUKEwjp6PbXmtfkAhXnRBUIHeQwD7kQFjABegQIBBAC&url=https%3A%2F%2Fpapers.nips.cc%2Fpaper%2F899-simplifying-neural-nets-by-discovering-flat-minima.pdf&usg=AOvVaw2U8yZT7vC8xRDmw4Dn6kWL) will help
  * The training data probably has mistakes, and if $(\mathbf{x}, y)$ is misclassified then maximising $\log(p(y|\mathbf{x}))$ is harmful.  Adding noise to the labels can help mitigate this, e.g. could say that labels are correct with probability $1-\epsilon$
  * _Label smoothing_ replaces a $k$-output softmax, by replacing the hard 0 and 1 with $\epsilon/(k-1)$ and $1-\epsilon$ respectively.  Direct softmax has convergence problems (softmax itself never actually reaches 0 or 1, so need weight decay to stop it getting carried away during training), but smooth version works better

### Semi-supervised Learning

  * _Semi-supervised learning_: use both unlabelled samples from $P(\mathbf{x})$ and labelled examples from $P(\mathbf{x}, \mathbf{y})$ to predict $P(\mathbf{y}|\mathbf{x})$.
  * In DL, semi-supervised learning usually refers to learning a representation $\mathbf{h} = f(\mathbf{x})$, so that samples from the same class have a similar representation (then apply a basic classifier on top of that)
  * Just discussed very briefly here, see [Chapelle et. al](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=2&cad=rja&uact=8&ved=2ahUKEwjb0eLrndfkAhWMTcAKHT_jBQsQFjABegQIBhAC&url=http%3A%2F%2Fwww.acad.bg%2Febook%2Fml%2FMITPress-%2520SemiSupervised%2520Learning.pdf&usg=AOvVaw3UO5UovSG0OzG7c8CHQ7Mo) for more details

### Multitask Learning

  * _Multitask learning_: improve generalisation by pooling the examples arising out of several tasks
    - E.g. Imagine tasks sharing common input $\mathbf{x}$, we might produce a shared representation $\mathbf{h}^{\text{shared}}$, then specialising in to two different hidden layers $\mathbf{h}^{(1)}$ from which we predict $\mathbf{y}^{(1)}$ and $\mathbf{h}^{(2)}$ from which we predict $\mathbf{y}^{(2)}$
    - The prior belief this is expressing is "among the factors that explain the variations observed in the data associated with the different tasks, some are shared across two or more tasks"

### Early Stopping

  * When training a model with large representational capacity (i.e., ability to overfit), we often see the training error decreasing steadily (albeit more slowly) as we train longer, but the validation error (and so probably the test error) might start increasing again after a while
  * Better to exit early in this case.  Can use the _early stopping_ algorithm: keep a copy of the parameters that had the best validation error, and to exit once we have performed $N$ iterations of the training phase without bettering the validation error
    - Early stopping is nice because it doesn't affect the learning dynamics.  In weight decay, you have to worry about things like whether you have made $\lambda$ to large.  OTOH, early stopping slots in quite nicely to what you're already doing
    - Early stopping can be used in conjunction with other regularisation techniques, of course
  * Early stopping requires a training/validation data split.  Once you've used early stopping to determine the optimal training time, you can feed the validation data back into the training data
    - One way is to take the new combined data set, and train for as long as early stopping told you to
    - Another way is to stack it on top of the work you did during early stopping - use what you already had, and chuck the validation data in as well.  Keep going for a while until things seem to stop improving.  This avoid retraining but has obvious downsides as well
  * Intuitively, early stopping is a regulariser because of the U-shape of the validation error curve.  Stopping at the trough of the U is limiting the distance we travel during gradient descent, so it is a means of regularising our parameters
  * For a simple linear model with quadratic error function and simple gradient descent, early stopping is equivalent to L2 regularisation:
    - Assume for simplicity the only parameters $\mathbf{\theta}$ are the weights $\mathbf{w}$
    - Take a quadratic approximation $\widehat{J}$ of the cost function $J$ in a neighbourhood of the empirically optimal point $\mathbf{w}^\*$, so $\widehat{J}(\mathbf{w}) = J(\mathbf{w}^\*) + \frac{1}{2} (\mathbf{w}-\mathbf{w}^\*)^{\intercal}H(\mathbf{w}-\mathbf{w}^\*)$
    - Then $\nabla\_\mathbf{W} \widehat{J}(\mathbf{w}) = H(\mathbf{w}-\mathbf{w}^\*)$
    - Write $(\mathbf{w}^{(\tau)})\_{\tau \geq 1}$ for the points we step on during gradient descent.  Then it's easy to see that $\mathbf{w}^{(\tau)}-\mathbf{w}^\* = (\mathbf{I}-\epsilon H)(\mathbf{w}^{(\tau-1)} - \mathbf{w}^\*)$
    - Diagonalising $H = Q\Lambda Q^{\intercal}$ with $Q$ orthonormal, we get $Q^{\intercal} (\mathbf{w}^{\tau} - \mathbf{w}^\*) = (\mathbf{I}-\epsilon \Lambda) Q^{\intercal}(\mathbf{w}^{(\tau-1)} - \mathbf{w}^\*)$, and hence $Q^{\intercal}w^{(\tau)} = (\mathbf{I} - (\mathbf{I}-\epsilon \Lambda)^\tau) Q^{\intercal} w^\* + (\mathbf{I}-\epsilon \Lambda)^\tau \mathbf{w}^{(0)}$
    - Assume $\mathbf{w}^{(0)}=0$ for simplicity, and assume that $\epsilon$ is small enough so that $|1-\epsilon \lambda\_i| < 1$, so $Q^{\intercal}\mathbf{w}^{(\tau)} = (\mathbf{I} - (\mathbf{I} - \epsilon \Lambda)^\tau) Q^{\intercal} \mathbf{w}^\*$
    - C.f. what we saw in L2 regularization, $\widetilde{\mathbf{w}} = Q(\Lambda + \alpha \mathbf{I})^{-1} \Lambda Q^{\intercal} \mathbf{w}^\*$.
    - Matching these up, the rough relation between these parameters is $\tau \approx 1/(\epsilon \alpha)$
    - The big advantage early stopping has over weight decay is that early stopping _chooses_ $\tau$, so whilst the approaches are roughly equivalent, there's no ad hoc choice of $\alpha$ but rather an algorithm for choosing $\tau$

### Parameter Tying and Parameter Sharing

  * Techniques like L2 regularisation and early stopping express a prior about where the parameters should like (near the origin, near where you started, etc.).  In some situations, though, our prior might not be a specific region for where the parameters should live, but instead something like a relationship between weights
  * Can imagine two classifiers with different input distributions and different output classes, but are doing basically the same job so might expect that the two models have similar weight vectors once trained optimally.
    - One way to express this prior (weights of my two different classifiers are related) is to train one then then regularize the second towards the weights of the first
    - Another way is to literally force sets of parameters to be equal when training, this is _parameter sharing_, which saves on some memory at least.  Parameter sharing is widely used in CNNs (will cover later)

### Sparse Representations

  * We have already seen how L1 penalisation induces sparse parameterisations, i.e. forces many of the weights to become zero.  In $\mathbf{h'} = \mathbf{W}\mathbf{h}$, this is basically making a bunch of entries in $\mathbf{W}$ zero
  * Sparsity can also be induced in the representation, i.e. making a bunch of entries of $\mathbf{h}$ be zero
  * _Sparse representations_ can be achieved in exactly the same way as we got sparse parameterisations: add a penalty (this time for the representation) to the cost function, $\widetilde{J}(\mathbf{\theta}; \mathbf{X}, \mathbf{y}) = J(\mathbf{\theta}; \mathbf{X}, \mathbf{y}) + \alpha \Omega(\mathbf{h})$.  Can use $\Omega$ an L1 penalty similar to before to induce sparsity

### Bagging and Other Ensemble Methods

  * Bagging (boostrap aggregating) is a technique for reducing generalisation error by combining several models, by having those models vote on the output for a test example.  This is an example of _model averaging_, and fits within _ensemble methods_
  * Intuition for model averaging: consider training $k$ regression models, and write $\epsilon\_i$ for the errors they make.  Assume the errors are mean zero, variance $v$, and have covariance $\mathbb{E}[\epsilon\_i \epsilon\_j] = c$.  If you average the predictions, then the expected squared error is $\mathbb{E}\left[\left(\frac{1}{k} \sum\_i \epsilon\right)^2\right] = \frac{1}{k}(v + (k-1)c)$
    - If $c=v$ (the errors are perfectly correlated) then the averaging has no effect
    - If $c=0$ (the errors are perfectly uncorrelated) then we have improved the error by a factor of $k$
  * Different ensemble methods construct the models in different ways; bagging is one way
  * In bagging, you construct $k$ different datasets by sampling with replacement from the original dataset, and get $k$ models by training on those $k$ datasets
  * Model averaging is "an extremely powerful and reliable method for reducing generalization error".  Scientific papers usually don't have any averaging, because everyone knows you can make your results better by averaging
  * Quick mention of _boosting_, which is an ensemble method which does not aim at regularisation, but rather at increasing the capacity of the input models

### Dropout

  * Bagging involves training multiple models, then evaluating each model on a test sample (then e.g. taking the majority vote).  This is computationally expensive, dropout is a cheaper alternative
  * Training with dropout:
    - For each training sample, randomly generate a bitmask on the input and hidden units.  The randomness is independent of the sample, typically probability of including an input unit is $0.8$, and $0.5$ for hidden units.  For the units that are off, exclude them from the network (e.g. multiply the unit's output by zero).
    - During training, we minimise $\mathbb{E}\_{\mathbf{\mu}} J(\mathbf{\theta}, \mathbf{\mu})$, where $\mathbf{\mu}$ is the bitmask vector.  This expectation has exponentialy many terms, but we can approximate it by sampling $\mathbf{\mu}$
  * For both bagging and dropout, we can make predictions by taking a majority vote.  If we are just classifying, it's clear what to do here.  For predicting an actual probability disstribution, we need to be more specific
    - Bagging: each model $i$ gives a prediction $p^{(i)}(y|\mathbf{x})$, and to combine we just average $\frac{1}{k} \sum\_i p^{(i)}(y|\mathbf{x})$
    - Dropout: this time weight by the probability of the bitmask, $\sum\_{\mu} p(\mathbf{\mu}) p(y|\mathbf{x})$.  Obviously we can't compute this full sum, but again approximate by sampling $\mathbf{\mu}$.  Apparently 10-20 samples of $\mathbf{\mu}$ is enough to see good performance improvements
    - We can also take the _geometric_ mean, $\widetilde{p}(y|\mathbf{x}) = \left(\prod\_\mathbf{\mu} p(y|\mathbf{x}, \mathbf{\mu})\right)^{2^{-d}}$, where $d$ is the length of $\mathbf{\mu}$.  We impose that none of the conditional probabilities on the RHS are zero.  Note that the LHS is not normalised
    - We can approximate $\widetilde{p}(y|\mathbf{x})$ with a single evaluation of a specific model: namely compute $p(y|\mathbf{x})$ in the model where we take the usual neural net and the output of unit $i$ by the probability that unit $i$ is included.  This is the _weight scaling inference rule_
    - E.g. consider a softmax classifier $P(y|\mathbf{v}) = \text{softmax}(\mathbf{W}^{\intercal}\mathbf{v} + \mathbf{b})\_y$, and suppose we do dropout where each unit is included with probability $1/2$.  A bit of maths shows that the network with weights $\frac{1}{2}\mathbf{W}$ computes $\widetilde{p}(y|\mathbf{v})$ _exactly_. This is an example of weight scaling inference, though in general it only gives an approximation 
  * Dropout is good because it doesn't add much computational cost, and it does not affect the kind of network that can be used (e.g. dropout in RNNs is just fine)
  * However, since dropout is a regularisation technique, it does reduce the capacity of the model, and hence you are probably offsetting this by using a larger model in the first place.  Dropout does not do well when there are very few labelled training examples

### Adverserial Training

  * Idea is to search for examples that the network will misclassify, i.e. find $\mathbf{x'}$ near $\mathbf{x}$ such that the model output at $\mathbf{x'}$ is very different to that at $\mathbf{x}$.  We call $\mathbf{x'}$ the _adverserial example_
  * Example of an image $\mathbf{x}$ of a panda (correctly classified).  Replace with $\mathbf{x} + \epsilon \text{sgn}(\nabla\_\mathbf{x} J(\mathbf{\theta}, \mathbf{x}, y))$.  This makes no visual difference, but the classifer now classifies it as a gibbon
  * _Adverserial training_: training on adverserial perturbed examples from the training set
  * Neural network respond to adversarial training, effectively this forces the neural network function to be locally constant around training datapoints.  This is something that is possible as we are regularising a large function family.  You could not do something like this with logistic regression

### Tangent Distance, Tangent Prop and Manifold Tangent Classifier

  * Recall the manifold hypothesis: assuming that the data lies on a low dimensional manifold in input space (e.g. real world images inside all possible images)
  * _Tangent distance algorithim_: doing classification, assume that examples on the same manifold share the same category.  Do a nearest neighbours algorithm, but rather than looking at distance between points, look at the distance between their manifolds.  We can approximate this by approximation the manifold by the tangent plane.  This requires upfront knowledge about what the tangent plane is (which you get by e.g. knowing your image is invariant by translation, etc.)
  * _Tangent prop_: want to force your classifier to be invariant along the manifold along which samples in this class are concentrated, so add a regularisation term $\Omega(f) = \sum\_i \left((\nabla\_\mathbf{x} f(\mathbf{x}))^{\intercal} \mathbf{v}^{(i)}\right)$, where $\mathbf{v}^{(i)}$ are the manifold tangent vectors at $\mathbf{x}$.  Again this requires upfront knowledge about the manifold and its tangent plane
  * Compare adverserial training and tangent prop: former encourages the model to be invariant to small perturbations in all directions, latter encourages the model to be invariant to small perturbations in specified directions
  * _Manifold tangent classifier_: removes the need to specify the tangent directions upfront.  First it uses an autoencoder to learn the manifold structure by unsupervised learning, then uses these learned tangents in tangent prop.  Will cover in more detail later

## Chapter 8 - Optimization for Training Deep Models

### How Learning Differs from Pure Optimisation

  * Fundamental optimisation problem of DL is to find parameters $\mathbf{\theta}$ of a neural network that significantly reduce a cost function $J(\mathbf{\theta})$
  * Typically the cost function can be written as an average over the training set, $J(\mathbf{\theta}) = \mathbb{E}\_{(\mathbf{x}, y) \sim \widehat{p}\_{\text{data}}} L(f(\mathbf{x}; \mathbf{\theta}), y)$, where $L$ is a per-example loss function
    - We call this function the _emperical risk_.  Minimising this is _empirical risk minimisation_
    - For simplicity, we focus on the unregularized supervised case (i.e. $L$ is a function of $f(\mathbf{x}; \mathbf{\theta})$ and $y$), though it's easy to generalize
  * Better than $J$ is if we can take the expectation over the actual distribution generating the data, $J^\*(\mathbf{\theta}) = \mathbb{E}\_{(\mathbf{x}, y) \sim p\_{\text{data}}} L(f(\mathbf{x}: \mathbf{\theta}), y)$
    - This quantity is known as the _risk_.  It's what we really want to minimise, but we don't know the distribution
    - Instead we could minimise the empirical risk and hope that it works for the risk too
  * In DL, we generally don't even minimise the empirical risk directly.  (Reason: neural nets have high capacity, so danger of overfitting. Also our main tool is gradient-based optimisation, and useful loss functions like 0-1 loss don't have derivaties)
  * A _surrogate loss function_ is something you jam in place of your actual loss function.
    - E.g., minimising 0-1 loss directly is hard (lack of derivative, exponential dependence on input dimensionality) so you replace it with negative log-likelihood of the correct class
    - The negative log-likelihood actually contains more information than what we're optimising (you can optimise it further and push class boundaries futher apart).  This happens sometimes
  * We generally don't optimise all the way to a true local minimum, as we force in some early stopping to prevent overfitting
  * Consider MLE, $\widehat{\mathbf{\theta}\_{\text{ML}}} = \text{argmax}\_{\mathbf{\theta}} \sum\_{i=1}^m \log p\_{\text{model}}(\mathbf{x}^{(i)}, y^{(i)}; \mathbf{\theta})$, which is argmaxing $J(\mathbf{\theta}) = \mathbb{E}\_{(\mathbf{x}, y) \sim \widehat{p}\_{\text{data}}} \log p\_{\text{model}} (x, y; \mathbf{\theta})$.  Doing the argmaxing involves computing the derivative, which is also an expectation over the training set.
    - Computing this is expensive (if the data set is large), so we typically approximate by sampling
    - Justification: most optimisation algorithms converge much faster if they are allowed to rapidly compute approximations rather than slowly compute the exact (and computing the exact is slow, because of the standard error of the mean is $\sigma/\sqrt{n}$, and the $\sqrt{n}$ in the denominator means it takes a lot of samples to reduce this).  Also, there migth be duplication/redundancy in the training set
    - Optimisation algorithms that use the whole dataset are called _batch_ (or _deterministic_) methods
    - Optimisation algorithms that onle use a single example at a time are called _stochastic_ (or _online_) methods
    - Most algorithms used in DL are somewhere in between, and are called _minibatch_ (or sometimes _stochastic_) methods.  Canonical example: stochastic gradient descent
  * Picking batch sizes is an art:
    - There are diminishing returns with very large batches
    - Low batch sizes (e.g. single sample) seem to help improve generalisation error, but require a small learning rate for stability (as low sample size means large variance in the estimate of the gradient), so this can be very costly
    - Samples in a batch can easily be parallelised, so you should make use of that.  But memory becomes a bottleneck for the amount of parallelism
    - GPU architecture means powers of two are common, anything in the range from $32$ to $256$
  * Purely first order optimisation methods (i.e., only need the gradient) can get away with batch size in the hundreds, but second order methods (i.e., need the Hessian) need batch sizes a couple of orders of magnitude higher, unsurprisingly
  * Obviously you have to pick minibatches randomly!  
    - Care might be needed here, e.g. the data might be presented to you in such a way that consecutive samples are correlated with each other
    - Most implementations of minibatch SGD suffle the data once then proceed in minibatch-sized chunks through the data multiple times.  The first pass gives an unbiased estimate of the _generalisation_ error, but later passes are not quite unbiased.  (To see this, just write out the equations.)
    - The later epochs (passes through the data) are good, at least for a few epochs, as the improvements they make to the training error offset the fact that they are probably increasing the gap between the training error and the test error
  * As datasets become larger and larger, increasingly ML applications are tending towards using each training sample exactly once.  When the dataset is very large, overfitting is not an issue; underfitting and computational efficiency are the main concerns

### Challenges in Neural Network Optimisation

  * Generally, in DL the optimisation is harder because you're doing non-convex optimisation
  * One problem (even in the convex case) is _ill conditioning_ of the Hessian:
    - To second order, a gradient descent step of $\epsilon \mathbf{g}$ adds $$\frac{1}{2} \epsilon \mathbf{g}^{\intercal} \mathbf{H} \mathbf{g} - \epsilon \mathbf{g}^{\intercal} \mathbf{g}$$
    - Ill-conditioning is when the first term dominates the second
  * Newton's method is one way to minimise convex functions with poorly conditioned Hessians, but can't be directly applied in the non-convex NN case
  * In convex optimisation, a local minimimum is guaranteed to be a global minimum.  However, for non-convex (in particular NN) problems, there are generally many local minima.  But just picking one isn't too bad
  * _Model identifiability_: a model is said to be identifiable if a sufficiently large training set can rule out all but one setting of a model's parameters.  This is related to uniquneess of minima.  DL models with many latent variables obviously do not have this property.  We can take advantage of _weight space symmetry_ to generate counterexamples, but depending on whether we've imposed additional regularisation or not there could be a bunch of other ways of generating counterexamples
  * Equivalent local minima (like the above examples) are fine: the cost is still the same.  The problem is if you end up in a local minimum which has a much higher cost than the global (or some other local) minimum
  * Can construct networks which have these bad local minima.  There is research on whether real, practical networks do have them, but generally people think not
  * Another problem is saddle points.  In higher dimensions, saddle points are much more likely than minima (think of Hessian with positive or negative eigenvalues; in high dimensional space, a mixture of positive and negative is much more likely then all positive)
  * Interestingly, for many random functions, if a critical point has low cost then it is more likely to be a local minimum (and of course vice versa for maxima).  There are various results trying to fit simple neural networks in to this class of random functions, but nothing definitive
  * Despite the proliferation of saddle points in high dimension, even first order (gradient-only) methods seem to do fine in practice, with SGD escaping the saddle points
  * Saddle points are an issue for Newton's method.  Dauphin _et al_ introduced a _saddle free Newton method_ to get around this
  * Large regions of near-flat gradient are a genuine problem.  In the non-convex case, you can just get stuck in a flat region of high cost, and that isn't good
  * Objective function for highly nonlinear DNNs often contains sharp nonlinearities (cliffs), that are obviously not a nice thing to have to deal with during your gradient-based optimisation.  These can be avoided with _clipping gradients_: when gradient descent suggests taking a very large step (due to large gradient), clip it off at a max value
  * Think of a recurrent network repeatedly feeding through a diagonal matrix: any eigenvalues bigger than one go off to infinity, any smaller than one go to zero.  Gradients go similarly, which is obviously bad.  Mostly this is just an issue for recurrent networks; workaround discussed later
  * One must always remember that the gradient is inexact, particularly if you are using some minibatching which means there is probabilistic uncertainty in here
  * Another issue is where the learning location starts.  There might be a "hill" between your starting point and a good minimum, but you never make it over that hill.  In higher dimensions, you can probably assume that you get some path around that hill, but this costs increased training time

### Basic Algorithms

  * We have already seen gradient descent (walk downhill in the direction of the gradient) and stochastic gradient descent (make an estimate of the gradient by sampling, and walk downhill in that estimated direction)
  * For SGD, the learning rate is an important parameter.  Previously we have thought of this as fixed, but in practice it is gradually decreased over time; write $\epsilon\_k$ for the $k$th iteration
    - The necessity of reducing the learning rate is due to the noise coming from the size-$m$ minibatch estimate for the gradient (it is not necessary for true gradient descent)
    - In practice, it is common to decay the learning rate linearly until some iteration $\tau$, then leave it constant after that
    - Choosing the (initial, say) learning rate is an art.  Too small and learning will take too long, too big and it will be unstable
  * To study the convergence rate of an optimisation algorithm, it is common to measure the _excess error_ $J(\mathbf{\theta}) - \min\_{\mathbf{\theta}} J(\mathbf{\theta})$, the amount by which the current cost function exceeds the minimum possible cost.  Generally, these are bounded by something like $O(1/\sqrt{k})$ or $O(1/k)$, with $k$ being the number of iterations.  Batch does better than stochastic theoretically, but the constants are important here and stochastic is generally better IRL
  * SGD can sometimes be slow, _momentum_ is a method to accelerate learning.  It maintains an exponentially weighted moving average of previous gradients and moves in the combined direction
    - It helps solve two problems:  ill-conditioned Hessians, and the variance coming from SGD
    - There is a parameter $\alpha$ controlling how much weight we give to previous gradients (we move along a velocity vector $\mathbf{v}\_t = \alpha \mathbf{v}\_{t-1} - \epsilon \mathbf{g}$.  $\alpha$ can be adapted over time as well.  Common (initial) choices are $\alpha = 0.5$, $\alpha = 0.9$, and $\alpha = 0.99$
    - Physical intuition: $\mathbf{\theta}$ behaves like the coordinates of a particle experience a force due to gravity (moving it downhill in the direction of the gradient) and a force due to drag (we choose vicuous drag, i.e. linear feedback; it converges nicely).  $\epsilon$ is like the strength of the gravitational force, $\alpha$ is like the strength of the drag force
  * In standard momentum, you compute the gradient at the starting point, before applying the velocity.  In _Nestorov momentum_, you first move direction $\alpha$ along the velocity then compute the gradient
  * Initial point for training is important (affects whether learning well converge at all, stability of convergence, etc.).  But we don't really know any good principled strategies for picking initial points
  * Probably the only thing you can say for sure is that the initial point must break symmetry between different units (input units with same path extruding from them must receive different input values, else the units will be treated the same forever and there's no point in having both).  Nothing complicated needed to ensure this, just pick initial point at random and you're almost surely fine
  * Scale of the initial weights (i.e., large or small numbers): large is good because more symmetry-breaking hence avoiding redundant units; large is bad because potential for numerical instability
  * Regularisation will probably be forcing the weights to be small in some sense.  After some renormalisation, in many cases the regularisation can be effectively expressing a prior that the final parameters should be close to the initial parameters
  * Rough heuristic for scale: take initial weights of a fully connected layer with $m$ inputs and $n$ outputs by sampling from $U\left(-\sqrt{\frac{6}{m+n}}, \sqrt{\frac{6}{m+n}}\right)$
  * _Sparse initialisation_: each unit is initialised to have exactly $k$ non-zero weights.  This counteracts the effect of dense initialisation where in a large layer (with $m$ inputs) the effect of any unit becomes small.  Downside is it expresses a strong prior, and it can take a long time to recover from a bad prior
  * When computational resources allow it, can treat the initial scale of the weights as a hyperparameter search problem
  * Setting the initial value of the biases (the $\mathbf{b}$ in $\mathbf{w}\mathbf{x} + \mathbf{b}$) is easier
    - If looking at the bias for an output unit, then you probably want to set the bias to ensure you get the correct marginal statistics for the problem overall (e.g., classification vector $\mathbf{c}$ specifying the probabilities, set $\mathbf{b}$ s.t. $\text{softmax}(\mathbf{b}) = \mathbf{c}$
    - Can also set the biases to be away from zero (avoiding saturation), or to ensure participation of dependent units in the network (example from LSTM network)
  * In general, most approaches for setting weights/biases/hyperparamters involves some amount of specifying a constant, and some amount of random guessing/search.  An alternative, discussed later in the book, is to set up an auxilary machine learning problem for this

### Algorithms with Adaptive Learning Rates

  * Learning rate is clearly an important parameter at learning time.  A blanket fixed learning rate, moving the same amount along each axis on each step, has obvious problems.  Momentum is one way to solve this, but it introduces another hyperparameter.  An alternative is an _adaptive learning rate_ algorithm
  * _AdaGrad_ adapts the learning rate of all model parameters by accumulating the norms of previous gradients $\mathbf{g}$ into $\mathbf{r}$, and taking the update step $(\Delta\mathbf{\theta})\_i = \frac{\epsilon}{\delta + \sqrt{r\_i}} g\_i$, where $\delta$ is a small positive parameter (something like $10^{-7}$) for numerical stability
    - This has good theoretical properties for _convex_ optimisation
    - For training DNNs, the accumulation of squared gradients from the beginning can result in premature and excessive decrease in effective learning rates
  * _RMSProp_ changes AdaGrad by switching the accumulating gradient in to an exponentially weighted moving average of previous gradients
    - This is more suited to nonconvex optimisation (in nonconvex optimisation, you probably don't want to keep a history of the weird area of random fluctuations you spent dozens of ticks in, before you finally escpaed on got to something that is actually a local minimum)
    - This can be combined with (Nestorov) momentum: apply the interim update (step based on previous velocity), compute the gradient at the interim point, accumulate the gradient (into an exponentially weighted moving average), compute the velocity update (based on drag and gravity, but gravity is now scaled by the accumulating gradient), and apply the position update
    - This is empirically good, is one of the go-to optimisation methods
  * _Adam_ (coming from "adaptive movements")
    - It is a variation on RMSProp with momentum (the momentum is applied differently)
    - It also includes bias corrections to the estimates of the first-order moments (the momentum term) and the second-order moments, to account for their initialisation at the origi
    - Concretely, we keep track of first and second order moment variables $\mathbf{s}$ and $\mathbf{t}$ (initially zero).  The algorithm is then
      - Sample a minibatch and compute a gradient estimate at the current point
      - Update our moment estimates $\mathbf{s} \leftarrow \rho\_1 \mathbf{s} + (1-\rho\_1)\mathbf{g}$ and $\mathbf{t} \leftarrow \rho\_2 \mathbf{t} + (1-\rho\_2) \mathbf{g} \circ \mathbf{g}$
      - Correct biases: $\widehat{\mathbf{s}} = \frac{\mathbf{s}}{1-\rho\_1^t}$ (where $t$ is the number of ticks elapsed), and similarly for $\mathbf{t}$
      - Compute update: $(\Delta\mathbf{\theta})\_i = -\epsilon \frac{\widehat{s}\_i}{\sqrt{\widehat{r}\_i} + \delta}$
    - Modulo the bias correction, this is basically RMSProp with non-Nestorov momentum.  The bias correction means fixes the potential for RMSProp to have high bias in the second order momentum estimate early in training; as a result, Adam is generally robust to the initial choice of hyperparameters
  * SGD, SGD with momentum, RMSProp, RMSProp with momentum, and Adam are all popular choices for optimisation algorithm.  There isn't really single best here though, just need to play around with it

### Approximate Second Order Models

  * For simplicity restrict attention to empiral risk $$J(\mathbf{\theta}) = \mathbb{E}\_{(\mathbf{x}, y) \sim \widehat{p}\_{\text{data}}(\mathbf{x}, y)} \left[L(f(\mathbf{x}; \mathbf{\theta}), y)\right],$$
though it's straightforward to generalise to more general objective functions (e.g. with a regularisation term)
  * Most widely used second order method is Newton's method.  This works by taking a second order approximation around $\mathbf{\theta}\_0$, $$J(\mathbf{\theta}) \approx J(\mathbf{\theta}\_0) + (\mathbf{\theta} - \mathbf{\theta}\_0)^{\intercal} \nabla\_{\mathbf{\theta}} J(\mathbf{\theta}\_0) + \frac{1}{2} (\mathbf{\theta}-\mathbf{\theta}\_0)^{\intercal} H (\mathbf{\theta} - \mathbf{\theta}\_0)$$, then solving for the critical point of this approximation and making the update $$\mathbf{\theta}^\* = \mathbf{\theta}\_0 - H^{-1} \nabla\_{\mathbf{\theta}} J(\mathbf{\theta}\_0)$$
    - If a function is locally quadratic, this is an exact route to the local minimum
    - If the function is convex but not quadratic, it can be iterated to give a path to the minimum
  * In general Newton's method is attracted to points that are critical (it's looking at critical points of second order approximation), which may well be saddle.  In high dimension situations like DL, there are lots of saddle points so this is bad.  Saddle points occur when the Hessian is not positive definite
  * Can force the Hessian to be diagonal by adding a regularisation term, work with $H + \alpha \mathbf{I}$.  Obviously this only works well if the negative eigenvalues of the Hessian are small (if the negative eigenvalues are large, then we need very large $\alpha$, and this completely the dominates the Hessian term and remove the point of doing this)
  * There is a big computational cost to using the Hessian in this way as well: the inverse Hessian has to be computed at every step
  * An alternative is _conjugate gradients_.  Imagine an algorithm taking steps along negative gradient, and doing line search along the negative gradient to determine optimal step size.  Conjugate gradients starts from the observation that, applying this to a (locally) quadratic function, the sequence of step directions $\mathbf{d}\_t$ we take are sequentially orthogonal, which is a somewhat inefficient way to walk
  * For conjugate gradients, we instead take a step satisfying $\mathbf{d}\_t = \nabla\_{\mathbf{\theta}} J(\mathbf{\theta}) + \beta\_t \mathbf{d}\_{t-1}$, where $\beta\_t$ controls how much of the previous direction we should retain
  * We say two gradients are conjugate if $\mathbf{d}\_{t}^{\intercal} H \mathbf{d}\_{t-1} = 0$
  * For a quadratic surface, the taking conjugate step directions ensures that the gradient along the previous direction does not increase in magnitude.  For a quadratic function in a $k$-dimensional parameter space, we therefore (with optimal step lengths) would take at most $k$ steps
  * For conjugate gradients, we can get the $\beta\_t$ by computing $H(\mathbf{\theta})$, but this is expensive aga0n
  * Two alternatives using gradients alone are Fletcher--Reeves and Polak--Ribiere
  * To adapt to non-quadratic cases, use _nonlinear conjugate gradients_, which occassionally reset the conjugation process and takes a free step along an unaltered gradient
  * Seems to work reasonably in practice.  There also exist other alterations specifically aimed at NNs
  * The _Broyden--Fletcher--Goldfrab--Shanno algorithm_ is an adaption of Netwon's method.  Instead of computing $H^{-1}$ at every step (which is expensive), we maintain a cheaper alternative $M\_t$.  Like conjugate gradients, once it knows the direction it does line search on that direction to determine the optimal step size
    - Advantage over conjugate gradients is that it is less insistent that the line search be done perfectly
    - Disadvantage is memory cost of maintaining $M$
    - There is a limited memory alternative 

### Optimisation Strategies and Meta-Algorithms

  * There are a bunch of optimisation tricks/tweaks that can be applied to various optmisation algorithms
  * _Batch normalisation_ is a method of adaptive reparameterisation
    - When we are updated in the parameters in a DNN, we compute the gradient of each parameter individually but only actually apply the update in bulk after we've processed each parameter.  Since all parameters depend on each other, by the time we apply the update, we've moved the goalposts
    - As a simple example, consider a linear change of functions, where we know what we want to tweak the output a bit.  Following back-propogation, we can find that optimal update to each parameter in the chain.  The updates in the penultimate layer contribute order one terms to the output, the updates in the antepenultimate layer contribute order two terms to the output, and so on.  To work out the correct step size to take to tweak the output layer, correctly accounting for all the contributions from the $n$ previous layers, we need an order $n$ method.  This is obviously not feasible for DNNs
    - Batch normalisation can be applied to any input or hidden layer in a NN.  Let $H$ be a minibatch of activations of the layer to normalise, with the activations of each example appearing the a row of the matrix (i.e., a design matrix).  Then $$(H')\_{i,j} = \frac{H\_{i,j}-\mathbf{\mu}\_j}{\mathbf{\sigma}\_j}$$ is the normalisation of $H$, where $\mathbf{\mu}$ is the vector of means (sample means based on this batch I assume?) of each unit, and $\mathbf{\sigma}$ is the standard deviation (maybe numerically stabilised away from zero).  We replace $H$ with $H'$ and the rest of the network operates on it in the same way
    - The obvious way to apply this is a stop-the-world renormalisation after each gradient descent step.  The downside is that often the learning algorithm will propose some change the increases the mean/variance, then the renormalisation will undo this change
    - Instead, you can actually _include_ the renormalisation inside your backprop computational graph, which means that the learning algorithm and the renormalisation will now work together.  This is much more efficient
    - To actually apply a model which includes renormalisation, you need to know $\mathbf{\mu}$ and $\mathbf{\sigma}$, so at training time you should accumulate a moving average of these over all the minibatches you had
    - The output layer of a NN is required to do a linear transformation.  It would make sense to remove all linear relationships between previous layers, and isolate all the linear work to the output.  Batch normalisation is in the same ballpark as this.  It is technically possible to remove all linear interactions in previous layers, but it is more expensive than batch normalisation
    - In practice it is common to replace $H$ with $\mathbf{\gamma}H' + \mathbf{\beta}$.  The learned parameters $\mathbf{\gamma}$ and $\mathbf{\beta}$ allow this layer to have any mean and variance.  In contrast to using the mean and variance inherent in $H$, though, these mean and variance are solely the opinion of the layer producing them.  The learning dynamics are therefore different (and presumably better)
  * _Coordinate descent_ is when you minimise $f(\mathbf{x})$ by iterating through indices $x\_i$ and minimising $f$ as a function of $x\_i$ at each step.  Block coordinate descent is when you do the same thing with groups of coordinates
  * This is definitely a good idea when the variables can be separated in such a way that you get a collection of easy optimisation problems
    - E.g. in sparse coding, we must find $H$ and $W$ minimising $$J(H, W) = \sum\_{i,j} |H\_{i,h}| + \sum\_{i,j} \left(X - W^{\intercal} H\right)\_{i,j}^2$$  This is not convex overall, but if we optimise $H$ and $W$ separately each subproblem is 
  * This is definitely a bad idea when the value of one variable strongly influences the optimal value of another variable.  It will still work, but it can potentially be very slow
  * _Polyak averaging_ proposes to do $t$ iterations of gradient descent (say), then backtrack and step according to the average.  This has good theoretical properties for convex problems and seems to work reasonably in practice for nonconvex ones.  For nonconvex you probably want to do an exponentially weighted moving average
  * _Pretraining_ is when you try a separate problem as a first step.  E.g., you make a simple model try to solve the task, then make the task more complicated.  Or you make a model solve an easier task, then move on to the actual task.  And so on.
  * _Greedy algorithm_: break the problem up into components, solve each one indivudally.  The combined solution may not be optimal for the combined problem, but maybe it's good enough, and at the least you've found a good initial starting point
  * There are lots of techniques combining these ideas.  One is _greedy supervised pretraining_, which is breaking supervised problems into easier supervised problems (then greedily solving those as pretraining)
    - One way is to have subproblems involving only a subset of the layers in the final model.  E.g. imagine $n$ layers, and subproblems for training only the first $m$ layers for $2 \leq m \leq n$, stacking the training problem for the $m$th layer on the $(m-1)$th layer in such a way that it doesn't waste any work
  * Transfer learning is related to supervised pretraining as well. Think stacking a domain-specific layer on top of ResNet.  You train the domain-specific layers by themselves at first; maybe you also _fine-tune_ to tweak some layers of ResNet jointly with the new part of the network 
  * Another idea is _FitNets_.  Train a short (5 layers) and fat network which is easy and quick; call this _teacher_.  Then create a new long (something like 11-19 layers) and thin network (_student_).  Normally the student network would be difficult to train, but if you give it the additional task of predicting the values of the 3rd layer of the teacher network, apparently this makes things better
    - Very roughly, I imagine this works because the early layers of a DNN would benefit from having something force them towards a reasonable value.  I imagine there's a risk in a very deep network for the early layers to lose track of reality
    - Experimentally, seems to help
  * We've talked a lot about improving the optimisation algorithsm.  Obviously it would be nice if the models themselves were easier to optimise.  In practice, using better models rather than better optimisation algorithms is where we have had wins recently (SGD with momentum has been around since the 80s and is still widely used)
  * More precisely, modern NNs "reflect a _design choice_ to use linear transformations between layers and activation functions that are differentiable a.e., with significant slope in large parts of their domain".  (Relativel) modern innovations (LSTM, ReLUs, ...) move toward linear functions and away from the sigmoids of the past.
  * There is another idea which gives gradient hints to early layers in the network (like we did for FitNet) - this is to add auxilary heads (attach outputs to checkpoints in the model) and train these alongside the final output
  * _Continuation methods_: construct a series of cost functions $J^{(0)}, J^{(1)}, ..., J^{(n)}=J$ such that $J^{(i-1)}$ is "easier" than $J^{(i)}$.  By "easier", we mean that the cost function is better behaved, so if you drop a point at random then it should be more likely that it has a reasonable path to an optimal solution for $J^{(i-1)}$ than it does for $J^{(i)}$.  Then you optimise each cost function in turn, using the optimal alue of one as the starting point for the next.
    - One way to construct a sequence is by "blurring" the initial cost function: $$J^{(i)}(\mathbf{\theta}) = \mathbb{E}\_{\mathbf{\theta}' \sim \mathcal{N}(\mathbf{\theta}'; \mathbf{\theta}, \sigma^{(i)2})} J(\mathbf{\theta}')$$
    - Intuitively, a nonconvex function becomes convex when blurred, but if you don't blur it too much you might retain information about a global minimum
    - This is a pretty loose intuition and can obviously break down in a bunch of ways
  * _Curriculum learning_: planning a learning process to begin by learning simple concepts and progress to more complex concepts.  Roughly in the same ballpark as continuation methods (with a series of cost functions)
    - Has been used on a bunch of NL and CV tasks
    - References a paper saying that stochastic cirriculum (random mix of easy/hard topics, but getting progressively harder on average) performs better than a deterministic progression

# Chapter 9 - Convolutional Neural Networks

## Introduction

  * Roughly, CNNs are a kind of NN for processing data that has a known grid-like topology.  Most famous for images (a 2D grid of pixels), but also time series (a 1D grid of samples)
  * More precisely, CNNs are just NNs that use convolution in place of matrix multiplication in one of their layers

## The Convolution Operator

  * Motivated by giving example of smoothing a noisy function $x(t)$ using a weight $w(t)$ by forming the convolution $$s(t)=\int x(a)w(t−a)da$$
  * We normally write this is $s(t)=(x\*w)(t)$. We call the first argument ($x$ here) the input, and we all the second argument ($w$ here) the kernel. The output is sometimes referred to as the feature map
  * Also discrete version: $$s(t)=(x\*w)(t)=\sum\_{a=-\infty}^{\infty}x(a)w(t−a)$$
  * In machine learning, the input is usually a tensor (multi-dimensional array) of features, and the kernel is usually a tensor of parameters to be adapted by the learning algorithm
  * We often use convolutions over more than one axis at a time. E.g., if the input is a 2D greyscale image $X$, then we probably want a 2D kernel $K$, and our convolution is $$S(i,j)=(X\*K)(i,j)=\sum\_m \sum\_n X(m,n)K(i−m,j−n)$$
  * The sum can obviously be rewritten in a bunch of ways. One is to _flip_ the kernel relative to the input $$(X\*K))(i,j)=\sum\_m \sum\_n X(i−m,j−n)K(m,n)$$
  * There is also the closely-related cross correlation, $$S′(i,j)=\sum\_m \sum\_n X(i+m,j+n)K(m,n)$$. This only differs from convolution by a trivial change to $K$
  * Neither of these is seriously different from direct convolution. It may be more numerically convenient to write it one way or the other. But at the end of the day, they are either literally the same or differ only in a trivial change to $K$, which the learning will be smart enough to realise anyway
  * Picture of convolution being applied in 2D. It’s what you would expect, though it does the notable thing of restricting to positions where the kernel fits entirely within the image (so it shrinks the input on axis $i$ by one less than the length of $w$ along axis $i$)
  * Doing a convolution is a bit like multiplying by a matrix except you use a different formula. You can actually represent it as matrix multiplication using things like Toeplitz matrices, block circulant matrices, etc.

## Motivation

  * Uses three important ideas: sparse interactions, parameter sharing, and equivariant representations
  * When you use matrix multiplication in a NN, you are using different parameters in the mapping between a given unit in layer $m$ and a given unit in layer $m+1$. In particular, typically there is a full mesh of interactions between units (though I guess you can regularize away from this)
  * Sparse interactions: In a convolutional layer, a given unit in layer $m$ will only interact with a small number of units in layer $m+1$ (proportional to the size of the convolutional filter). Think of sliding a 2x2 convolutional filter over a 2D array of pixels - a given pixel appears in at most 4 outputs in the next layer
  * Obviously this is much cheaper computationally as well
  * Parameter sharing: Again coming from the difference between multiplying by a matrix and applying a convolution. In matrix multiplication, a specific element $w\_{i,j}$ of $W$ is only relevant in the mapping between $x\_i$ and $y\_j$. In a convolutional network, the coefficients of the convolution are shared through the mapping of (pretty much) every element of $x$ to every element of $y$
  * The parameter sharing doesn’t save any computational cost, but it saves a bit of memory
  * Nice practical example with a picture of a dog and the application of a convolution which looks at the difference between two adjacent pixels on a line (i.e., applies $(−1,1)$ as a convolution). This recognises vertical edges, and is enough to give an outline of the dog already
  * Equivariance: Say a function $f$ is equivariant to $g$ if $f(g(x))=g(f(x)$). (Slightly weird definition of equivariance, but I guess it’s fine for these applications). Note that convolution is equivariant to any function $g$ which translates (shifts) the input
  * For image data, think of function shifting image data one pixel to the right. Convolution is equivariant to that (modulo boundary issues)
  * For time series data, shift in time, convolution doesn’t care.
  * Convolution “creates a map of where certain features appear in the input”. This is useful for when we know that some function of a small number of neighbouring datapoints is useful when applied throughout the image (e.g. edge detection in images, spike detection in time series, …)

## Pooling

  * Convolution is typically applied as:
    - Input
    - Goes to a convolution (the actual affine transformation)
    - Goes to a detector stage (apply the nonlinearity, e.g. ReLU)
    - Goes to pooling stage (applies some further modifaction)
    - Goes to output
  * We typically call all three stages doing work the "convolutional layer". However, other terminologies would only call the first layer the "convolutional layer" and would distinguish a "detector layer" and a "pooling layer"
  * A pooling function replaces the outputs at a certain location with a summary statistic of the nearby outputs. E.g., max pooling reports the maximum output within a rectangular region
  * Pooling helps make the representation approximately invariant to small translations in the input (e.g. shift one pixel to the right)
  * This is a useful thing to aim for if you care more about whether some feature is present rather than exactly where it is
  * The use of pooling can be regarded as an infinitely strong prior that the function the layer learns must be invariant to small translations. If this assumption is correct, then it’s efficient to have the prior built in
  * Can use max pooling to learn invariance. Example with three filters designed to detect different orientations of the digit "5". Pool the responses of these three filters with a max pooling unit and it no longer which one of them activated
  * Pooling can also be used to downsample. Roughly, if you are grouping together $k$ input units, then you can probably get away with only $1/k$ of the output units for many applications
  * Pooling can be used to handle inputs of varying size. E.g. if you are going to do classification on images, probably the core of your network assumes a fixed number of inputs. If you have to classify an image that is not of exactly the correct size, you could use pooling to replace areas with summary statistics to produce something of the correct size

## Convolution and Pooling as an Infinity Strong Prior

  * Can think of a convolutional network as being similar to a fully connected network, but having an infinitely strong prior saying that:
    - The weights for one hidden unit are identical to its neighbour at the same layer
    - The weights are all zero outside of a spatially contiguous region associated with that hidden unit
  * As a corollary, this imposes a prior that the learned function should know about only local interactions and be equivariant to translation
  * Similarly, pooling imposes a prior that the learned function be invariant to translation
  * If the assumptions baked in to the prior are not valid, this will underfit

## Variants of the Basic Convolution Function

  * In practice, the convolution used in NNs is not the mathematical discrete convolution, but differs slightly. We now describe them
  * We normally parallelise, so the operation done in a NN contains many convolutions in parallel. Convolution with a single kernel can extract only one kind of feature, and we usually want to extract many
  * The input is normally a grid of vector values (rather than a grid of real values) as well (e.g. each RGB vector at each pixel)
  * For images we normally think of everything in terms of 3D tensors: two dimensions index into the spatial position, one dimension indexes in to the channel values
  * Write $V\_{i,j,k}$ for the input, $Z\_{i,j,k}$ for the output, and $K\_{i,j,k,l}$ for the kernel, so that $$Z\_{i,j,k}=\sum\_{l,m,n}V\+{l,j+m−1,k+n−1}K\+{i,j,m,n}$$ (This uses $1$-based indexing; in $0$-based indexing you can omit the $−1$s)
  * Can downsample to only retain one in $s$ of the output pixels, $$Z\_{i,j,k}=c(K,V,s)\_{i,j,k}=\sum\_{l,m,n}V\_{l,s(j−1)+m,s(k−1)+n}K\_{i,j,m,n}$$ We call $s$ the stride. Here it is the same across both axes, but it could easily be different
  * There is the question of zero-padding at the boundary of the images as well.
    - No padding at all (called a valid convolution in MATLAB). This shrinks the output image by the size of the kernel minus one
    - Just enough zero padding to keep the output of the same size (called a same convolution in MATLAB). No effect on the size, but pixels near the centre have more influence than pixels at the edge, as pixels near the edge are visited few times
      - Add enough zero padding so that every pixel from the original image is visited $k^2$ times (a full convolution in MATLAB). This increases the size of the output image
  * Usually the best option is somewhere between valid and same
  * Unshared convolution (or locally connected layer):
    - Have a formula like $Z\_{i,j,k}=\sum\_{l,m,n}V\_{l,j−1+m,k−1+n}W\_{i,j,k,l,m,n} with a 6-tensor $W$. The indices into $W$ are $i$ (output channel), $j$ and $k$ (output row/column), $l$ (input channel), $m$ and $n$ (row and column offset in the input)
   - Giving the extra freedom to vary the value as $j$ and $k$ varies means there is nothing shared, so this doesn’t really look like passing the same “detecting filter” over the image
   - This pattern is useful when we know that each feature should be a function of a small part of space, but we don’t have a prior that the same feature should occur over all of space
  * Can also restrict which output channels are influenced by a given input channel
  * Tiled convolutions
    - Compromise between a convolution and an unshared convolution. Learn a set of filters that we cycle through as we move through space
    - If $t$ is the number of filters we cycle through, then the formula is $$Z\_{i,j,k}=\sum\_{l,m,n}V\_{l,j−1+m,k−1+n}K\_{i,l,m,j%t+1,k%t+1}$
    - If $t=1$ this is convolution, if $t$ is the size of the output this is unshared convolution
  * For training a convolutional network, we’re probably going to need three types of operation: convolution itself, backprop from outputs to weights, and backprop from outputs to inputs (the latter is used for autoencoders, apparently). We give a simple 2D examples of these operations:
    - Suppose want to train a convolutional network that uses a strided convolution $c(K,V,s)$, and the loss function we want to minimise is $J(V,K)$
    - For forward propogation, we use $c$ itself to output $Z$
    - During backprop, we will receive a tensor $G$ s.t. $$G\_{i,j,k}=\frac{\partial}\partial Z\_{i,j,}} J(V,K).$$ To train the network, we need to compute the derivatives w.r.t. the weights in the kernel, $$\frac{\partial}J(V,K)}{\partial K\_{i,j,kl}} = \sum\_{m,n}G\_{i,m,n}V\_{j,s(m−1)+k,s(n−1)+l}
    - If this is not the bottom layer, we need to compute the gradient with respect to $V$ as well in order to propogate back to the next stage. For this, we have the formula $$\frac{\partial J(V,K)}{\partial V\_{i,j,k}}=\sum\_{l,m;s(l−1)+m=}j\sum\_{n,p;s(n−1)+p=k}\sum\_q K\_{q,i,m,p}G\_{q,l,n}$$
  * Autoencoder networks are feedforward networks trained to copy their input to their output, e.g. PCA copying $x$ to a reconstruction $W^{\intercal}Wx$
  * Transposes often appear in autoencoders. If the autoencoder has a convolutional layer, we want an efficient way to do that transpose. Unsurprisingly, it basically turns out to be the same formulas as above
  * Quick comment about biases: you might think that any bias added to the convolution should follow the same pattern as the convolution itself (full convolution, locally connected, tiled, …). But you can also just train a separate bias at each position. This is less satistically efficient, but might be a good idea to account for non-uniformities in the input

## Structured Outputs

  * Convolutional networks can output a high-dimensional structured object (as well as the class / real-value we’re used to from classification / regression). Typically it’s just a tensor
  * E.g. could output a tensor $S$ where $S\_{i,j,k}$ is the probability that the pixel at position $(j,k)$ belongs to class $i$
  * Doesn’t seem particularly interesting?

## Data Types

  * Data used in a CNN usually consists of several channels, with each channel being the observation of a different quantity at some point in space (or time). Some examples:
    - Audio waveform: 1D single channel. Data is just the amplitude of the waveform at even time step.
    - Skeleton animation data: 1D multi-channel. Data is the measurement of various angles between limbs, evolving over time.
    - Audio data that has been preprocessed with FT: 2D single channel. Data is the frequency and time
    - Color image data. 2D multi-channel. Data is the RGB values at every 2D position.
    - Volumetric data. 3D single channel. Imagine something coming from a CT scan.
    - Color video data. 3D multi-channel. RGB values for 2D images changing over time.
  * One advantage of convolution networks is that they are not sensitive to the exact size of the input. E.g. think of classifying images: a weight matrix approach would require all training and test data to be the same size, but a convolutional approach allows more freedom
  * Of course, you need an assumption that you’re still working on fundamentally the same problem (e.g. detecting the same thing over a longer period of time)

## Efficient Convolution Algorithms

  * Since CNNs can have millions of units, you want both a good implementation and a good algorithm for computing convolutions
  * Can compute the convolution by applying a FT then multiplying in the frequency domain (this is the convolution theorem). This is more efficient for some problem sizes (FFT is $O(N\logN)$, convolution is $O(N^2)$, but the sweet spot will depend on the constants).
  * If the kernel is separable (can be expressed as an outer product of vectors) then the convolution decomposes and this speeds things up

## Random or Unsupervised Features

  * Typically in a CNN the number of features going in to the output layer is pretty small because there is a lot of pooling going on. Most of the training cost is therefore learning the features
  * When performing supervised learning, need a full pass forward and back through the network. Can try to speed this up by not doing supervised training
  * Three basic approachs to getting kernels to select features without supervised training are:
    - Initialise them randomly. Works surpisingly well with pooling etc. to force robustness. Can choose a bunch of different random kernels then just train the final layer, then take the best network overall. Apparently this does reasonably well.
    - Hand-craft them (e.g. hand-crafted edge detector, etc.)
    - Learn kernels with an unsupervised criterion (discussed in more detail later)
  * Another option is to use greedy layer-wise pretraining (train the first layer in isolation, train the second layer in isolation given the features of the first, etc.). Canonical example is the convolutional deep belief network

## The Neuroscientific Basis for Convolutional Networks

  * CNNs are inspired by neuroscience. Hubel and Wiesel did a lot of work on mammalian vision and got a Nobel Prize for it. In one experiment, they studied cat neurons in isolation and saw that neurons in the early visual system responded strongly to certain patterns of light and didn’t respond at all to other patterns
  * In a simplified view, the “early visual system” lives in V1, the primary visual cortex
  * Pipeline from outside world to V1 can be thought of just transmission and detection (eye focuses on to light-sensor on the retina, passes through the lateral geniculate nucleus, hits V1)
  * A convolutional layer captures three properties of V1:
    - Spatially defined features: light hitting the lower half of the retina only activates half of V1
    - Lots of simple cells: activated by a small spatially localized receptive field
    - Lots of complex cells: simple cells with inavariance built in
  * We know most about V1, but generally people believe the same is true for other parts of the visual system (which would be other layers of the neural network). This leads people to believe (roughly) in things like “grandmother cells” (activated by seeing your grandmother in any pose) deeper in the visual system. Some evidence for this in the medial temporal lobe (but the concept seems to go beyond just image recognition)
  * Closest analogue to a CNNs last layer is the inferotemporal cortex (IT). There is V1, V2, V4, then IT. This seems like a fairly simple feedforward network, at least for the first 100ms of a human viewing an obejct. After that there is a bunch of complicated feedback going on

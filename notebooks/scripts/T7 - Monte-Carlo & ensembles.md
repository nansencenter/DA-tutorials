---
jupyter:
  jupytext:
    formats: ipynb,scripts//py:light,scripts//md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.17.2
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

```python
remote = "https://raw.githubusercontent.com/nansencenter/DA-tutorials"
!wget -qO- {remote}/master/notebooks/resources/colab_bootstrap.sh | bash -s
```

```python
from resources import show_answer, interact, import_from_nb
%matplotlib inline
import numpy as np
import matplotlib as mpl
import scipy.stats as ss
import numpy.random as rnd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
plt.ion();
```

```python
(pdf_G1, grid1d) = import_from_nb("T2", ("pdf_G1", "grid1d"))
```

In [T5](T5%20-%20Multivariate%20Kalman%20filter.ipynb#Exc----The-%22Gain%22-form-of-the-KF) we derived the classical Kalman filter (KF),
$
\newcommand{\Expect}[0]{\mathbb{E}}
\newcommand{\NormDist}{\mathscr{N}}
\newcommand{\DynMod}[0]{\mathscr{M}}
\newcommand{\ObsMod}[0]{\mathscr{H}}
\newcommand{\mat}[1]{{\mathbf{{#1}}}}
\newcommand{\vect}[1]{{\mathbf{#1}}}
\newcommand{\trsign}{{\mathsf{T}}}
\newcommand{\tr}{^{\trsign}}
\newcommand{\ceq}[0]{\mathrel{≔}}
\newcommand{\xDim}[0]{D}
\newcommand{\ta}[0]{\text{a}}
\newcommand{\tf}[0]{\text{f}}
\newcommand{\I}[0]{\mat{I}}
\newcommand{\X}[0]{\mat{X}}
\newcommand{\Y}[0]{\mat{Y}}
\newcommand{\E}[0]{\mat{E}}
\newcommand{\x}[0]{\vect{x}}
\newcommand{\y}[0]{\vect{y}}
\newcommand{\z}[0]{\vect{z}}
\newcommand{\bx}[0]{\vect{\bar{x}}}
\newcommand{\by}[0]{\vect{\bar{y}}}
\newcommand{\bP}[0]{\mat{P}}
\newcommand{\barC}[0]{\mat{\bar{C}}}
\newcommand{\ones}[0]{\vect{1}}
\newcommand{\AN}[0]{\big( \I_N - \ones \ones\tr / N \big)}
$
wherein the dynamics (and measurements) are assumed linear,
i.e. $\DynMod, \ObsMod$ are matrices.
Furthermore, two different forms were derived,@
whose efficiency depends on the relative size of the covariance matrices involved.
But [T6](T6%20-%20Chaos%20%26%20Lorenz%20[optional].ipynb)
illustrated several *non-linear* dynamical systems
that we would like to be able track (estimate).
The classical approach to handle non-linearity
is called the *extended* KF (**EKF**), and its derivation is straightforward:
replace $\DynMod \x^\ta$ by $\DynMod(\x^\ta)$,
and $\DynMod \, \bP^\ta$ by $\frac{\partial \DynMod}{\partial \x}(\x^\ta) \, \bP^\ta$
(where the Jacobian is the integrated TLM also seen in [T6](T6%20-%20Chaos%20%26%20Lorenz%20[optional].ipynb#Error/perturbation-propagation))
and do likewise for $\ObsMod$ with $\x^f$ and $\bP^f$.
The EKF is still highly useful in many engineering problems,
but for the class of problems generally found in geoscience,

- the TLM linearisation is sometimes too inaccurate (or insufficiently robust to the uncertainty),
and the process of deriving and coding up the TLM too arduous
(several PhD years, unless auto-differentiable frameworks have been used)
or downright illegal (proprietary software).
- the size of the covariances $\bP^{\tf / \ta}$ is simply too large to keep in memory.

Therefore, another approach is needed...

# T7 - The ensemble (Monte-Carlo) approach

**Monte-Carlo (M-C) methods** are a class of computational algorithms that rely on random/stochastic sampling.
They generally trade off higher (though random!) error for lower technical complexity [<sup>[1]</sup>](#Footnote-1:).
Examples from optimisation include randomly choosing search directions, swarms,
evolutionary mutations, or perturbations for gradient approximation.
But the main application area is the computation of (deterministic) integrals via sample averages,
which is rooted in the fact that any integral can be formulated as expectations,
combined with the law of large numbers (LLN).
Thus M-C methods apply to surprisingly large class of problems, including for
example a way to [inefficiently approximate the value of $\pi$](https://en.wikipedia.org/wiki/Monte_Carlo_method#Overview).
Indeed, many of the integrals of interest are inherently expectations,
in particular the forecast distribution. Its [integral](T4%20-%20Time%20series%20filtering.ipynb#The-(general)-Bayesian-filtering-recursions)
is intractable, due to the non-trivial nature of the generating process.
However, a Monte-Carlo sample of the forecast distribution
can be generated simply by repeated simulation of eqn. (DynMod).
Then, the ensemble Kalman filter (**EnKF**) analysis update is obtained by replacing
$\DynMod \x^\ta$ and $\DynMod \, \bP^\ta$ by the appropriate ensemble moments/statistics[<sup>2</sup>](#Footnote-2:).
Similarly to the EKF, this relies on linear-Gaussian assumptions,
but the resulting approximation and computational cost will be improved.
The EnKF will be developed in full later;
at present, our focus is on the use of a sample
to reconstruct (estimate) the underlying distribution.
Let us start by learning how to sample our most important distribution.

**Exc – Multivariate Gaussian sampling:**
Suppose $\z$ is a standard Gaussian,
i.e. $p(\z) = \NormDist(\z \mid \vect{0},\I_{\xDim})$,
where $\I_{\xDim}$ is the $\xDim$-dimensional identity matrix.  
Each component, $z_i$, is independent of all others,
and pseudo-random samples thereof can be generated on any modern computer
using one of [these algorithms](https://en.wikipedia.org/wiki/Normal_distribution#Computational_methods).
Now, let $\x = \mat{L}\z + \mu$.

- (a – optional) Refer to the exercise on
  [change of variables](T2%20-%20Gaussian%20distribution.ipynb#Exc-(optional)----Change-of-variables)
  to show that $p(\x) = \NormDist(\x \mid \mu, \mat{C})$,
  where $\mat{C} = \mat{L}^{}\mat{L}^T$.
  In other words, the linear (affine) transformation into $\x$
  yields a shifted (by $\mu$) and "colored" (by $\mat{C}$) random variable.
- (b) The code below samples $N = 100$ realizations of $\x$
  and collects them in an ${\xDim}$-by-$N$ "ensemble matrix" $\E$.
  But `for` loops are slow in Python (and Matlab).
  Replace it with something akin to `E = mu + L@Z`.
  *Hint: this snippet will fail because it's trying to add a vector to a matrix.*

```python
mu = np.array([1, 100, 5])
xDim = len(mu)
L = np.diag(1+np.arange(xDim))
C = L @ L.T
Z = rnd.randn(xDim, N)

# Using a loop ("slow")
E = np.zeros((xDim, N))
for n in range(N):
    E[:, n] = mu + L@Z[:, n]
```

```python
# show_answer('Gaussian sampling', 'b')
```

The following prints some numbers that can be used to ascertain if you got it right.
Note that the estimates will never be exact:
they contain some amount of random error, a.k.a. ***sampling error***.

```python
with np.printoptions(precision=1, suppress=True):
    print("Estimated mean =", np.mean(E, axis=1))
    print("Estimated cov =", np.cov(E), sep="\n")
```

**Exc – Moment estimation code:** Above, we used numpy's (`np`) functions to compute the sample-estimated mean and covariance matrix,
$\bx$ and $\barC$,
from the ensemble matrix $\E$.
Now, instead, implement these estimators yourself:
$$\begin{align}\bx &\ceq \frac{1}{N}   \sum_{n=1}^N \x_n \,, \\
   \barC &\ceq \frac{1}{N-1} \sum_{n=1}^N (\x_n - \bx) (\x_n - \bx)^T \,. \end{align}$$

```python
# Don't use numpy's mean, cov, but feel free to use a `for` loop.
def estimate_mean_and_cov(E):
    xDim, N = E.shape

    ### FIX THIS ###
    x_bar = np.zeros(xDim)
    C_bar = np.zeros((xDim, xDim))

    return x_bar, C_bar

x_bar, C_bar = estimate_mean_and_cov(E)
with np.printoptions(precision=1):
    print("Mean =", x_bar)
    print("Covar =", C_bar, sep="\n")
```

```python
# show_answer('ensemble moments, loop')
```

It can be shown that the above estimators for the mean and the covariance are *consistent and unbiased*[<sup>3</sup>](#Footnote-3:).
***Consistent*** means that if we let $N \rightarrow \infty$, their sampling error will vanish ("almost surely").
***Unbiased*** means that if we repeat the estimation experiment many times (but use a fixed, finite $N$),
then the average of sampling errors will also vanish.
Under relatively mild regularity conditions, the [absence of bias implies consistency](https://en.wikipedia.org/wiki/Consistent_estimator#Bias_versus_consistency).

The following computes a large number ($K$) of $\barC$ and $1/\barC$, estimated with a given ensemble size ($N$).
Note that the true variance is $C = 1$.
The histograms of the estimates is plotted, along with vertical lines displaying the mean values.

```python
K = 10000
@interact(N=(2, 30), bottom=True)
def var_and_precision_estimates(N=4):
    E = rnd.randn(K, N)
    estims = np.var(E, ddof=1, axis=-1)
    bins = np.linspace(0, 6, 40)
    plt.figure()
    plt.hist(estims,   bins, alpha=.6, density=1)
    plt.hist(1/estims, bins, alpha=.6, density=1)
    plt.axvline(np.mean(estims),   color="C0", label="C")
    plt.axvline(np.mean(1/estims), color="C1", label="1/C")
    plt.legend()
    plt.show()
```

**Exc – There's bias, and then there's bias:**

- Note that $1/\barC$ does not appear to be an unbiased estimate of $1/C = 1$.  
  Explain this by referring to a well-known property of the expectation, $\Expect$.  
  In view of this, consider the role and utility of "unbiasedness" in estimation.
- What, roughly, is the dependence of the mean values (vertical lines) on the ensemble size?  
  What do they tend to as $N$ goes to $0$?  
  What about $+\infty$ ?
- Optional: What are the theoretical distributions of $\barC$ and $1/\barC$ ?

```python
# show_answer('variance estimate statistics')
```

**Exc (optional) – Error notions:**

- (a). What's the difference between error and residual?
- (b). What's the difference between error and bias?
- (c). Show that mean-square-error (MSE) = Bias${}^2$ + Var.  
  *Hint: start by writing down the definitions of error, bias, and variance (of $\hat{\theta}$).*

```python
# show_answer('errors')
```

**Exc – Vectorization:** Python (numpy) is quicker if you "vectorize" loops (similar to Matlab and other high-level languages).
This is eminently possible with computations of ensemble moments:
Let $\X \ceq \begin{bmatrix} \x_1 -\bx, & \ldots & \x_N -\bx \end{bmatrix} \,.$

- (a). Show that $\X = \E \AN$, where $\ones$ is the column vector of length $N$ with all elements equal to $1$.  
  *Hint: consider column $n$ of $\X$.*  
  *PS: it can be shown that $\ones \ones\tr / N$ and its complement is a "projection matrix".*
- (b). Show that $\barC = \X \X^T /(N-1)$.
- (c). Code up this, latest, formula for $\barC$ and insert it in `estimate_mean_and_cov(E)`

```python
# show_answer('ensemble moments vectorized')
```

**Exc – Moment estimation code, part 2:** The cross-covariance between two random vectors, $\bx$ and $\by$, is given by
$$\begin{align}
\barC_{\x,\y}
&\ceq \frac{1}{N-1} \sum_{n=1}^N
(\x_n - \bx) (\y_n - \by)^T \\\
&= \X \Y^T /(N-1)
\end{align}$$
where $\Y$ is, similar to $\X$, the matrix whose columns are $\y_n - \by$ for $n=1,\ldots,N$.  
Note that this is simply the covariance formula, but for two different variables.  
I.e. if $\Y = \X$, then $\barC_{\x,\y} = \barC_{\x}$ (which we have denoted $\barC$ in the above).

Implement the cross-covariance estimator in the code-cell below.

```python
def estimate_cross_cov(Ex, Ey):
    Cxy = np.zeros((len(Ex), len(Ey)))  ### INSERT ANSWER ###
    return Cxy
```

```python
# show_answer('estimate cross')
```
We have seen that a sample can be used to estimate the underlying mean and covariance.
Indeed, it can be used to estimate any statistic (expected value wrt. the distribution) of the density.
Another way of stating the same point is that the ensemble can be used to *reconstruct* the underlying distribution.
Indeed, as we have repeatedly seen since T2, a Gaussian distribution can be
described ('parametrized') only through its first two moments,
whereupon the density can be computed through the familiar eqn. (GM).
Another "reconstruction" method that should be familiar to you is that of histograms.
A third option, that we will not detail here, is kernel density estimation (KDE),
which can be seen as a continuous sort of histogram.
These methods are illustrated in the widget below.
Note that the sample/ensemble gets generated via `randn`,
which samples $\NormDist(0, 1)$, and plotted as thin narrow lines.

```python
mu = 0
sigma2 = 25
N = 80

@interact(              seed=(1, 10), nbins=(2, 60), bw=(0.1, 1))
def pdf_reconstructions(seed=5,       nbins=10,      bw=.3):
    rnd.seed(seed)
    E = mu + np.sqrt(sigma2)*rnd.randn(N)

    fig, ax = plt.subplots()
    ax.plot(grid1d, pdf_G1(grid1d, mu, sigma2), lw=5,                      label="True")
    ax.plot(E, np.zeros(N), '|k', ms=100, mew=.4,                          label="_raw ens")
    ax.hist(E, nbins, density=1, alpha=.7, color="C5",                     label="Histogram")
    ax.plot(grid1d, pdf_G1(grid1d, np.mean(E), np.var(E)), lw=5,           label="Parametric")
    ax.plot(grid1d, gaussian_kde(E.ravel(), bw**2).evaluate(grid1d), lw=5, label="KDE")
    ax.set_ylim(top=(3*sigma2)**-.5)
    ax.legend()
    plt.show()
```

**Exc – A matter of taste?:**

- Which approximation to the true pdf looks better?
- Which approximation starts with more information?  
  What is the downside of making such assumptions?
- What value of `bw` causes the "KDE" method to most closely
  reproduce/recover the "Parametric" method?
  What about the "Histogram" method?  
  *PS: we might say that the KDE method "bridges" the other two.*

Thus, an ensemble can be used to characterize uncertainty:
either by using it to compute (estimate) *statistics* thereof, such as the mean, median,
variance, covariance, skewness, confidence intervals, etc
(any function of the ensemble can be seen as a "statistic"),
or by using it to reconstruct the distribution/density from which it is sampled,
as illustrated by the widget above.

### What about linearisation?

We began this tutorial mentioning that M-C can improve on the TLM
for propagating uncertainty (represented by covariances, or an ensemble).
But we yet to tie this back into the discussion of linearisation.
More specifically, what can we say about the way non-linearity is handled by the EnKF?
As it turns out, the EnKF is doing linear least-squares regression [[Anderson (2001)]](#References)
and therefore one can reference the Gauss-Markov theorem to make certain optimality (e.g., BLUE) claims.
Meanwhile, by viewing the ensemble as a set of finite difference perturbations (with large, pseudo-random spread),
ensemble methods were intuitively but heuristically thought to compute an **"average"** linear model somewhat.
This was formalized by [[Raanes (2019)]](#References),
and could have made [[Raanes (2017)]](#References) much shorter (chain rule applies for LLS regression).

#### Exc: Stein's lemma

TODO:
- Univariate
- Proof
  It is important to note that this derivation of the ensemble linearisation
  shows that errors (from different members) cancel out,
  and shows exactly the linearisation converges to,
  both of which are not present in any derivation starting with Taylor-series expansions.
- A similar result was recognized by [[Stordal (2016)]](#References).

## Summary

Monte-Carlo methods use random sampling to estimate expectations and distributions,
making them powerful for complex or nonlinear problems.
Ensembles – i.i.d. samples – allow us to estimate statistics and reconstruct distributions,
with accuracy improving as the ensemble size grows.
Parametric assumptions (e.g. assuming Gaussianity) can be useful in approximating distributions.
Sample mean and covariance estimators are consistent and unbiased,
but nonlinear functions of these (like the inverse covariance) may be biased.
Vectorized computation of ensemble statistics is both efficient and essential for practical use.
The ensemble approach naturally handles nonlinearity by simulating the full system,
forming the basis for methods like the EnKF.

### Next: [T8 - Spatial statistics ("geostatistics") & Kriging](T8%20-%20Geostats%20%26%20Kriging%20[optional].ipynb)

- - -

- ###### Footnote 1:
<a name="Footnote-1:"></a>
  Monte-Carlo is *easy to apply* for any domain of integration,
  and its (pseudo) randomness means makes it robust against hard-to-foresee biases.
  It is sometimes claimed that M-C somewhat escapes the curse of dimensionality because
  – by the CLT or Chebyshev's inequality – the probabilistic error of the M-C approximation
  asymptotically converges to zero at a rate proportional to $N^{-1/2}$,
  regardless of the dimension of the integral, $D$
  (whereas the absolute error of grid-based quadrature methods converges proportional to $N^{-k/D}$,
  for some order $k$).
  However, the "starting" coefficient of the M-C error is generally highly dependent on $D$,
  and much more important than the theoretical asymptote in high dimensions.
  Finally, the **low-discrepancy sequences** of **quasi** M-C
  (arguably the middle-ground between quadrature and M-C)
  usually provide convergence at a rate of $(\log N)^D / N$,
  which is a good deal faster than plain M-C,
  and should dispel any notion that randomness is somehow the secret sauce for fast convergence.
- ###### Footnote 2:
<a name="Footnote-2:"></a>
  **An ensemble** is an *i.i.d.* **sample**.
  Its "members" ("particles", "realizations", or "sample points") have supposedly been drawn ("sampled")
  independently from the same distribution.
  With the EnKF, these assumptions are generally tenuous, but pragmatic.

  Another derivation consists in **hiding** away the non-linearity of $\ObsMod$ by augmenting the state vector with the observations.
  We do not favor this approach pedagogically, since it makes it even less clear just what approximations are being made due to the non-linearity.
- ###### Footnote 3:
<a name="Footnote-3:"></a>
  Why should $(N-1)$ and not simply $N$ be used to normalize the covariance estimate (for unbiasedness)?
  [Formal proof](https://en.wikipedia.org/wiki/Variance#Sample_variance).
  Intuitively: It is because the mean is also unknown, necessitating the use of *its* estimate as well.
  But since one of the terms in $\bx$ is $\x_n$ the two will be positively correlated,
  which causes their difference to be smaller than that of $\mu$ and $\x_n$.
  *PS: in practice, in DA, the use of $(N-1)$ is more of a convention than a requirement,
  since its impact is attenuated by repeat cycling [[Raanes (2019)](#References)], as well as inflation and localisation.*

<a name="References"></a>

### References

<!--
@article{raanes2019adaptive,
    author = {Raanes, Patrick N. and Bocquet, Marc and Carrassi, Alberto},
    title = {Adaptive covariance inflation in the ensemble {K}alman filter by {G}aussian scale mixtures},
    file={~/P/Refs/articles/raanes2019adaptive.pdf},
    doi={10.1002/qj.3386},
    journal = {Quarterly Journal of the Royal Meteorological Society},
    volume={145},
    number={718},
    pages={53--75},
    year={2019},
    publisher={Wiley Online Library}
}

@article{caflisch1998monte,
  title={Monte Carlo and quasi-Monte Carlo methods},
  author={Caflisch, Russel E.},
  journal={Acta numerica},
  volume={7},
  pages={1--49},
  year={1998},
  publisher={Cambridge University Press}
}
-->

- **Raanes (2019)**:
  Patrick N. Raanes, Marc Bocquet, and Alberto Carrassi,
  "Adaptive covariance inflation in the ensemble Kalman filter by Gaussian scale mixtures",
  Quarterly Journal of the Royal Meteorological Society, 2019.
- **Caflisch (1998)**:
  Russel E. Caflisch,
  "Monte Carlo and quasi-Monte Carlo methods",
  Acta Numerica, 1998.

# ---
# jupyter:
#   jupytext:
#     formats: ipynb,nb_mirrors//py:light,nb_mirrors//md
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

remote = "https://raw.githubusercontent.com/nansencenter/DA-tutorials"
# !wget -qO- {remote}/master/notebooks/resources/colab_bootstrap.sh | bash -s

from resources import show_answer, interact, import_from_nb
# %matplotlib inline
import numpy as np
import matplotlib as mpl
import scipy.stats as ss
import numpy.random as rnd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
plt.ion();

(pdf_G1, grid1d, sample_GM) = import_from_nb("T2", ("pdf_G1", "grid1d", "sample_GM"))


# In [T5](T5%20-%20Multivariate%20Kalman%20filter.ipynb#Exc----The-%22Gain%22-form-of-the-KF) we derived the classical Kalman filter (KF),
# $
# \newcommand{\Expect}[0]{\mathbb{E}}
# \newcommand{\NormDist}{\mathscr{N}}
# \newcommand{\DynMod}[0]{\mathscr{M}}
# \newcommand{\ObsMod}[0]{\mathscr{H}}
# \newcommand{\mat}[1]{{\mathbf{{#1}}}}
# \newcommand{\vect}[1]{{\mathbf{#1}}}
# \newcommand{\trsign}{{\mathsf{T}}}
# \newcommand{\tr}{^{\trsign}}
# \newcommand{\ceq}[0]{\mathrel{≔}}
# \newcommand{\xDim}[0]{D}
# \newcommand{\ta}[0]{\text{a}}
# \newcommand{\tf}[0]{\text{f}}
# \newcommand{\I}[0]{\mat{I}}
# \newcommand{\X}[0]{\mat{X}}
# \newcommand{\Y}[0]{\mat{Y}}
# \newcommand{\E}[0]{\mat{E}}
# \newcommand{\x}[0]{\vect{x}}
# \newcommand{\y}[0]{\vect{y}}
# \newcommand{\z}[0]{\vect{z}}
# \newcommand{\bx}[0]{\vect{\bar{x}}}
# \newcommand{\by}[0]{\vect{\bar{y}}}
# \newcommand{\bP}[0]{\mat{P}}
# \newcommand{\barC}[0]{\mat{\bar{C}}}
# \newcommand{\ones}[0]{\vect{1}}
# \newcommand{\AN}[0]{\big( \I_N - \ones \ones\tr / N \big)}
# $
# wherein the dynamics (and measurements) are assumed linear,
# i.e. $\DynMod, \ObsMod$ are matrices.
# Furthermore, two different forms were derived,
# whose efficiency depends on the relative size of the covariance matrices involved.
# But [T6](T6%20-%20Chaos%20%26%20Lorenz%20[optional].ipynb)
# illustrated several *non-linear* dynamical systems
# that we would like to be able track (estimate).
# The classical approach to handle non-linearity
# is called the *extended* KF (**EKF**), and its "derivation" is straightforward:
# replace $\DynMod \x^\ta$ by $\DynMod(\x^\ta)$,
# and $\DynMod \, \bP^\ta$ by $\frac{\partial \DynMod}{\partial \x}(\x^\ta) \, \bP^\ta$
# (the Jacobian can also be seen as the integrated TLM of [T6](T6%20-%20Chaos%20%26%20Lorenz%20[optional].ipynb#Error/perturbation-propagation))
# and do likewise for $\ObsMod$ with $\x^f$ and $\bP^f$.
# The EKF is widely used in engineering,
# but for the class of problems generally found in geoscience,
#
# - the TLM linearisation is sometimes too inaccurate (or insufficiently robust to the uncertainty),
# and the process of deriving and coding up the TLM too arduous
# (several PhD years, unless auto-differentiable frameworks have been used)
# or downright illegal (proprietary software).
# - the size of the covariances $\bP^{\tf / \ta}$ is simply too large to keep in memory,
#   as highlighted in [T7](T7%20-%20Geostats%20%26%20Kriging%20%5Boptional%5D.ipynb).
#
# Therefore, another approach is needed...
#
# # T8 - Monte-Carlo & cov. estimation
#
# **Monte-Carlo (M-C) methods** are a class of computational algorithms that rely on random/stochastic sampling.
# They generally trade off higher (though random!) error for lower technical complexity [<sup>[1]</sup>](#Footnote-1:).
# Examples from optimisation include randomly choosing search directions, swarms,
# evolutionary mutations, or perturbations for gradient approximation.
# But the main application area is the computation of (deterministic) integrals via sample averages,
# which is rooted in the fact that any integral can be formulated as expectations,
# combined with the law of large numbers ([LLN](T2%20-%20Gaussian%20distribution.ipynb#Probability-essentials)).
# Thus M-C methods apply to surprisingly large class of problems, including for
# example a way to [inefficiently approximate the value of $\pi$](https://en.wikipedia.org/wiki/Monte_Carlo_method#Overview).
# Indeed, many of the integrals of interest are inherently expectations,
# in particular the forecast distribution. Its [integral](T4%20-%20Time%20series%20filtering.ipynb#The-(general)-Bayesian-filtering-recursions)
# is intractable, due to the non-trivial nature of the generating process.
# However, a Monte-Carlo sample of the forecast distribution
# can be generated simply by repeated simulation of eqn. (DynMod),
# constituting the forecast step of the ensemble Kalman filter (**EnKF**).
# Meanwhile, its analysis update is obtained by replacing
# $\ObsMod \x^\tf$ and $\ObsMod \, \bP^\tf$ by the appropriate ensemble moments/statistics[<sup>2</sup>](#Footnote-2:).
# Outside of the linear-Gaussian case, this swap is an approximation,
# but the computational cost and/or accuracy may be improved compared with the EKF.
# The EnKF will be developed in full later;
# at present, our focus is on the use of a sample
# to reconstruct, estimate, or represent the underlying distribution.
# If it is assumed Gaussian, this mostly comes down to the estimation of its covariance matrix.
#
# ### Moment estimation

def estimate_mean_and_cov(E):
    #### REPLACE WITH YOUR IMPLEMENTATION ####
    x_bar = np.mean(E, axis=1)
    C_bar = np.cov(E)
    return x_bar, C_bar


# **Exc – barbar implementation:** Above, we've used numpy's (`np`) functions
# to estimate the mean and covariance, $\bx$ and $\barC$,
# from the ensemble matrix $\E = \begin{bmatrix} x_{1},& x_{2}, \ldots x_{N} \end{bmatrix}$:
# Now, instead, implement these estimators yourself:
# $$\begin{align}\bx &\ceq \frac{1}{N}   \sum_{n=1}^N \x_n \,, \\
#    \barC &\ceq \frac{1}{N-1} \sum_{n=1}^N (\x_n - \bx) (\x_n - \bx)^T \,. \end{align}$$
# Use a `for` loop, but don't use numpy's `mean`, `cov`.
# *Hint: it's convenient to start by allocation: `x_bar = np.zeros(...)`*
#
# The following prints some numbers that can be used to check if you got it right.
# Note that the estimates will never be exact:
# they contain some amount of random error, a.k.a. ***sampling error***.

# +
# Draw
N = 80
mu = np.array([1, 100, 5])
L = np.diag([1, 2, 3]) # ⇒ C = diag([1, 4, 9, ...])
E = sample_GM(mu, L=L, N=N)

x_bar, C_bar = estimate_mean_and_cov(E)

with np.printoptions(precision=1, suppress=True):
    print("Estimated mean =", x_bar)
    print("Estimated cov =", C_bar, sep="\n")


# +
# show_answer('ensemble moments, loop')
# -
# **Exc – representation and vectorization**
# Denote the *centered* ensemble matrix
# $\X \ceq \begin{bmatrix} \x_1 -\bx, & \ldots & \x_N -\bx \end{bmatrix} \,.$
#
# - (a): Show that $\X = \E \AN$, where $\ones$ is the column vector of length $N$ with all elements equal to $1$.  
#   *Hint: consider column $n$ of $\X$.*  
#   *PS: it can be shown that $\ones \ones\tr / N$ and its complement is a "projection matrix".*
# - (b): Python (numpy) is quicker if you "vectorize" loops (similar to Matlab and other high-level languages).
#   This is eminently possible with computations of ensemble moments.
#   Show that $$\barC = \X \X^T /(N-1) \,.$$
# - (c) *Optional*: But why don't we try to estimate the "square root" (Cholesky factor), `L`, instead of $\mat{C}$ ?
# - (d): What is the memory requirement of $\X$ vs $\barC$?
# - (e): Code up this latest formula for $\barC$ and insert it in `estimate_mean_and_cov(E)`

# +
# show_answer('ensemble moments vectorized', 'a')
# -

# **Exc – cross-cov:** The cross-covariance between two random vectors, $\bx$ and $\by$, is given by
# $$\begin{align}
# \barC_{\x,\y}
# &\ceq \frac{1}{N-1} \sum_{n=1}^N
# (\x_n - \bx) (\y_n - \by)^T \\\
# &= \X \Y^T /(N-1)
# \end{align}$$
# where $\Y$ is, similar to $\X$, the matrix whose columns are $\y_n - \by$ for $n=1,\ldots,N$.  
# Note that this is simply the covariance formula, but for two different variables,
# i.e., if $\Y = \X$, then $\barC_{\x,\y} = \barC_{\x}$ (denoted $\barC$ above).
#
# The code below uses `np.cov` to compute the cross-covariance (in a wasteful manner).
# Now, instead, implement the above formula yourself:

# +
def estimate_cross_cov(Ex, Ey):
    xDim = len(Ex)
    Cxy = np.cov(Ex, Ey) # cov of (X,Y) jointly
    Cxy = Cxy[:xDim, xDim:]
    return Cxy

Ey = 3 * E + 44444
Cxy_bar = estimate_cross_cov(E, Ey)

with np.printoptions(precision=1, suppress=True):
    print("Estimated cross cov =", Cxy_bar, sep="\n")

# +
# show_answer('estimate cross')
# -
# ### Estimation errors
#
# It can be shown that the above estimators for the mean and the covariance are *consistent and unbiased*[<sup>3</sup>](#Footnote-3:).
# ***Consistent*** means that the error vanishes as $N \rightarrow \infty$.
# ***Unbiased*** means that if we repeat the estimation experiment many times (but use a fixed, finite $N$),
# then the average of sampling errors will also vanish.
# Under relatively mild regularity conditions, the [absence of bias implies consistency](https://en.wikipedia.org/wiki/Consistent_estimator#Bias_versus_consistency).
#
# The following computes a large number ($K$) of $\barC$ and $1/\barC$, estimated with a given ensemble size ($N$).
# Note that the true variance is $C = 1$.
# The histograms of the estimates is plotted, along with vertical lines displaying the mean values.

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


# **Exc – There's bias, and then there's bias:**
#
# - Note that $1/\barC$ does not appear to be an unbiased estimate of $1/C = 1$.  
#   Explain this by referring to a well-known property of the expectation, $\Expect$.  
#   In view of this, consider the role and utility of "unbiasedness" in estimation.
# - What, roughly, is the dependence of the mean values (vertical lines) on the ensemble size?  
#   What do they tend to as $N$ goes to $0$?  
#   What about $+\infty$ ?
# - Optional: What are the theoretical distributions of $\barC$ and $1/\barC$ ?

# +
# show_answer('variance estimate statistics')
# -

# **Exc (optional) – Error notions:**
#
# - (a). What's the difference between error and residual?
# - (b). What's the difference between error and bias?
# - (c). Show that mean-square-error (MSE) = Bias${}^2$ + Var.  
#   *Hint: start by writing down the definitions of error, bias, and variance (of $\hat{\theta}$).*

# +
# show_answer('errors')
# -

# ### Ensemble *representation*
#
# We have seen that a sample can be used to estimate the underlying mean and covariance.
# Indeed, it can be used to estimate any statistic (expected value) of (wrt.) the distribution.
# Another way of stating the same point is that the ensemble can be used to *reconstruct* the underlying distribution.
# Indeed, as we have repeatedly seen since T2, a Gaussian distribution can be
# described ('parametrized') only through its first two moments,
# whereupon the density can be computed through the familiar eqn. (GM).
# Another reconstruction that should be familiar to you is that of histograms.
# Of course, their step-like nature can be off-putting,
# and therefore we should also consider their continuous counterpart,
# namely kernel density estimation (KDE).
#
# These methods are illustrated in the widget below.
# Note that the sample/ensemble gets generated via `randn`,
# which samples $\NormDist(0, 1)$, and plotted as thin narrow lines.

# +
mu = 0
sigma2 = 25

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
# -

# **Exc – A matter of taste?:**
#
# - Which approximation to the true pdf looks better to your eyes?
# - Which approximation starts with more information?  
#   What is the downside of making such assumptions?
# - What value of `bw` causes the "KDE" method to most closely
#   reproduce/recover the "Parametric" method? What about `nbins`?  
#
# Thus, an ensemble can be used to characterize uncertainty:
# either by using it to compute (estimate) *statistics* thereof, such as the mean, median,
# variance, covariance, skewness, confidence intervals, etc
# (any function of the ensemble can be seen as a "statistic"),
# or by using it to reconstruct the distribution/density from which it is sampled,
# as illustrated by the widget above.
#
# ### What about the linearisation?
#
# **Exc – The ensemble switcheroo:**
# Show that, if the covariance matrix $\bP^\tf$ is replaced by its estimate based on $\E^\tf$ then, in the linear case,
# $\ObsMod \, \bP^\tf$ equals the cross covariance estimated from $\ObsMod \, \E^\tf$ and $\E^\tf$.

# +
# show_answer('associativity')
# -
# Now, $\ObsMod \, \bP^\tf$ figures in the KF,
# but is not applicable in the non-linear (and hence non-Gaussian).
# On the other hand, the latter approach, i.e. the EnKF, is applicable.
#
# But what can be said about the way non-linearity is handled by the EnKF?
#
# - Stein ⇒ Average derivative [[Raanes (2019)](#References)] (or derivative of average) .
#   - Exc: Stein's lemma
#     - Univariate
#     - Proof
#       It is important to note that this derivation of the ensemble linearisation
#       shows that errors (from different members) cancel out,
#       and shows exactly the linearisation converges to,
#       both of which are not present in any derivation starting with Taylor-series expansions.
#     - A similar result was recognized by [[Stordal (2016)]](#References).
#
#   Formalizes view of ensemble method as a set of finite difference perturbations (with large, pseudo-random spread).
# - When recognized as preconditioned LLS regression [[Anderson (2001)](#References)], can talk about BLUE.
# - But the whole of the EnKF can be seen as LLS regression (Snyder),
#   which is perhaps not terribly surprising considering that the whole of the KF is called BLUE,
#   i.e. is LLS (but rarely spoken of as such).
#
# ## Summary
#
# Monte-Carlo methods use random sampling to estimate expectations and distributions,
# making them powerful for complex or nonlinear problems.
# Ensembles – i.i.d. samples – allow us to estimate statistics and reconstruct distributions,
# with accuracy improving as the ensemble size grows.
# Parametric assumptions (e.g. assuming Gaussianity) can be useful in approximating distributions.
# Sample mean and covariance estimators are consistent and unbiased,
# but nonlinear functions of these (like the inverse covariance) may be biased.
# Vectorized computation of ensemble statistics is both efficient and essential for practical use.
# The ensemble approach naturally handles nonlinearity by simulating the full system,
# forming the basis for methods like the EnKF.
#
# ### Next: [T9 - Writing your own EnKF](T9%20-%20Writing%20your%20own%20EnKF.ipynb)

# - - -
#
# - ###### Footnote 1:
# <a name="Footnote-1:"></a>
#   Monte-Carlo is *easy to apply* for any domain of integration,
#   and its (pseudo) randomness means makes it robust against hard-to-foresee biases.
#   It is sometimes claimed that M-C somewhat escapes the curse of dimensionality because
#   – by the CLT or Chebyshev's inequality – the probabilistic error of the M-C approximation
#   asymptotically converges to zero at a rate proportional to $N^{-1/2}$,
#   regardless of the dimension of the integral, $\xDim$
#   (whereas the absolute error of grid-based quadrature methods converges proportional to $N^{-k/\xDim}$,
#   for some order $k$).
#   However, the "starting" coefficient of the M-C error is generally highly dependent on $\xDim$,
#   and (in high dimensions) much more important than a theoretical asymptote.
#   Finally, the low-discrepancy sequences of **quasi** M-C [[Caflisch (1998)](#References)]
#   (arguably the middle-ground between quadrature and M-C)
#   usually provide convergence at a rate of $(\log N)^\xDim / N$
#   – a good deal faster than plain M-C –
#   which should dispel any notion that randomness is somehow the secret sauce for fast convergence.
# - ###### Footnote 2:
# <a name="Footnote-2:"></a>
#   **An ensemble** is an *i.i.d.* **sample**.
#   Its "members" ("particles", "realizations", or "sample points") have supposedly been drawn ("sampled")
#   independently from the same distribution.
#   With the EnKF, these assumptions are generally tenuous, but pragmatic.
#
#   Another derivation consists in **hiding** away the non-linearity of $\ObsMod$ by augmenting the state vector with the observations.
#   We do not favor this approach pedagogically, since it makes it even less clear just what approximations are being made due to the non-linearity.
# - ###### Footnote 3:
# <a name="Footnote-3:"></a>
#   Why should $(N-1)$ and not simply $N$ be used to normalize the covariance estimate (for unbiasedness)?
#   Because the left hand side (LHS) of $\sum_n (\x_n - \mu)^2 = N (\bx - \mu)^2 + \sum_n (\x_n - \bx)^2$
#   is always larger than the RHS.
#   *PS: in practice, in DA, the use of $(N-1)$ is more of a convention than a requirement,
#   since its impact is attenuated by repeat cycling [[Raanes (2019)](#References)], as well as inflation and localisation.*
#
# <a name="References"></a>
#
# ### References
#
# <!--
# @article{raanes2019adaptive,
#     author = {Raanes, Patrick N. and Bocquet, Marc and Carrassi, Alberto},
#     title = {Adaptive covariance inflation in the ensemble {K}alman filter by {G}aussian scale mixtures},
#     file={~/P/Refs/articles/raanes2019adaptive.pdf},
#     doi={10.1002/qj.3386},
#     journal = {Quarterly Journal of the Royal Meteorological Society},
#     volume={145},
#     number={718},
#     pages={53--75},
#     year={2019},
#     publisher={Wiley Online Library}
# }
#
# @article{caflisch1998monte,
#   title={Monte Carlo and quasi-Monte Carlo methods},
#   author={Caflisch, Russel E.},
#   journal={Acta numerica},
#   volume={7},
#   pages={1--49},
#   year={1998},
#   publisher={Cambridge University Press}
# }
#
# @article{sakov2008implications,
#     title={Implications of the form of the ensemble transformation in the ensemble square root filters},
#     author={Sakov, Pavel and Oke, Peter R.},
#     file={~/P/Refs/articles/sakov2008implications.pdf},
#     journal={Monthly Weather Review},
#     volume={136},
#     number={3},
#     pages={1042--1053},
#     year={2008}
# }
#
# @article{ott2004local,
#     title={A local ensemble {K}alman filter for atmospheric data assimilation},
#     author={Ott, Edward and Hunt, Brian R. and Szunyogh, Istvan and Zimin, Aleksey V. and Kostelich, Eric J. and Corazza, Matteo and Kalnay, Eugenia and Patil, D. J. and Yorke, James A.},
#     file={~/P/Refs/articles/ott2004local.pdf},
#     journal={Tellus A},
#     volume={56},
#     number={5},
#     pages={415--428},
#     year={2004},
#     publisher={Wiley Online Library}
# }
# -->
#
# - **Raanes (2019)**:
#   Patrick N. Raanes, Marc Bocquet, and Alberto Carrassi,
#   "Adaptive covariance inflation in the ensemble Kalman filter by Gaussian scale mixtures",
#   Quarterly Journal of the Royal Meteorological Society, 2019.
# - **Caflisch (1998)**:
#   Russel E. Caflisch,
#   "Monte Carlo and quasi-Monte Carlo methods",
#   Acta Numerica, 1998.
# - **Sakov (2008)**:
#   Pavel Sakov and Peter R. Oke,
#   "Implications of the form of the ensemble transformation in the ensemble square root filters",
#   Monthly Weather Review, 2008.
# - **Ott (2004)**:
#   Edward Ott, Brian R. Hunt, Istvan Szunyogh, Aleksey V. Zimin, Eric J. Kostelich, Matteo Corazza, Eugenia Kalnay, D. J. Patil, and James A. Yorke,
#   "A local ensemble Kalman filter for atmospheric data assimilation",
#   Tellus A, 2004.

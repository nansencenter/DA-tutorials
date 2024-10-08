# ---
# jupyter:
#   jupytext:
#     formats: ipynb,scripts//py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.1
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

(pdf_G1, grid1d) = import_from_nb("T2", ("pdf_G1", "grid1d"))

# # T8 - The ensemble (Monte-Carlo) approach
# is an approximate method for doing Bayesian inference.
# Instead of computing the full (gridvalues, or parameters, of the) posterior distributions,
# we instead try to generate ensembles from them.
# An ensemble is an *iid* sample. I.e. a set of "members" ("particles", "realizations", or "sample points") that have been drawn ("sampled") independently from the same distribution. With the EnKF, these assumptions are generally tenuous, but pragmatic.
# $
# % ######################################## Loading TeX (MathJax)... Please wait ########################################
# \newcommand{\Reals}{\mathbb{R}} \newcommand{\Expect}[0]{\mathbb{E}} \newcommand{\NormDist}{\mathscr{N}} \newcommand{\DynMod}[0]{\mathscr{M}} \newcommand{\ObsMod}[0]{\mathscr{H}} \newcommand{\mat}[1]{{\mathbf{{#1}}}} \newcommand{\bvec}[1]{{\mathbf{#1}}} \newcommand{\trsign}{{\mathsf{T}}} \newcommand{\tr}{^{\trsign}} \newcommand{\ceq}[0]{\mathrel{≔}} \newcommand{\xDim}[0]{D} \newcommand{\supa}[0]{^\text{a}} \newcommand{\supf}[0]{^\text{f}} \newcommand{\I}[0]{\mat{I}} \newcommand{\K}[0]{\mat{K}} \newcommand{\bP}[0]{\mat{P}} \newcommand{\bH}[0]{\mat{H}} \newcommand{\bF}[0]{\mat{F}} \newcommand{\R}[0]{\mat{R}} \newcommand{\Q}[0]{\mat{Q}} \newcommand{\B}[0]{\mat{B}} \newcommand{\C}[0]{\mat{C}} \newcommand{\Ri}[0]{\R^{-1}} \newcommand{\Bi}[0]{\B^{-1}} \newcommand{\X}[0]{\mat{X}} \newcommand{\A}[0]{\mat{A}} \newcommand{\Y}[0]{\mat{Y}} \newcommand{\E}[0]{\mat{E}} \newcommand{\U}[0]{\mat{U}} \newcommand{\V}[0]{\mat{V}} \newcommand{\x}[0]{\bvec{x}} \newcommand{\y}[0]{\bvec{y}} \newcommand{\z}[0]{\bvec{z}} \newcommand{\q}[0]{\bvec{q}} \newcommand{\br}[0]{\bvec{r}} \newcommand{\bb}[0]{\bvec{b}} \newcommand{\bx}[0]{\bvec{\bar{x}}} \newcommand{\by}[0]{\bvec{\bar{y}}} \newcommand{\barB}[0]{\mat{\bar{B}}} \newcommand{\barP}[0]{\mat{\bar{P}}} \newcommand{\barC}[0]{\mat{\bar{C}}} \newcommand{\barK}[0]{\mat{\bar{K}}} \newcommand{\D}[0]{\mat{D}} \newcommand{\Dobs}[0]{\mat{D}_{\text{obs}}} \newcommand{\Dmod}[0]{\mat{D}_{\text{obs}}} \newcommand{\ones}[0]{\bvec{1}} \newcommand{\AN}[0]{\big( \I_N - \ones \ones\tr / N \big)}
# $
#
# Ensembles can be used to characterize uncertainty: either by using it to compute (estimate) *statistics* thereof, such as the mean, median, variance, covariance, skewness, confidence intervals, etc (any function of the ensemble can be seen as a "statistic"), or by using it to reconstruct the distribution/density from which it is sampled. The latter is illustrated by the plot below. Take a moment to digest its code, and then answer the following exercises.

# +
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


# -

# **Exc -- A matter of taste?:**
# - Which approximation to the true pdf looks better?
# - Which approximation starts with more information?  
#   What is the downside of making such assumptions?
# - What value of `bw` causes the "KDE" method to most closely
#   reproduce/recover the "Parametric" method?
#   What about the "Histogram" method?  
#   *PS: we might say that the KDE method "bridges" the other two.*.

# Being able to sample a multivariate Gaussian distribution is a building block of the EnKF.
# That is the objective of the following exercise.
#
# **Exc -- Multivariate Gaussian sampling:**
# Suppose $\z$ is a standard Gaussian,
# i.e. $p(\z) = \NormDist(\z \mid \bvec{0},\I_{\xDim})$,
# where $\I_{\xDim}$ is the $\xDim$-dimensional identity matrix.  
# Let $\x = \mat{L}\z + \mu$.
#
#  * (a -- optional). Refer to the exercise on [change of variables](T2%20-%20Gaussian%20distribution.ipynb#Exc-(optional)----Probability-and-Change-of-variables) to show that $p(\x) = \mathcal{N}(\x \mid \mu, \mat{C})$, where $\mat{C} = \mat{L}^{}\mat{L}^T$.
#  * (b). The code below samples $N = 100$ realizations of $\x$
#    and collects them in an ${\xDim}$-by-$N$ "ensemble matrix" $\E$.
#    But `for` loops are slow in plain Python (and Matlab).
#    Replace it with something akin to `E = mu + L@Z`.
#    *Hint: this code snippet fails because it's trying to add a vector to a matrix.*

# +
mu = np.array([1, 100, 5])
xDim = len(mu)
L = np.diag(1+np.arange(xDim))
C = L @ L.T
Z = rnd.randn(xDim, N)

# Using a loop ("slow")
E = np.zeros((xDim, N))
for n in range(N):
    E[:, n] = mu + L@Z[:, n]

# +
# show_answer('Gaussian sampling', 'b')
# -

# The following prints some numbers that can be used to ascertain if you got it right.
# Note that the estimates will never be exact:
# they contain some amount of random error, a.k.a. ***sampling error***.

with np.printoptions(precision=1):
    print("Estimated mean =", np.mean(E, axis=1))
    print("Estimated cov =", np.cov(E), sep="\n")


# **Exc -- Moment estimation code:** Above, we used numpy's (`np`) functions to compute the sample-estimated mean and covariance matrix,
# $\bx$ and $\barC$,
# from the ensemble matrix $\E$.
# Now, instead, implement these estimators yourself:
# $$\begin{align}\bx &\ceq \frac{1}{N}   \sum_{n=1}^N \x_n \,, \\
#    \barC &\ceq \frac{1}{N-1} \sum_{n=1}^N (\x_n - \bx) (\x_n - \bx)^T \,. \end{align}$$

# +
# Don't use numpy's mean, cov, but rather a `for` loop.
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

# +
# show_answer('ensemble moments, loop')
# -

# **Exc -- An obsession?:** Why do we normalize by $(N-1)$ for the covariance computation?

# +
# show_answer('Why (N-1)')
# -

# It can be shown that the above estimators are ***consistent and unbiased***.
# Thus, if we let $N \rightarrow \infty$, their sampling error will vanish ("almost surely"),
# and we therefor say that our estimators are *consistent*.
# Meanwhile, if we repeat the estimation experiment many times (but use a fixed, finite $N$),
# then the average of sampling errors will also vanish, since our estimators are also *unbiased*.
# Under relatively mild assumptions, the [absence of bias implies concistency](https://en.wikipedia.org/wiki/Consistent_estimator#Bias_versus_consistency).

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


# **Exc -- There's bias, and then there's bias:**
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

# **Exc (optional) -- Error notions:**
#  * (a). What's the difference between error and residual?
#  * (b). What's the difference between error and bias?
#  * (c). Show that `"mean-square-error" (RMSE^2) = Bias^2 + Var`.  
#    *Hint: Let $e = \hat{\theta} - \theta$ be the random "error" referred to above.
#    Express each term using the expectation $\Expect$.*

# +
# show_answer('errors')
# -

# **Exc -- Vectorization:** Like Matlab, Python (numpy) is quicker if you "vectorize" loops.
# This is eminently possible with computations of ensemble moments.  
# Let $\X \ceq
# \begin{bmatrix}
# 		\x_1 -\bx, & \ldots & \x_N -\bx
# 	\end{bmatrix} \,.$
#  * (a). Show that $\X = \E \AN$, where $\ones$ is the column vector of length $N$ with all elements equal to $1$.  
#    *Hint: consider column $n$ of $\X$.*  
#    *PS: it can be shown that $\ones \ones\tr / N$ and its complement is a "projection matrix".*
#  * (b). Show that $\barC = \X \X^T /(N-1)$.
#  * (c). Code up this, latest, formula for $\barC$ and insert it in `estimate_mean_and_cov(E)`

# +
# show_answer('ensemble moments vectorized')
# -

# **Exc -- Moment estimation code, part 2:** The cross-covariance between two random vectors, $\bx$ and $\by$, is given by
# $$\begin{align}
# \barC_{\x,\y}
# &\ceq \frac{1}{N-1} \sum_{n=1}^N
# (\x_n - \bx) (\y_n - \by)^T \\\
# &= \X \Y^T /(N-1)
# \end{align}$$
# where $\Y$ is, similar to $\X$, the matrix whose columns are $\y_n - \by$ for $n=1,\ldots,N$.  
# Note that this is simply the covariance formula, but for two different variables.  
# I.e. if $\Y = \X$, then $\barC_{\x,\y} = \barC_{\x}$ (which we have denoted $\barC$ in the above).
#
# Implement the cross-covariance estimator in the code-cell below.

def estimate_cross_cov(Ex, Ey):
    Cxy = np.zeros((len(Ex), len(Ey)))  ### INSERT ANSWER ###
    return Cxy

# +
# show_answer('estimate cross')
# -

# ## Summary
# Parametric assumptions (e.g. assuming Gaussianity) can be useful in approximating distributions.
# Sample covariance estimates can be expressed and computed in a vectorized form.
#
# ### Next: [T9 - Writing your own EnKF](T9%20-%20Writing%20your%20own%20EnKF.ipynb)

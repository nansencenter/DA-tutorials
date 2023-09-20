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
plt.ion();

(pdf_G1, grid1d) = import_from_nb("T2", ("pdf_G1", "grid1d"))

# # T8 - The ensemble (Monte-Carlo) approach
# is an approximate method for doing Bayesian inference.
# Instead of computing the full (gridvalues, or parameters, of the) posterior distributions,
# we instead try to generate ensembles from them.
# An ensemble is an *iid* sample. I.e. a set of "members" ("particles", "realizations", or "sample points") that have been drawn ("sampled") independently from the same distribution. With the EnKF, these assumptions are generally tenuous, but pragmatic.
# Ensembles can be used to characterize uncertainty: either by reconstructing (estimating) the distribution from which it is assumed drawn, or by directly computing *statistics* thereof, such as the mean, median, variance, covariance, skewness, confidence intervals, etc (any function of the ensemble can be seen as a "statistic"). This is illustrated by the code below.
# $
# % ######################################## Loading TeX (MathJax)... Please wait ########################################
# \newcommand{\Reals}{\mathbb{R}} \newcommand{\Expect}[0]{\mathbb{E}} \newcommand{\NormDist}{\mathcal{N}} \newcommand{\DynMod}[0]{\mathscr{M}} \newcommand{\ObsMod}[0]{\mathscr{H}} \newcommand{\mat}[1]{{\mathbf{{#1}}}} \newcommand{\bvec}[1]{{\mathbf{#1}}} \newcommand{\trsign}{{\mathsf{T}}} \newcommand{\tr}{^{\trsign}} \newcommand{\ceq}[0]{\mathrel{≔}} \newcommand{\xDim}[0]{D} \newcommand{\supa}[0]{^\text{a}} \newcommand{\supf}[0]{^\text{f}} \newcommand{\I}[0]{\mat{I}} \newcommand{\K}[0]{\mat{K}} \newcommand{\bP}[0]{\mat{P}} \newcommand{\bH}[0]{\mat{H}} \newcommand{\bF}[0]{\mat{F}} \newcommand{\R}[0]{\mat{R}} \newcommand{\Q}[0]{\mat{Q}} \newcommand{\B}[0]{\mat{B}} \newcommand{\C}[0]{\mat{C}} \newcommand{\Ri}[0]{\R^{-1}} \newcommand{\Bi}[0]{\B^{-1}} \newcommand{\X}[0]{\mat{X}} \newcommand{\A}[0]{\mat{A}} \newcommand{\Y}[0]{\mat{Y}} \newcommand{\E}[0]{\mat{E}} \newcommand{\U}[0]{\mat{U}} \newcommand{\V}[0]{\mat{V}} \newcommand{\x}[0]{\bvec{x}} \newcommand{\y}[0]{\bvec{y}} \newcommand{\z}[0]{\bvec{z}} \newcommand{\q}[0]{\bvec{q}} \newcommand{\br}[0]{\bvec{r}} \newcommand{\bb}[0]{\bvec{b}} \newcommand{\bx}[0]{\bvec{\bar{x}}} \newcommand{\by}[0]{\bvec{\bar{y}}} \newcommand{\barB}[0]{\mat{\bar{B}}} \newcommand{\barP}[0]{\mat{\bar{P}}} \newcommand{\barC}[0]{\mat{\bar{C}}} \newcommand{\barK}[0]{\mat{\bar{K}}} \newcommand{\D}[0]{\mat{D}} \newcommand{\Dobs}[0]{\mat{D}_{\text{obs}}} \newcommand{\Dmod}[0]{\mat{D}_{\text{obs}}} \newcommand{\ones}[0]{\bvec{1}} \newcommand{\AN}[0]{\big( \I_N - \ones \ones\tr / N \big)}
# $

# +
mu = 0
sigma2 = 25

@interact(seed=(1, 10), nbins=(2, 100), log_bw=(-2.0, 1))
def pdf_reconstructions(seed=1, nbins=10, log_bw=0.):    
    rnd.seed(seed)
    E = mu + np.sqrt(sigma2)*rnd.randn(100)

    # Plot
    fig, ax = plt.subplots()
    ax.plot(grid1d, pdf_G1(grid1d, mu, sigma2), label="True");
    ax.plot(E, np.zeros(len(E)), '|k', ms=100, mew=.4)
    ax.hist(E, density=1, bins=nbins, alpha=.7, label="Histogram")
    ax.plot(grid1d, pdf_G1(grid1d, np.mean(E), np.var(E)), label="Parametric")
    ax.legend();


# -

# The plot demonstrates that the true distribution can be represented by a sample thereof (since we can almost reconstruct the Gaussian distribution by estimating the moments from the sample). However, there are other ways to reconstruct (estimate) a distribution from a sample. For example: a histogram.
#
# **Exc -- A matter of taste?:** Which approximation to the true pdf looks better: Histogram or the parametric?
# Does one approximation actually start with more information? The EnKF takes advantage of this.

# #### Exc -- KDE
# Use the method of `gaussian_kde` from `scipy.stats` to make another density reconstruction and add it to the plot it above. Explain how this method "bridges" the histogram and the parametric methods (i.e. how to configure it to recover one and then the other).

# +
# show_answer("KDE")
# -

# Being able to sample a Gaussian distribution is a building block of the EnKF.
# In the previous example, we generated samples from a Gaussian distribution using the `randn` function.
# However, that was just for a scalar (univariate) case. We need to be able to sample a multivariate Gaussian distribution. That is the objective of the following exercise.

# **Exc -- Multivariate Gaussian sampling:**
# Suppose $\z$ is a standard Gaussian,
# i.e. $p(\z) = \mathcal{N}(\z \mid \bvec{0},\I_{\xDim})$,
# where $\I_{\xDim}$ is the $\xDim$-dimensional identity matrix.  
# Let $\x = \mat{L}\z + \mu$.
#
#  * (a). Refer to the exercise on [change of variables](T2%20-%20Gaussian%20distribution.ipynb#Exc-(optional)----Probability-and-Change-of-variables) to show that $p(\x) = \mathcal{N}(\x \mid \mu, \mat{C})$, where $\mat{C} = \mat{L}^{}\mat{L}^T$.
#  * (b). Consider the above definition of $\x$ and the code below.
#  Complete it so as to generate a random realization of $\x$.  
#  Hint: matrix-vector multiplication can be done using the symbol `@`.

# +
# show_answer('Gaussian sampling', 'b')

# +
xDim = 3
mu = 10*np.ones(xDim)
C = np.diag(1+np.arange(xDim))
L = np.linalg.cholesky(C) # sqrtm
print("True mean and cov:")
print(mu)
print(C)

### INSERT ANSWER (b) ###
# -

#  * (c). In the code cell below, sample $N = 100$ realizations of $\x$
#    and collect them in an ${\xDim}$-by-$N$ "ensemble matrix" $\E$.
#    Try to avoid the use of `for` loops.
#    *Hint: adding the mean vector to a matrix is a bit tricky due to the chosen orientation of $\E$.*

# +
# show_answer('Gaussian sampling', 'c')

# +
N = 100 # ensemble size

E = np.zeros((xDim, N))

### INSERT ANSWER (c) ###

# Inspect the sample mean and covariance and compare to true values
x_bar = np.mean(E, axis=1)
C_bar = np.cov(E)
with np.printoptions(precision=1):
    print("Estimated mean:")
    print(x_bar)
    print("Estimated covariance:")
    print(C_bar)
plt.matshow(C_bar, cmap="Blues"); plt.colorbar();


# -

# **Exc (optional) -- Sampling error:** How erroneous are the ensemble estimates on average?

# +
# show_answer('Average sampling error')
# -

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

    ### INSERT ANSWER ###

    return x_bar, C_bar

x_bar, C_bar = estimate_mean_and_cov(E)
with np.printoptions(precision=1):
    print(x_bar)
    print(C_bar)


# +
# show_answer('ensemble moments')
# -

# **Exc -- An obsession?:** Why do we normalize by $(N-1)$ for the covariance computation?

# +
# show_answer('Why (N-1)')
# -

# The following computes many $\barC$ and $1/\barC$ estimated with a given ensemble size.
# Note that the values of the true variance being used is 1, as is its inverse.
# The histograms of the estimates is plotted, along with vertical lines displaying their mean values.

@interact(N=(2, 30))
def var_and_precision_estimates(N=4):
    E = rnd.randn(10000, N)
    estims = np.var(E, ddof=1, axis=-1)
    bins = np.linspace(0, 6, 40)
    plt.figure()
    plt.hist(estims, bins, alpha=.6, );
    plt.hist(1/estims, bins, alpha=.6);
    plt.axvline(np.mean(estims), color="C0")
    plt.axvline(np.mean(1/estims), color="C1")
    plt.show()


# **Exc -- There's bias, and then there's bias:**
# - Is $1/\barC$ an unbiased estimate of the true precision (i.e. the reciprocal of the variance, i.e. $1$)?
# - What, roughly, is the dependence of the mean values (vertical lines) on the ensemble size?
#   What do they tend to as $N$ goes to $0$? What about $+\infty$ ?
# - What are the theoretical distributions of $\barC$ and $1/\barC$ ?

# +
# show_answer('variance estimate statistics')
# -

# **Exc -- Vectorization:** Like Matlab, Python (numpy) is quicker if you "vectorize" loops.
# This is eminently possible with computations of ensemble moments.  
# Let $\X \ceq
# \begin{bmatrix}
# 		\x_1 -\bx, & \ldots & \x_n -\bx, & \ldots & \x_N -\bx
# 	\end{bmatrix} \,.$
#  * (a). Show that $\X = \E \AN$, where $\ones$ is the column vector of length $N$ with all elements equal to $1$.
#  Hint: consider column $n$ of $\X$.
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

# **Exc (optional) -- Error notions:**
#  * (a). What's the difference between error residual?
#  * (b). What's the difference between error and bias?
#  * (c). Show `MSE = RMSE^2 = Bias^2 + Var`

# +
# show_answer('errors')
# -

# ## Summary
# Parametric assumptions (e.g. assuming Gaussianity) can be useful in approximating distributions.
# Sample covariance estimates can be expressed and computed in a vectorized form.
#
# ### Next: [T9 - Writing your own EnKF](T9%20-%20Writing%20your%20own%20EnKF.ipynb)

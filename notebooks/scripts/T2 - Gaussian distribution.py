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

from resources import show_answer, interact
# %matplotlib inline
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
plt.ion();

# Before discussing sequential, time-dependent inference,
# we need to know how to estimate unknowns based on a single data/observations (vector).
# But before discussing *Bayes' rule*,
# we should review the most useful of probability distributions.
# # T2 - The Gaussian (Normal) distribution
# $
# % ######################################## Loading TeX (MathJax)... Please wait ########################################
# \newcommand{\Reals}{\mathbb{R}} \newcommand{\Expect}[0]{\mathbb{E}} \newcommand{\NormDist}{\mathscr{N}} \newcommand{\DynMod}[0]{\mathscr{M}} \newcommand{\ObsMod}[0]{\mathscr{H}} \newcommand{\mat}[1]{{\mathbf{{#1}}}} \newcommand{\bvec}[1]{{\mathbf{#1}}} \newcommand{\trsign}{{\mathsf{T}}} \newcommand{\tr}{^{\trsign}} \newcommand{\ceq}[0]{\mathrel{≔}} \newcommand{\xDim}[0]{D} \newcommand{\supa}[0]{^\text{a}} \newcommand{\supf}[0]{^\text{f}} \newcommand{\I}[0]{\mat{I}} \newcommand{\K}[0]{\mat{K}} \newcommand{\bP}[0]{\mat{P}} \newcommand{\bH}[0]{\mat{H}} \newcommand{\bF}[0]{\mat{F}} \newcommand{\R}[0]{\mat{R}} \newcommand{\Q}[0]{\mat{Q}} \newcommand{\B}[0]{\mat{B}} \newcommand{\C}[0]{\mat{C}} \newcommand{\Ri}[0]{\R^{-1}} \newcommand{\Bi}[0]{\B^{-1}} \newcommand{\X}[0]{\mat{X}} \newcommand{\A}[0]{\mat{A}} \newcommand{\Y}[0]{\mat{Y}} \newcommand{\E}[0]{\mat{E}} \newcommand{\U}[0]{\mat{U}} \newcommand{\V}[0]{\mat{V}} \newcommand{\x}[0]{\bvec{x}} \newcommand{\y}[0]{\bvec{y}} \newcommand{\z}[0]{\bvec{z}} \newcommand{\q}[0]{\bvec{q}} \newcommand{\br}[0]{\bvec{r}} \newcommand{\bb}[0]{\bvec{b}} \newcommand{\bx}[0]{\bvec{\bar{x}}} \newcommand{\by}[0]{\bvec{\bar{y}}} \newcommand{\barB}[0]{\mat{\bar{B}}} \newcommand{\barP}[0]{\mat{\bar{P}}} \newcommand{\barC}[0]{\mat{\bar{C}}} \newcommand{\barK}[0]{\mat{\bar{K}}} \newcommand{\D}[0]{\mat{D}} \newcommand{\Dobs}[0]{\mat{D}_{\text{obs}}} \newcommand{\Dmod}[0]{\mat{D}_{\text{obs}}} \newcommand{\ones}[0]{\bvec{1}} \newcommand{\AN}[0]{\big( \I_N - \ones \ones\tr / N \big)}
# $
# Computers generally represent functions *numerically* by their values on a grid
# of points (nodes), an approach called ***discretisation***.
# Don't hesitate to change the grid resolution as you go along!

bounds = -20, 20
N = 201                         # num of grid points
grid1d = np.linspace(*bounds,N) # grid
dx = grid1d[1] - grid1d[0]      # grid spacing


# ## The univariate (a.k.a. 1-dimensional, scalar) case
# Consider the Gaussian random variable $x \sim \NormDist(\mu, \sigma^2)$.  
# Its probability density function (**pdf**),
# $
# p(x) = \NormDist(x \mid \mu, \sigma^2)
# $ for $x \in (-\infty, +\infty)$,
# is given by
# $$\begin{align}
# \NormDist(x \mid \mu, \sigma^2) = (2 \pi \sigma^2)^{-1/2} e^{-(x-\mu)^2/2 \sigma^2} \,. \tag{G1}
# \end{align}$$
#
# Run the cell below to define a function to compute the pdf (G1) using the `scipy` library.

def pdf_G1(x, mu, sigma2):
    "Univariate Gaussian pdf"
    pdf_values = sp.stats.norm.pdf(x, loc=mu, scale=np.sqrt(sigma2))
    return pdf_values


# The following code plots the Gaussian pdf.

hist = []
@interact(mu=bounds, sigma=(.1, 10, 1))
def plot_pdf(mu=0, sigma=5):
    plt.figure(figsize=(6, 2))
    colors = plt.get_cmap('hsv')([(k-len(hist))%9/9 for k in range(9)])
    plt.xlim(*bounds)
    plt.ylim(0, .2)
    hist.insert(0, pdf_G1(grid1d, mu, sigma**2))
    for density_values, color in zip(hist, colors):
        plt.plot(grid1d, density_values, c=color)
    plt.show()

# #### Exc -- parameter influence
# Play around with `mu` and `sigma` to answer these questions:
#  * How does the pdf curve change when `mu` changes?
#  * How does the pdf curve change when you increase `sigma`?
#  * In a few words, describe the shape of the Gaussian pdf curve.
#    Does this ring a bell? *Hint: it should be clear as a bell!*

# **Exc -- Implementation:** Change the implementation of `pdf_G1` so as to not use `scipy`, but your own code (using `numpy` only). Re-run all of the above cells and check that you get the same plots as before.  
# *Hint: `**` is the exponentiation/power operator, but $e^x$ is more efficiently computed with `np.exp(x)`*

# +
# show_answer('pdf_G1')
# -

# **Exc -- Derivatives:** Recall $p(x) = \NormDist(x \mid \mu, \sigma^2)$ from eqn (G1).  
# Use pen, paper, and calculus to answer the following questions,  
# which derive some helpful mnemonics about the distribution.
#
#  * (i) Find $x$ such that $p(x) = 0$.
#  * (ii) Where is the location of the **mode (maximum)** of the density?  
#    I.e. find $x$ such that $\frac{d p}{d x}(x) = 0$.
#    *Hint: begin by writing $p(x)$ as $c e^{- J(x)}$ for some $J(x)$.*
#  * (iii) Where is the **inflection point**? I.e. where $\frac{d^2 p}{d x^2}(x) = 0$.
#  * (iv) *Optional*: Some forms of *sensitivity analysis* (typically for non-Gaussian $p$) consist in estimating/approximating the Hessian, i.e. $\frac{d^2 \log p}{d x^2}$. Explain what this has to do with *uncertainty quantification*.

# #### Exc (optional) -- Probability and Change of variables
# Let $z = \phi(x)$ for some monotonic function $\phi$,
# and $p_x$ and $p_z$ be their probability density functions (pdf).
# - (a): Show that $p_z(z) = p_x\big(\phi^{-1}(z)\big) \frac{1}{|\phi'(z)|}$,
# - (b): Recall the definition of the expectation, $ \Expect[x] ≔ \int  x \, p_x(x) \, d x $, where ***the integral is over the domain***
#   (i.e. from $-\infty$ to $+\infty$ in the case of Gaussian distributions).
#   Show that you don't need to derive the density of $z$ in order to compute its expectation, i.e. that
#   $$ \Expect[z] = \int  \phi(x) \, p_x(x) \, d x ≕ \Expect[\phi(x)] \,,$$
#   *Hint: while the proof is convoluted, the result itself is [pretty intuitive](https://en.wikipedia.org/wiki/Law_of_the_unconscious_statistician).*

# +
# show_answer('CVar in proba')
# -

# #### Exc (optional) -- Integrals
# Recall $p(x) = \NormDist(x \mid \mu, \sigma^2)$ from eqn (G1). Abbreviate it using $c = (2 \pi \sigma^2)^{-1/2}$.  
# Use pen, paper, and calculus to show that
#  - (i) the first parameter, $\mu$, indicates its **mean**, i.e. that $$\mu = \Expect[x] \,.$$
#    *Hint: you can rely on the result of (iii)*
#  - (ii) the second parameter, $\sigma^2>0$, indicates its **variance**,
#    i.e. that $$\sigma^2 = \mathbb{Var}(x) \mathrel{≔} \Expect[(x-\mu)^2] \,.$$
#    *Hint: use $x^2 = x x$ to enable integration by parts.*
#  - (iii) $E[1] = 1$,  
#    thus proving that (G1) indeed uses the right normalising constant.  
#    *Hint: Neither Bernouilli and Laplace managed this,
#    until Gauss did by first deriving $(E[1])^2$.  
#    For more (visual) help, watch [3Blue1Brown](https://www.youtube.com/watch?v=cy8r7WSuT1I&t=3m52s).*

# +
# show_answer('Gauss integrals')
# -

# **Exc -- The uniform pdf**:
# Below is the pdf of the [uniform/flat/box distribution](https://en.wikipedia.org/wiki/Uniform_distribution_(continuous))
# for a given mean and variance.
# - Replace `_G1` by `_U1` in the code generating the above interactive plot.
# - Why are the walls (ever so slightly) inclined?
# - Write your own implementation below, and check that it reproduces the `scipy` version already in place.

def pdf_U1(x, mu, sigma2):
    a = mu - np.sqrt(3*sigma2)
    b = mu + np.sqrt(3*sigma2)
    pdf_values = sp.stats.uniform(loc=a, scale=(b-a)).pdf(x)
    # Your own implementation:
    # height = ...
    # pdf_values = height * np.ones_like(x)
    # pdf_values[x<a] = ...
    # pdf_values[x>b] = ...
    return pdf_values


# +
# show_answer('pdf_U1')
# -

# ## The multivariate (i.e. vector) case
# Here's the pdf of the *multivariate* Gaussian (for any dimension $\ge 1$):
# $$\begin{align}
# \NormDist(\x \mid  \mathbf{\mu}, \mathbf{\Sigma})
# &=
# |2 \pi \mathbf{\Sigma}|^{-1/2} \, \exp\Big(-\frac{1}{2}\|\x-\mathbf{\mu}\|^2_\mathbf{\Sigma} \Big) \,, \tag{GM}
# \end{align}$$
# where $|.|$ represents the matrix determinant,  
# and $\|.\|_\mathbf{W}$ represents a weighted 2-norm: $\|\x\|^2_\mathbf{W} = \x^T \mathbf{W}^{-1} \x$.  
# *PS: The norm (quadratic form) is invariant to antisymmetry in the weight matrix,
# so we take $\mathbf{\Sigma}$ to be symmetric.
# Further, the density (GM) is only integrable over $\Reals^{\xDim}$ if $\mathbf{\Sigma}$ is positive-definite.*
#
# It is important to recognize how similar eqn. (GM) is to the univariate (scalar) case (G1).
# Moreover, [as above](#Exc-(optional)----Integrals) it can be shown that
# - $\mathbf{\mu} = \Expect[\x]$,
# - $\mathbf{\Sigma} = \Expect[(\x-\mu)(\x-\mu)\tr]$.
#
# Note that that the elements of $\mathbf{\Sigma}$ are individual covariances,
# $\Sigma_{i,j} = \Expect[(x_i-\mu_i)(x_j-\mu_j)] = \mathbb{Cov}(x_i, x_j)$.
# Therefore $\mathbf{\Sigma}$ is called the *covariance (matrix)*.
# and its diagonal entries are simply variances, $\Sigma_{i,i} = \mathbb{Var}(x_i)$.
#
# The following implements the pdf (GM). Take a moment to digest the code, but don't worry if you don't understand it all. Hints:
#  * `@` produces matrix multiplication (`*` in `Matlab`);
#  * `*` produces array multiplication (`.*` in `Matlab`);
#  * `axis=-1` makes `np.sum()` work along the last dimension of an ND-array.

# +
from numpy.linalg import det, inv

def weighted_norm22(points, Wi):
    "Computes the weighted norm of each vector (row in `points`)."
    return np.sum( (points @ inv(Wi)) * points, axis=-1)

def pdf_GM(points, mu, Sigma):
    "pdf -- Gaussian, Multivariate: N(x | mu, Sigma) for each x in `points`."
    c = np.sqrt(det(2*np.pi*Sigma))
    return 1/c * np.exp(-0.5*weighted_norm22(points - mu, Sigma))


# -

# The following code plots the pdf as contour (iso-density) curves.

# +
grid2d = np.dstack(np.meshgrid(grid1d, grid1d))

@interact(corr=(-1, 1, .001), std_x=(1e-5, 10, 1))
def plot_pdf_G2(corr=0.7, std_x=1):
    # Form covariance matrix (C) from input and some constants
    var_x = std_x**2
    var_y = 1
    cv_xy = np.sqrt(var_x * var_y) * corr
    C = 25 * np.array([[var_x, cv_xy],
                       [cv_xy, var_y]])
    # Evaluate (compute)
    density_values = pdf_GM(grid2d, mu=0, Sigma=C)
    # Plot
    plt.figure(figsize=(4, 4))
    height = 1/np.sqrt(det(2*np.pi*C))
    plt.contour(grid1d, grid1d, density_values,
               levels=np.linspace(1e-4, height, 11), cmap="plasma")
    plt.axis('equal');
    plt.show()
# -

# **Exc -- Correlation influence:** How do the contours look? Try to understand why. Cases:
#  * (a) correlation=0.
#  * (b) correlation=0.99.
#  * (c) correlation=0.5. (Note that we've used `plt.axis('equal')`).
#  * (d) correlation=0.5, but with non-equal variances.
#
# Finally (optional): why does the code "crash" when `corr = +/- 1` ? Is this a good or a bad thing?  
# *Hint: do you like playing with fire?*

# **Exc Correlation game:** Play [here](http://guessthecorrelation.com/) until you get a score (gold coins) of 5 or more.  
# *PS: you can probably tell that the samples are not drawn from Gaussian distributions. However, the quantitiy $\mathbb{Cov}(x_i, x_i)$ is well defined and can be estimated from the samples.*

# **Exc -- Correlation disambiguation:**
# * What's the difference between correlation and covariance?
# * What's the difference between non-zero (C) correlation (or covariance) and (D) dependence?
#   *Hint: consider this [image](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient#/media/File:Correlation_examples2.svg).*  
#   - Does $C \Rightarrow D$ or the converse?  
#   - What about the negation, $\neg D \Rightarrow \neg C$, or its converse?*  
#   - What about the the (jointly) Gaussian case?
# * Does correlation (or dependence) imply causation?
# * Suppose $x$ and $y$ have non-zero correlation, but neither one causes the other.
#   Does information about $y$ give you information about $x$?

# **Exc (optional) -- Gaussian ubuiqity:** Why are we so fond of the Gaussian assumption?

# +
# show_answer('Why Gaussian')
# -

# ## Summary
# The Normal/Gaussian distribution is bell-shaped.
# Its parameters are the mean and the variance.
# In the multivariate case, the mean is a vector,
# while the second parameter becomes a covariance *matrix*,
# whose off-diagonal elements represent scaled correlation factors,
# which measure *linear* dependence.
#
# ### Next: [T3 - Bayesian inference](T3%20-%20Bayesian%20inference.ipynb)

# ---
# jupyter:
#   jupytext:
#     formats: ipynb,scripts//py:light,scripts//md
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

from resources import show_answer, interact
# %matplotlib inline
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
plt.ion();


# We start by reviewing the most useful of probability distributions.
#
# # T2 - The Gaussian (Normal) distribution
#
# But first, let's refresh some basic theory.
# $
# % ######################################## Loading TeX (MathJax)... Please wait ########################################
# \newcommand{\Reals}{\mathbb{R}} \newcommand{\Expect}[0]{\mathbb{E}} \newcommand{\NormDist}{\mathscr{N}} \newcommand{\DynMod}[0]{\mathscr{M}} \newcommand{\ObsMod}[0]{\mathscr{H}} \newcommand{\mat}[1]{{\mathbf{{#1}}}} \newcommand{\bvec}[1]{{\mathbf{#1}}} \newcommand{\trsign}{{\mathsf{T}}} \newcommand{\tr}{^{\trsign}} \newcommand{\ceq}[0]{\mathrel{≔}} \newcommand{\xDim}[0]{D} \newcommand{\supa}[0]{^\text{a}} \newcommand{\supf}[0]{^\text{f}} \newcommand{\I}[0]{\mat{I}} \newcommand{\K}[0]{\mat{K}} \newcommand{\bP}[0]{\mat{P}} \newcommand{\bH}[0]{\mat{H}} \newcommand{\bF}[0]{\mat{F}} \newcommand{\R}[0]{\mat{R}} \newcommand{\Q}[0]{\mat{Q}} \newcommand{\B}[0]{\mat{B}} \newcommand{\C}[0]{\mat{C}} \newcommand{\Ri}[0]{\R^{-1}} \newcommand{\Bi}[0]{\B^{-1}} \newcommand{\X}[0]{\mat{X}} \newcommand{\A}[0]{\mat{A}} \newcommand{\Y}[0]{\mat{Y}} \newcommand{\E}[0]{\mat{E}} \newcommand{\U}[0]{\mat{U}} \newcommand{\V}[0]{\mat{V}} \newcommand{\x}[0]{\bvec{x}} \newcommand{\y}[0]{\bvec{y}} \newcommand{\z}[0]{\bvec{z}} \newcommand{\q}[0]{\bvec{q}} \newcommand{\br}[0]{\bvec{r}} \newcommand{\bb}[0]{\bvec{b}} \newcommand{\bx}[0]{\bvec{\bar{x}}} \newcommand{\by}[0]{\bvec{\bar{y}}} \newcommand{\barB}[0]{\mat{\bar{B}}} \newcommand{\barP}[0]{\mat{\bar{P}}} \newcommand{\barC}[0]{\mat{\bar{C}}} \newcommand{\barK}[0]{\mat{\bar{K}}} \newcommand{\D}[0]{\mat{D}} \newcommand{\Dobs}[0]{\mat{D}_{\text{obs}}} \newcommand{\Dmod}[0]{\mat{D}_{\text{obs}}} \newcommand{\ones}[0]{\bvec{1}} \newcommand{\AN}[0]{\big( \I_N - \ones \ones\tr / N \big)}
# $
#
# ## Probability essentials
#
# As stated by James Bernoulli (1713) and elucidated by [Laplace (1812)](#References):
#
# > The Probability for an event is the ratio of the number of cases favorable to it, to the number of all
# > cases possible when nothing leads us to expect that any one of these cases should occur more than any other,
# > which renders them, for us, equally possible:
#
# $$ \mathbb{P}(\text{event}) = \frac{\text{# favorable outcomes}}{\text{# possible outcomes}} $$
#
# - A *discrete* random variable, $X$, has a probability *mass* function (**pmf**) defined by $p(x) = \mathbb{P}(X{=}x)$.  
#   Sometimes clarity will necessitate denotering it $p_X(x)$, to distinguish it from $p_Y(y)$.
# - The *joint* probability of two random variables $X$ and $Y$ is defined by the intersections:
#   $p(x, y) = \mathbb{P}(X{=}x \cap Y{=}y)$.  
#   The *marginal* $p(x)$ is recovered by summing over all $y$, and vice-versa.
# - The *conditional* probability of $X$ given $Y$ is defined by $p(x|y) = p(x,y)/p(y)$.
# - A *continuous* random variable has a probability *density* function (**pdf**) defined by
#   $p(x) = \mathbb{P}(X \in [x, x+\delta x])/\delta x$, with $\delta x \to 0$.  
#   Equivalently, $p(x) = F'(x)$, where $F$ is the cumulative distribution function (**cdf**), $F(x) = \mathbb{P}(X \le x)$.
#
# A **sample average** based on draws from a random variable $x$ (we no longer use uppercase for random variables!)
# is denoted with an overhead bar:
# $$ \bar{x} := \frac{1}{N} \sum_{n=1}^{N} x_n \,. $$
# By the *law of large numbers (LLN)*, the sample average converges for $N \to \infty$ to the **expected value** (*sometimes* called the **mean**):
# $$ \Expect[x] ≔ \int x \, p(x) \, d x \,, $$
# where the (omitted) domain of integration is *all values of $x$*.
#
# ## The univariate (a.k.a. 1-dimensional, scalar) Gaussian
#
# If $x$ is Gaussian (a.k.a. "Normal"), we write
# $x \sim \NormDist(\mu, \sigma^2)$, or $p(x) = \NormDist(x \mid \mu, \sigma^2)$,
# where the parameters $\mu$ and $\sigma^2$ are called the mean and variance
# (for reasons that will become clear below).
# The Gaussian pdf is, for $x \in (-\infty, +\infty)$,
# $$ \large \NormDist(x \mid \mu, \sigma^2) = (2 \pi \sigma^2)^{-1/2} e^{-(x-\mu)^2/2 \sigma^2} \,. \tag{G1} $$
#
# Run the cell below to define a function to compute the pdf (G1) using the `scipy` library.

def pdf_G1(x, mu, sigma2):
    "Univariate Gaussian pdf"
    pdf_values = sp.stats.norm.pdf(x, loc=mu, scale=np.sqrt(sigma2))
    return pdf_values


# Computers typically represent functions *numerically* by their values on a grid
# of points (nodes), an approach called ***discretisation***.

bounds = -20, 20
N = 201                         # num of grid points
grid1d = np.linspace(*bounds,N) # grid
dx = grid1d[1] - grid1d[0]      # grid spacing

# Feel free to come back here later and change the grid resolution to see how
# it affects the cells below (upon re-running them).
#
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
#
# Play around with `mu` and `sigma` to answer these questions:
#
# - How does the pdf curve change when `mu` changes? Options (several might be right/wrong)
#   1. It changes the curve into a uniform distribution.
#   1. It changes the width of the curve.
#   1. It shifts the peak of the curve to the left or right.
#   1. It changes the height of the curve.
#   1. It transforms the curve into a binomial distribution.
#   1. It makes the curve wider or narrower.
#   1. It modifies the skewness (asymmetry) of the curve.
#   1. It causes the curve to expand vertically while keeping the width the same.
#   1. It translates the curve horizontally.
#   1. It alters the kurtosis (peakedness) of the curve.
#   1. It rotates the curve around the origin.
#   1. It makes the curve a straight line.
# - How does the pdf curve change when you increase `sigma`?  
#   Refer to the same options as previous question.
# - In a few words, describe the shape of the Gaussian pdf curve.
#   Does this ring a bell? *Hint: it should be clear as a bell!*
#
# **Exc -- Implementation:** Change the implementation of `pdf_G1` so as to not use `scipy`, but your own code (using `numpy` only). Re-run all of the above cells and check that you get the same plots as before.  
# *Hint: `**` is the exponentiation/power operator, but $e^x$ is more efficiently computed with `np.exp(x)`*

# +
# show_answer('pdf_G1')
# -

# **Exc -- Derivatives:** Recall $p(x) = \NormDist(x \mid \mu, \sigma^2)$ from eqn. (G1).  
# Use pen, paper, and calculus to answer the following questions,  
# which derive some helpful mnemonics about the distribution.
#
# - (i) Find $x$ such that $p(x) = 0$.
# - (ii) Where is the location of the **mode (maximum)** of the density?  
#   I.e. find $x$ such that $\frac{d p}{d x}(x) = 0$.
#   *Hint: begin by writing $p(x)$ as $c e^{- J(x)}$ for some $J(x)$.*
# - (iii) Where is the **inflection point**? I.e. where $\frac{d^2 p}{d x^2}(x) = 0$.
# - (iv) *Optional*: Some forms of *sensitivity analysis* (typically for non-Gaussian $p$) consist in estimating/approximating the Hessian, i.e. $\frac{d^2 \log p}{d x^2}$. Explain what this has to do with *uncertainty quantification*.
#
# <a name="Exc-(optional)----Change-of-variables"></a>
#
# #### Exc (optional) -- Change of variables
#
# Let $z = \phi(x)$ for some monotonic function $\phi$,
# and $p_x$ and $p_z$ be their probability density functions (pdf).
#
# - (a): Show that $p_z(z) = p_x\big(\phi^{-1}(z)\big) \frac{1}{|\phi'(z)|}$,
# - (b): Show that you don't need to derive the density of $z$ in order to compute its expectation, i.e. that
#   $$ \Expect[z] = \int  \phi(x) \, p_x(x) \, d x ≕ \Expect[\phi(x)] \,,$$
#   *Hint: while the proof is convoluted, the result itself is [pretty intuitive](https://en.wikipedia.org/wiki/Law_of_the_unconscious_statistician).*

# +
# show_answer('CVar in proba')
# -

# <a name="Exc-(optional)----Integrals"></a>
#
# #### Exc (optional) -- Integrals
#
# Recall $p(x) = \NormDist(x \mid \mu, \sigma^2)$ from eqn. (G1). Abbreviate it using $c = (2 \pi \sigma^2)^{-1/2}$.  
# Use pen, paper, and calculus to show that
#
# - (i) the first parameter, $\mu$, indicates its **mean**, i.e. that $$\mu = \Expect[x] \,.$$
#   *Hint: you can rely on the result of (iii)*
# - (ii) the second parameter, $\sigma^2>0$, indicates its **variance**,
#   i.e. that $$\sigma^2 = \mathbb{Var}(x) \mathrel{≔} \Expect[(x-\mu)^2] \,.$$
#   *Hint: use $x^2 = x x$ to enable integration by parts.*
# - (iii) $E[1] = 1$,  
#   thus proving that (G1) indeed uses the right normalising constant.  
#   *Hint: Neither Bernoulli and Laplace managed this,
#   until [Gauss (1809)](#References) did by first deriving $(E[1])^2$.  
#   For more (visual) help, watch [3Blue1Brown](https://www.youtube.com/watch?v=cy8r7WSuT1I&t=3m52s).*

# +
# show_answer('Gauss integrals')
# -

# **Exc -- The uniform pdf**:
# Below is the pdf of the [uniform/flat/box distribution](https://en.wikipedia.org/wiki/Uniform_distribution_(continuous))
# for a given mean and variance.
#
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

# ## The multivariate (i.e. vector) Gaussian
#
# A *multivariate* random variable, i.e. **vector**, is simply a collection of scalar variables (on the same probability space).
# I.e. its density is the joint density of its components.
# The pdf of the multivariate Gaussian (for any dimension $\ge 1$) is
#
# $$\large \NormDist(\x \mid  \mathbf{\mu}, \mathbf{\Sigma}) = |2 \pi \mathbf{\Sigma}|^{-1/2} \, \exp\Big(-\frac{1}{2}\|\x-\mathbf{\mu}\|^2_\mathbf{\Sigma} \Big) \,, \tag{GM} $$
# where $|.|$ represents the matrix determinant,  
# and $\|.\|_\mathbf{W}$ represents a weighted 2-norm: $\|\x\|^2_\mathbf{W} = \x^T \mathbf{W}^{-1} \x$.  
#
# <details style="border: 1px solid #aaaaaa; border-radius: 4px; padding: 0.5em 0.5em 0;">
# <summary style="font-weight: normal; font-style: italic; margin: -0.5em -0.5em 0; padding: 0.5em;">
#   $\mathbf{W}$ must be symmetric-positive-definite (SPD) because ... (optional reading 🔍)
# </summary>
#
# - The norm (a quadratic form) is invariant to any asymmetry in the weight matrix.
# - The density (GM) would not be integrable (over $\Reals^{\xDim}$) if $\x\tr \mathbf{\Sigma} \x > 0$.
#
# - - -
# </details>
#
# It is important to recognize how similar eqn. (GM) is to the univariate (scalar) case (G1).
# Moreover, [similarly as above](#Exc-(optional)----Integrals), it can be shown that
#
# - $\mathbf{\mu} = \Expect[\x]$,
# - $\mathbf{\Sigma} = \Expect[(\x-\mu)(\x-\mu)\tr]$,
#
# I.e. the elements of $\mathbf{\Sigma}$ are the individual covariances,
# $\Sigma_{i,j} = \Expect[(x_i-\mu_i)(x_j-\mu_j)] =: \mathbb{Cov}(x_i, x_j)$
# and, on the diagonal ($i=j$), variances: $\Sigma_{i,i} = \mathbb{Var}(x_i)$.
# Therefore $\mathbf{\Sigma}$ is called the *covariance (matrix)*.
#
# The following implements the pdf (GM). Take a moment to digest the code, but don't worry if you don't understand it all. Hints:
#
# - `@` produces matrix multiplication (`*` in `Matlab`);
# - `*` produces array multiplication (`.*` in `Matlab`);
# - `axis=-1` makes `np.sum()` work along the last dimension of an ND-array.

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

# The following code plots the pdf as contour (level) curves.

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

# The code defines the covariance `cv_xy` from the input ***correlation*** `corr`.
# This is a coefficient (number), defined for any two random variables $x$ and $y$ (not necessarily Gaussian) by
# $$ \rho[x,y]=\frac{\mathbb{Cov}[x,y]}{\sigma_x \sigma_y} \,. $$
# This correlation quantifies (defines) the ***linear dependence*** between $x$ and $y$. Indeed,
#
# - $-1\leq \rho \leq 1$ (by Cauchy-Swartz)
# - **If** $X$ and $Y$ are *independent*, i.e. $p(x,y) = p(x) \, p(y)$ for all $x, y$, then $\rho[X,Y]=0$.
#
# **Exc -- Correlation influence:** How do the contours look? Try to understand why. Cases:
#
# - (a) correlation=0.
# - (b) correlation=0.99.
# - (c) correlation=0.5. (Note that we've used `plt.axis('equal')`).
# - (d) correlation=0.5, but with non-equal variances.
#
# Finally (optional): why does the code "crash" when `corr = +/- 1` ? Is this a good or a bad thing?  
# *Hint: do you like playing with fire?*
#
# **Exc Correlation game:** [Play](http://guessthecorrelation.com/) until you get a score (gold coins) of 5 or more.  
#
# **Exc -- Correlation disambiguation:**
#
# - What's the difference between correlation and covariance (in words)?
# - What's the difference between non-zero (C) correlation (or covariance) and (D) dependence?
#   *Hint: consider this [image](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient#/media/File:Correlation_examples2.svg).*  
#   - Does $C \Rightarrow D$ or the converse?  
#   - What about the negation, $\neg D \Rightarrow \neg C$, or its converse?*  
#   - What about the (jointly) Gaussian case?
# - Does correlation (or dependence) imply causation?
# - Suppose $x$ and $y$ have non-zero correlation, but neither one causes the other.
#   Does information about $y$ give you information about $x$?
#
# **Exc (optional) -- Gaussian ubiquity:** Why are we so fond of the Gaussian assumption?

# +
# show_answer('Why Gaussian')
# -

# ## Summary
#
# The Normal/Gaussian distribution is bell-shaped.
# Its parameters are the mean and the variance.
# In the multivariate case, the mean is a vector,
# while the second parameter becomes a covariance *matrix*,
# whose off-diagonal elements represent scaled correlation factors,
# which measure *linear* dependence.
#
# ### Next: [T3 - Bayesian inference](T3%20-%20Bayesian%20inference.ipynb)
#
# <a name="References"></a>
#
# ### References
#
# - **Laplace (1812)**: P. S. Laplace, "Théorie Analytique des Probabilités", 1812.
# - **Gauss (1809)**: Gauss, C. F. (1809). *Theoria Motus Corporum Coelestium in Sectionibus Conicis Solem Ambientium*. Specifically, Book II, Section 3, Art. 177-179, where he presents the method of least squares (which will be very relevant to us) and its probabilistic justification based on the normal distribution of errors).

---
jupyter:
  jupytext:
    formats: ipynb,nb_mirrors//py:light,nb_mirrors//md
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
from resources import show_answer, interact
%matplotlib inline
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import numpy.random as rnd
plt.ion();
rnd.seed(3000)
```

# T2 - The Gaussian (Normal) distribution

We begin by reviewing the most useful of probability distributions.
But first, let's refresh some basic theory.
$
\newcommand{\Reals}{\mathbb{R}}
\newcommand{\Expect}[0]{\mathbb{E}}
\newcommand{\NormDist}{\mathscr{N}}
\newcommand{\mat}[1]{{\mathbf{{#1}}}}
\newcommand{\vect}[1]{{\mathbf{#1}}}
\newcommand{\trsign}{{\mathsf{T}}}
\newcommand{\tr}{^{\trsign}}
\newcommand{\z}[0]{\vect{z}}
\newcommand{\E}[0]{\mat{E}}
\newcommand{\I}[0]{\mat{I}}
\newcommand{\x}[0]{\vect{x}}
\newcommand{\X}[0]{\mat{X}}
$

<a name="Probability-essentials"></a>

## Probability essentials

As stated by James Bernoulli (1713) and elucidated by [Laplace (1812)](#References):

> The Probability for an event is the ratio of the number of cases favorable to it, to the number of all
> cases possible when nothing leads us to expect that any one of these cases should occur more than any other,
> which renders them, for us, equally possible:

$$ \mathbb{P}(\text{event}) = \frac{\text{number of} \textit{ favorable } \text{outcomes}}{\text{number of} \textit{ possible } \text{outcomes}} $$

The probability of *both* events $A$ and $B$ occurring is given by their intersection:
$\mathbb{P}(A \cap B)$, while the probability of *either (or)* is obtained by their union $\mathbb{P}(A \cup B)$.
The *conditional* probability of $A$ given $B$ restricts our attention (count)
to cases where $B$ occurs: $\mathbb{P}(A | B) = \frac{\mathbb{P}(A \cap B)}{\mathbb{P}(B)}$.

A **random variable**, $X$, is a *numeric quantity* taking values as a function of some underlying random process.
Rather than "*did* event $A$ occur or no?",
random variables conveniently enable the question
"*what* was the value of $X$?".
Each value, $x$, constitutes an event that is disjoint from all others (functions never being one-to-many),
and so they define a probability space of outcomes with associated probabilities,
which can be tabulated into *distributions*.
If $X$ is *discrete*, then $p_X(x) := \mathbb{P}(X{=}x)$ is a list mapping outcomes to probabilities
called the probability *mass* function (**pmf**).
It sums to 1, and may be written $p(x)$ if contextually unambiguous.
The cumulative distribution function (**cdf**) is defined as $F(x) := \mathbb{P}(X \le x)$.
The 2D table of *joint* probabilities of $X$ and $Y$ is denoted $p(x, y) = \mathbb{P}(X{=}x \cap Y{=}y)$,
while the conditionals are denoted $p(x|y) = \frac{p(x,y)}{p(y)}$.

- The *marginal* pmf, $p(x)$, can be recovered from the joint pmf, $p(x, y)$, by summing over all $y$.
- *Independence* means $p(x, y) = p(x) \, p(y)$ for all possible $x, y$. Equivalently $p(x|y) = p(x)$.

We will mainly be concerned with *continuous* random variables,
for which $\mathbb{P}(X \in I)$ may be non-zero for any interval, $I$.
The distribution of $X$ is then characterised by its probability *density* function (**pdf**),
defined as $p(x) = F'(x)$ or

$$p(x) = \lim_{h \to 0} \frac{\mathbb{P}(X \in [x,\, x{+} h])}{h} \,.$$

The **sample average** of draws from a random variable $X$
is denoted with an overhead bar:
$$ \bar{x} := \frac{1}{N} \sum_{n=1}^{N} x_n \,. $$
The *law of large numbers (LLN)* states that, as $N \to \infty$,
the sample average converges to the **expected value** (sometimes called the **mean**):
$$ \Expect[X] ≔ \int x \, p(x) \, d x \,, $$
where the (omitted) domain of integration is *all values of $x$*.

## The univariate (a.k.a. 1-dimensional, scalar) Gaussian

If $X$ is Gaussian (also known as "Normal"), we write
$X \sim \NormDist(\mu, \sigma^2)$, or $p(x) = \NormDist(x \mid \mu, \sigma^2)$,
where the parameters $\mu$ and $\sigma^2$ are called the mean and variance
(for reasons that will become clear below).
The Gaussian pdf, for $x \in (-\infty, +\infty)$, is
$$ \large \NormDist(x \mid \mu, \sigma^2) = (2 \pi \sigma^2)^{-1/2} e^{-(x-\mu)^2/2 \sigma^2} \, . \tag{G1} $$

Run the cell below to define a function to compute the pdf (G1) using the `scipy` library.

```python
def pdf_G1(x, mu, sigma2):
    "Univariate Gaussian pdf"
    pdf_values = sp.stats.norm.pdf(x, loc=mu, scale=np.sqrt(sigma2))
    return pdf_values
```

Computers typically represent functions *numerically* by their values at a set of grid points (nodes),
an approach called ***discretization***.

```python
bounds = -20, 20
N = 201                         # num of grid points
grid1d = np.linspace(*bounds,N) # grid
dx = grid1d[1] - grid1d[0]      # grid spacing
```

Feel free to return here later and change the grid resolution to see how
it affects the cells below (after re-running them).

The following code plots the Gaussian pdf.

```python
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
```

#### Exc – parameter influence

Experiment with `mu` and `sigma` to answer these questions:

- How does the pdf curve change when `mu` changes? (Several options may be correct or incorrect)
<details style="border: 1px solid #aaaaaa; border-radius: 4px; padding: 0.5em 0.5em 0;">
<summary style="font-weight: normal; font-style: italic; margin: -0.5em -0.5em 0; padding: 0.5em;">
  Click to view options 🔍
</summary>

  1. It changes the curve into a uniform distribution.
  1. It changes the width of the curve.
  1. It shifts the peak of the curve to the left or right.
  1. It changes the height of the curve.
  1. It transforms the curve into a binomial distribution.
  1. It makes the curve wider or narrower.
  1. It modifies the skewness (asymmetry) of the curve.
  1. It causes the curve to expand vertically while keeping the width the same.
  1. It translates the curve horizontally.
  1. It alters the kurtosis (peakedness) of the curve.
  1. It rotates the curve around the origin.
  1. It makes the curve a straight line.
</details>

- How does the pdf curve change when you increase `sigma`?  
  Refer to the same options as the previous question.
- In a few words, describe the shape of the Gaussian pdf curve.
  Does this remind you of anything? *Hint: it should be clear as a bell!*

**Exc – Implementation:** Change the implementation of `pdf_G1` so that it does not use `scipy`, but instead uses your own code (with `numpy` only). Re-run all of the above cells and check that you get the same plots as before.  
*Hint: `**` is the exponentiation/power operator, but $e^x$ is more efficiently computed with `np.exp(x)`*

```python
# show_answer('pdf_G1')
```

**Exc – Derivatives:** Recall $p(x) = \NormDist(x \mid \mu, \sigma^2)$ from eqn. (G1).  
Use pen, paper, and calculus to answer the following questions,
which will help you remember some key properties of the distribution.

- (i) Find $x$ such that $p(x) = 0$.
- (ii) Where is the location of the **mode (maximum)** of the density?  
  I.e. find $x$ such that $\frac{d p}{d x}(x) = 0$.
  *Hint: begin by writing $p(x)$ as $c e^{- J(x)}$ for some $J(x)$.*
- (iii) Where is the **inflection point**? I.e. where $\frac{d^2 p}{d x^2}(x) = 0$.
- (iv) *Optional*: Some forms of *sensitivity analysis* (typically for non-Gaussian $p$) consist in estimating/approximating the Hessian, i.e. $\frac{d^2 \log p}{d x^2}$. Explain what this has to do with *uncertainty quantification*.

<a name="Exc-(optional)----Change-of-variables"></a>

#### Exc (optional) – Change of variables

Let $U = \phi(X)$ for some monotonic function $\phi$,
and let $p_x$ and $p_u$ be their probability density functions (pdf).

- (a): Show that $p_u(u) = p_x\big(\phi^{-1}(u)\big) \frac{1}{|\phi'(u)|}$,
- (b): Show that you don't need to derive the density of $u$ in order to compute its expectation, i.e. that
  $$ \Expect[U] = \int  \phi(x) \, p_x(x) \, d x ≕ \Expect[\phi(x)] \,,$$
  *PS: this result is [pretty intuitive](https://en.wikipedia.org/wiki/Law_of_the_unconscious_statistician),
  and also holds for non-injective transformations, $\phi$,
  as well as functions of multiple random variables.*

```python
# show_answer('CVar in proba')
```

<a name="Exc-(optional)----Integrals"></a>

#### Exc (optional) – Integrals

Recall $p(x) = \NormDist(x \mid \mu, \sigma^2)$ from eqn. (G1). Abbreviate it as $c = (2 \pi \sigma^2)^{-1/2}$.  
Use pen, paper, and calculus to show that

- (i) the first parameter, $\mu$, indicates its **mean**, i.e. that $$\mu = \Expect[X] \,.$$
  *Hint: you can rely on the result of (iii)*
- (ii) the second parameter, $\sigma^2>0$, indicates its **variance**,
  i.e. that $$\sigma^2 = \mathbb{Var}(X) \mathrel{≔} \Expect[(X-\mu)^2] \,.$$
  *Hint: use $x^2 = x x$ to enable integration by parts.*
- (iii) $c$ is indeed the right normalizing constant, i.e. that
  $$E[1] = 1 \,.$$
  *Hint: Neither Bernoulli and Laplace managed this,
  until [Gauss (1809)](#References) did by first deriving $(E[1])^2$.
  Here is a nice [video demonstration by 3Blue1Brown](https://www.youtube.com/watch?v=cy8r7WSuT1I&t=3m52s).*

```python
# show_answer('Gauss integrals')
```

**Exc (optional) – Riemann sums**:
Recall that integrals (for example for the mean and variance)
compute an "area under the curve".
On a discrete grid, integrals can be approximated using the [Trapezoidal rule](https://en.wikipedia.org/wiki/Riemann_sum#Trapezoidal_rule).

- (a) Replace `np.trapezoid` below with your own implementation (using `sum()`).
- (b) Use `np.trapezoid` to compute the probability that a scalar Gaussian $X$ lies within $1$ standard deviation of its mean.
  *Hint: the numerical answer you should find is $\mathbb{P}(X \in [\mu {-} \sigma, \mu {+} \sigma]) \approx 68\%$.*

```python
def mean_and_var(pdf_values, grid):
    f, x = pdf_values, grid
    mu = np.trapezoid(f*x, x)
    s2 = np.trapezoid(f*(x-mu)**2, x)
    return mu, s2

mu, sigma = 0, 2 # example
pdf_vals = pdf_G1(grid1d, mu=mu, sigma2=sigma**2)
'Should equal mu and sigma2: %f, %f' % mean_and_var(pdf_vals, grid1d)
```

```python
# show_answer('Riemann sums', 'a')
```

**Exc – The uniform pdf**:
Below is the pdf of the [uniform/flat/box distribution](https://en.wikipedia.org/wiki/Uniform_distribution_(continuous))
for a given mean and variance.

- Use `mean_and_var()` to verify `pdf_U1` (as is).
- Replace `_G1` with `_U1` in the code generating the above interactive plot.
- Why are the walls (ever so slightly) inclined?
- Write your own implementation below, and check that it reproduces the `scipy` version already in place.

```python
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
```

```python
# show_answer('pdf_U1')
```

## The multivariate (i.e. vector) Gaussian

A *multivariate* random variable, i.e. a **vector**, is simply a collection of scalar variables (on the same probability space).
Its distribution is the *joint* distribution of its components.
The pdf of the multivariate Gaussian $\X$ (for any dimension $\ge 1$) is

$$\large \NormDist(\x \mid \mathbf{\mu}, \mathbf{\Sigma}) =
|2 \pi \mathbf{\Sigma}|^{-1/2} \, \exp\Big(-\frac{1}{2}\|\x-\mathbf{\mu}\|^2_\mathbf{\Sigma} \Big) \,, \tag{GM} $$
which is very similar to the univariate (scalar) case (G1),
but with $|.|$ representing the matrix determinant,
and $\|.\|_\mathbf{W}$ representing the weighted 2-norm: $\|\x\|^2_\mathbf{W} = \x^T \mathbf{W}^{-1} \x$.  

<details style="border: 1px solid #aaaaaa; border-radius: 4px; padding: 0.5em 0.5em 0;">
<summary style="font-weight: normal; font-style: italic; margin: -0.5em -0.5em 0; padding: 0.5em;">
  $\mathbf{W}$ must be symmetric-positive-definite (SPD) because ... (optional reading 🔍)
</summary>

- The norm (a quadratic form) is invariant to any asymmetry in the weight matrix.
- The density (GM) would not be integrable (over $\Reals^{d}$) if $\x\tr \mathbf{\Sigma} \x > 0$.

- - -
</details>

Moreover, [as above](#Exc-(optional)----Integrals), it can be shown that

- $\mathbf{\mu} = \Expect[\X]$,
- $\mathbf{\Sigma} = \Expect[(\X-\mu)(\X-\mu)\tr]  =: \mathbb{Cov}(\X)$.

As such, $\mathbf{\Sigma}$ is called the **covariance matrix**,
whose individual elements are individual covariances,
$\Sigma_{i,j} = \Expect[(X_i-\mu_i)(X_j-\mu_j)] =: \mathbb{Cov}(X_i, X_j)$,
and – on the diagonal – variances: $\Sigma_{i,i} = \mathbb{Var}(X_i)$.

The following implements the pdf (GM).

```python
import numpy.linalg as la

def pdf_GM(x, mu, Sigma):
    "pdf – Gaussian, Multivariate: N(x | mu, Sigma) for each x."
    c = np.sqrt(la.det(2*np.pi*Sigma))
    return 1/c * np.exp(-0.5*weighted_norm22(x - mu, Sigma))

def weighted_norm22(points, cov):
    "Computes the weighted norm of each vector (a row in `points`)."
    W = la.inv(cov) # NB: replace by la.solve() in real applications!
    return np.sum( (points @ W) * points, axis=-1)
```

The norm implementation is a bit tricky because it uses `@` (matrix multiplication), `*` (array, i.e. element-wise multiplication) and `axis=-1` (sum along the last dimension) to enable inputting the entire lattice/grid at once without shape manipulation.

```python
grid2d = np.dstack(np.meshgrid(grid1d, grid1d))
grid2d.shape
```

The following code plots the pdf as contour (level) curves.

```python
@interact(corr=(-1, 1, .001), std_x=(1e-5, 10, 1), seed=(0, 9))
def plot_pdf_G2(corr=0.7, std_x=1, seed=0):
    mu = 0
    var_x = std_x**2
    var_y = 1
    cv_xy = np.sqrt(var_x * var_y) * corr

    # Assemble covariance matrix (C)
    C = 25 * np.array([[var_x, cv_xy],
                       [cv_xy, var_y]])

    # Evaluate (compute)
    density_values = pdf_GM(grid2d, mu=mu, Sigma=C)

    # Plot
    plt.figure(figsize=(4, 4))
    plt.contour(grid1d, grid1d, density_values, cmap="plasma",
                # Because built-in heuristical levels cause animation noise:
                levels=np.linspace(1e-4, 1/np.sqrt(la.det(2*np.pi*C)), 11))

    # See exc. below
    if seed and 'sample_GM' in globals():
        plt.scatter(*sample_GM(mu, C=C, N=100, rng=seed))

    plt.axis('equal');
    plt.show()
```

Note that the code defines the covariance `cv_xy` from the input ***correlation*** `corr`.
This is a coefficient (number),
defined for any two random variables $X$ and $Y$ (not necessarily Gaussian) as
$$ \rho[X,Y]=\frac{\mathbb{Cov}[X,Y]}{\sigma_x \sigma_y} \,.$$
Equivalently, it is the covariance between the *standardized* variables,
i.e. $\rho[X,Y] = \mathbb{Cov}[X / \sigma_x, Y / \sigma_y]$.
It quantifies (defines) the ***linear dependence*** between $X$ and $Y$,
as illustrated by the following exercises.

**Exc – Correlation influence:** How do the contours look? Try to understand why. Cases:

- (a) correlation=0.
- (b) correlation=0.99.
- (c) correlation=0.5. (Note that we've used `plt.axis('equal')`).
- (d) correlation=0.5, but with non-equal variances.

Finally (optional): why does the code "crash" when `corr = +/- 1`? Is this a good or a bad thing?  

More generally, it can be shown that $\rho^2$ is the proportion of the variance of $Y$
captured/explained by a simple linear regression from $X$.

<a name="Exc-–-correlation-extremes"></a>

**Exc (optional) – Correlation extremes**

Show that

- (a) $\rho[X,Y] = 0$ if $X$ and $Y$ are independent.
- (b) $\rho = 1$ if $Y = a X$ for some $a > 0$.
- (c) $\rho = -1$ if $Y = a X$ for some $a < 0$.

Otherwise, it can be shown by Cauchy-Swartz, that $-1\leq \rho \leq 1$.

```python
# show_answer('Correlation extremes', 'a')
```

**Exc Correlation game:** [Play](http://guessthecorrelation.com/) until you get a score (gold coins) of 5 or more.  

**Exc – Correlation disambiguation:**

- What's the difference between correlation and covariance (in a single sentence)?
- What's the difference between non-zero (C) correlation (or covariance) and (D) dependence?
  *Hint: consider this [image](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient#/media/File:Correlation_examples2.svg).*  
  - Does $C \Rightarrow D$ or the converse?  
  - What about the negation, $\neg D \Rightarrow \neg C$, or its converse?*  
  - What about the (jointly) Gaussian case?
- Does correlation (or dependence) imply causation?
- Suppose $x$ and $y$ have non-zero correlation, but neither one causes the other.
  Does information about $y$ give you information about $x$?

<a name="Exc-–-linear-algebra-of-with-random-variables"></a>

#### Exc – linear algebra with random variables

- (a) Prove the linearity of the expectation operator:
  $\Expect[a X + Y] = a \Expect[X] + \Expect[Y]$.
- (b) Thereby, show that $\mathbb{Var}[ a  X + Y ] = a^2 \mathbb{Var} [X] + \mathbb{Var} [Y]$
  if $X$ and $Y$ are independent.  
- (c) Similarly, prove:
  $\mathbb{Cov}[ \vect{A} \, \vect{X} + \vect{Y} ] = \mat{A} \, \mathbb{Cov} [\vect{X}] \, \mat{A}\tr + \mathbb{Cov}[\vect{Y}]$ if $\vect{X}$ and $\vect{Y}$ are independent.
- (d – *optional*) If $X$ and $Y$ are Gaussian, then so is $X + Y$.
  Proof in the [next tutorial](T3%20-%20Bayesian%20inference.ipynb#Exc-–-BR-LG1). Meanwhile watch the [`3blue1brown` video](https://www.youtube.com/watch?v=d_qvLDhkg00&t=266s&ab_channel=3Blue1Brown).
- (e) Let $\vect{Z} \sim \NormDist(\vect{0}, \I)$,  where $\I$ is the identity matrix.
  Show that each component, $Z_i$, is independent of all others.

```python
# show_answer('RV linear algebra', 'a')
```

#### Exc – Gaussian (multivariate) sampling

Pseudo-random samples of $Z_i\sim \NormDist(0, 1)$
can be generated on any modern computer using one of [these algorithms](https://en.wikipedia.org/wiki/Normal_distribution#Computational_methods).
As shown above, the "transformation" $\X = \mat{L} \vect{Z} + \mu$
yields $\vect{X} \sim \NormDist(\mu, \mat{C})$,
which can be used to sample with a desired mean and covariance, $\mat{C} = \mat{L}^{}\mat{L}^T$.
Indeed, the code below samples $N$ realizations of $\X$,
and assembles them as columns in an "ensemble matrix", $\E$.
But `for` loops are slow in Python (and Matlab).
Replace it with something akin to `E = mu + L@Z`.
*Hint: this snippet will fail because it's trying to add a vector to a matrix.*

```python
def sample_GM(mu=0, L=None, C=None, N=1, reg=0, rng=rnd):
    # Seed random number generator
    if isinstance(rng, int):
        rng = rnd.default_rng(seed=rng)

    # Compute L from C (if needed)
    if L is None:
        from numpy.linalg import cholesky
        if reg:
            C = C + reg * np.eye(len(C))
        L = cholesky(C)

    d = len(L) # len (number of dims) of x
    Z = rng.standard_normal((N, d)).T

    # Ensure mu is 1d
    if np.isscalar(mu):
        mu = mu * np.ones(d)

    # Using a loop ("slow"):
    E = np.zeros((d, N))
    for n in range(N):
        E[:, n] = mu + L @ Z[:, n]

    return E
```

Go back up to the interactive illustration of the 2D Gaussian distribution re-run its cell to check (eyeball measure) your implementation.

```python
# show_answer('Broadcasting')
```

**Exc (optional) – Gaussian ubiquity:** Why are we so fond of the Gaussian assumption?

```python
# show_answer('Why Gaussian')
```

## Summary

The Normal/Gaussian distribution is bell-shaped.
Its parameters are the mean and the variance.
In the multivariate case, the mean is a vector,
while the second parameter becomes a covariance *matrix*,
whose off-diagonal elements represent scaled correlation factors,
which measure *linear* dependence.

### Next: [T3 - Bayesian inference](T3%20-%20Bayesian%20inference.ipynb)

<a name="References"></a>

### References

<!--
@book{laplace1820theorie,
title={Th{\'e}orie analytique des probabilit{\'e}s},
author={de Laplace, Pierre Simon},
volume={7},
year={1820},
publisher={Courcier}
}

@book{gauss1877theoria,
title={Theoria motus corporum coelestium in sectionibus conicis solem ambientium},
author={Gauss, Carl Friedrich},
volume={7},
year={1877},
publisher={FA Perthes}
}
-->

- **Laplace (1812)**: P. S. Laplace, "Théorie Analytique des Probabilités", 1812.
- **Gauss (1809)**: Gauss, C. F. (1809). *Theoria Motus Corporum Coelestium in Sectionibus Conicis Solem Ambientium*. Specifically, Book II, Section 3, Art. 177-179, where he presents the method of least squares (which will be very relevant to us) and its probabilistic justification based on the normal distribution of errors.

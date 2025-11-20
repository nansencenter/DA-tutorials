---
jupyter:
  jupytext:
    cell_metadata_filter: -all
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
from resources import show_answer, interact,  import_from_nb, nonchalance, colorbar
%matplotlib inline
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy.random as rnd
import scipy.linalg as sla
from mpl_tools.place import freshfig
plt.ion();
```

# T7 - Spatial statistics ("geostatistics") & kriging

In T4 and T5, we performed state estimation for scalar and vector (length-4) noisy, linear AR(1) processes,
while T6 illustrated several nonlinear and chaotic dynamics, including the 1D (length-40) Lorenz-96 system.
We now turn to estimating static (non-dynamic) 1D, 2D, and 3D spatial fields
using the method known (in geostatistics) as **kriging**,
**radial basis function (RBF)** interpolation (in mathematics),
and **Gaussian process (GP)** regression (in machine learning).
$
\newcommand{\mat}[1]{{\mathbf{{#1}}}}
\newcommand{\vect}[1]{{\mathbf{#1}}}
\newcommand{\Expect}[0]{\mathbb{E}}
\newcommand{\NormDist}{\mathscr{N}}
\newcommand{\trsign}{{\mathsf{T}}}
\newcommand{\tr}{^{\trsign}}
$

Once again, randomness will serve as a stand-in for uncertainty and will generate our synthetic experiments.
Conceptually, little is new – once discretized and flattened, spatial fields are simply long vectors of unknowns, which we can estimate using covariances very similarly as in the Kalman filter.
The challenge lies in their scale: such long vectors lead to large and unwieldy covariance matrices.
Moreover, rather than treating the covariance matrix as merely a symmetric table of numbers,
we will explicitly model it through spatial relationships and distances
between field elements. Such covariance modeling will also be useful for the EnKF in that they can encode initial priors, and the experience gained will help in understanding covariance localization.

As always, we discretize the domains.
The following sets up a grid/lattice for fields on a 1D domain.

```python
grid1D = np.linspace(0, 1, 21)
```

The following sets up a 2D grid.

```python
grid2x, grid2y = np.meshgrid(grid1D, grid1D)
grid2x.shape  # `grid2y` has same shape
```

Now we "flatten" (a.k.a. "(un)ravel", "vectorize", or "string out") this explicitly 2D grid of nodes into a list of 2D points.

```python
grid2D = np.dstack([grid2x, grid2y]).reshape((-1, 2))
grid2D.shape
```

## Random fields and variograms

Denote a *physical* location (i.e., a 1D/2D/3D coordinate) by $s$.
A ***random field***, $X(s)$, is a function that takes random values at each point $s$.
The near-synonymous term *regionalized variable* emphasizes the continuity of $s$.

Random fields are commonly assumed to be ***stationary***,
meaning that the dependence between $X(s_{1})$ and $X(s_{2})$
for any two locations $s_1, s_2$ can be described solely in terms of their separation or,
assuming isotropy (rotational symmetry), their distance $d = \| s_{1} - s_{2} \|$.

For now, we assume that the mean $\Expect X(s)$ is known and, by stationarity, constant,
leaving the covariance as the most important descriptor.
Again, by stationarity, this depends only on distance,
so we can describe the full covariance of the field solely in terms of the
"(auto-)***covariance function***": $$C(d) = \mathbb{Cov}[X(s_{1}), X(s_{2})] \,.$$
<details style="border: 1px solid #aaaaaa; border-radius: 4px; padding: 0.5em 0.5em 0;">
  <summary style="font-weight: normal; font-style: italic; margin: -0.5em -0.5em 0; padding: 0.5em;">
    Geostatistics usually works with a reformulation of $C(d)$ called "(semi-)<strong>variogram"</strong>,
    which flips the graph of $C(d)$ on its head ... (optional reading 🔍):
    $$\gamma(d) = C(0) - C(d)$$
  </summary>

  Not all functions are valid variograms; they must evidently be non-negative,
  have the value $0$ at $d=0$ (why?),
  and result in positive-definite covariance matrices.
  A sufficient condition is that the corresponding covariance is convex and decays
  (but does not reach) toward zero at infinity.
  However, covariance functions that oscillate around zero (Matérn)
  or have finite support (triangular) also exist.

  A wider definition of the variogram also exists:
  half (whence the optional "semi" prefix) the variance of the increment, i.e. of $X(s_{1}) - X(s_{2})$.
  This definition only requires second-order stationarity of the *increments*,
  which is a weaker requirement called "intrinsic stationarity",
  and does not require or imply the existence of a covariance function $C(d)$.
  Thus, it can be used to estimate infinite-variance (heavy-tail or Brownian-motion-like) processes,
  but only if the method constrains the sum of its weights to one,
  making the linear combination of the estimate "authorized/allowable."
  This condition coincides with that imposed by an unknown mean
  and does not represent a severe restriction.
  - - -
</details>

Turning things around, we can start by prescribing a *model* variogram (i.e., covariance)
that we believe the random field has.
Some of the most commonly used examples are implemented below.

```python
def variograms(power=1.5, transf01="expo", Range=0.3, nugget=0, sill=1):
    """Create variogram (function) for the given parameters."""
    def vg(dists):
        dists = np.asarray(dists)
        dists = dists / Range
        gamma = transforms01[transf01](dists**power)
        gamma *= sill
        gamma = np.where(dists != 0, nugget + (1 - nugget) * gamma, gamma)
        return gamma
    return vg
```

Here, `transforms01` monotonically decreasingly maps $[0, +\infty]$ to $[0, 1]$ .

```python
transforms01 = {
    "none":  lambda x: x,               # identity
    "expo":  lambda x: 1 - np.exp(-x),  # similar to Gaussian densities
    "hyper": lambda x: 1 - 1 / (1 + x), # similar to Cauchy densities
    "clip":  lambda x: x.clip(max=1)    # similar to "hat" function
}
vg_params = dict(transf01=list(transforms01), power=(0.01, 4, .01),
                 Range=(.01, 4), nugget=(0, 1, .01), sill=(0.1, 5))
```

Below is a visual illustration.

```python
@interact(**vg_params)
def plot_variograms(power=1.5, Range=0.3, nugget=0, sill=1):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.set_ylim(0, 1.05 * sill)
    ax.set_xlim(left=0)
    for i, transf01 in enumerate(transforms01):
        vg = variograms(power, transf01, Range, nugget, sill)
        ax.plot(grid1D, vg(grid1D), lw=5-i, color=f"C{i+1}", label=transf01)
    plt.xlabel("distance/lag")
    ax.legend(title="transf 0→1", loc="lower right")
    plt.show()
```

Now that we're in possession of a valid variogram,
we can compute a covariance matrix between any sets of points as follows.

```python
def covar_modelled(coords, variogram):
    C0 = variogram(np.inf)
    assert C0 < np.inf, "Not a valid covariance"
    dists = dist_euclid(coords, coords)
    return C0 - variogram(dists)
```

We also need to be able to compute pair-wise distances for each (discretized) location of our field.
The following computes distances between two sets (not necessarily of same length) of points.

```python
def dist_euclid(A, B):
    """l2-norm between each point (row) of A and B"""
    # Like scipy.spatial.distance.pdist(A) + `squareform` if A==B
    if A.ndim == 1:
        return abs(A[:, None] - B)
    d = A[:, None, :] - B
    d2 = np.sum(d**2, axis=-1)
    return np.sqrt(d2)
```

## Gaussian fields

<details style="border: 1px solid #aaaaaa; border-radius: 4px; padding: 0.5em 0.5em 0;">
  <summary style="font-weight: normal; font-style: italic; margin: -0.5em -0.5em 0; padding: 0.5em;">
    Gaussianity becomes even more useful when working with spatial fields ... (optional reading 🔍)
  </summary>
  Non-Gaussian random fields, such as those with Student-t or Laplace marginals, exist,
  but Gaussianity makes sampling and conditioning uniquely
  tractable, even for large vectors, since everything reduces to linear algebra.

  Besides, although **Markov** random fields of any dimension exist for many
  distributions (all in the Gibbs family), the familiar **locality** of the
  first-order autoregressive (AR(1)) process—that the conditionals are Markov,
  have linear expectation (w.r.t. the neighboring value), and constant variance (idem)—
  requires Gaussianity in higher dimensions.
  To be clear, the 1D field defined by a simple (and physically interpretable) AR(1) process
  $x_{k+1} = a x_k + \epsilon_k$ exhibits these properties for *any* distribution by construction.
  But in 2D or higher-dimensional lattices, such "directed autoregressive" constructions become
  anisotropic and dependent on the chosen ordering, losing stationarity.
  Instead, the AR(1) properties can be salvaged by imposing them directly on the conditionals but,
  as shown by Besag (1974), Gaussianity is required to consistently define a global (joint) distribution.
  The Markov (sparse) structure also makes the sampling and conditioning even more efficient.
  - - -
</details>
and so will be assumed in what follows.
Recall that Gaussian random variables (including vectors and fields)
are fully specified by their mean and covariance.
Thus, once we are in possession of a covariance matrix,
we can sample discrete fields as random vectors, just as we did previously.

```python
sample_GM, = import_from_nb("T2", ["sample_GM"])
```

Let's try it out.

### 1D example

```python
@interact(**vg_params, N=(1, 30), bottom=list(vg_params))
def sample_1D(power=1.5, transf01="expo", Range=0.3, nugget=1e-2, sill=1, N=10):
    variogram = variograms(power, transf01, Range, nugget, sill)
    C = covar_modelled(grid1D[:, None], variogram)
    fields = sample_GM(C=C, N=N, reg=1e-12, rng=3000)

    fig, axs = plt.subplots(figsize=(9, 4), ncols=2)
    colorbar(fig, axs[0].matshow(C, cmap="inferno", vmin=0, vmax=sill), axs[0])
    axs[0].grid(False)
    axs[1].plot(grid1D, fields)
    plt.show()
```

#### Exc – impact of variogram parameters

Refer to both of the above interactive widgets to answer the following.

- (a) What happens to the covariance matrix and the fields when `Range` $\rightarrow \infty$ ?
- (b) What about `Range` $\rightarrow 0$ ?
- (c) Can you reproduce this using `nugget` somehow ?
- (d) Try `nugget = 0` and `0.01`. Explain why the impact is much more noticeable
  in the Gauss and Cauchy cases.
- (e) How does changing `sill` affect the fields? Can you express `sill` in terms of $C(d)$ somehow? Thus, what other name could we give it?

```python
# show_answer('variogram params')
```

### 2D example

```python
vg = variograms(2, "expo", Range=1, nugget=0, sill=1)
C = covar_modelled(grid2D, vg)
C.shape
```

Note that the size of the covariance matrix is the square of a single 2D field's size.
It's interesting to inspect the covariance matrix in this 2D case,
which has a nested structure.

```python
fig, ax = freshfig("2D covar")
C0 = vg(np.inf)
ax.matshow(C, cmap="inferno", vmin=0, vmax=C0);
ax.grid(False)
```

Now let's generate and plot some realizations of the corresponding Gaussian
random field.

```python
def plot2d(ax, field, contour=True, show_obs=True, cmap="PiYG"):
    vmin = -3.5*np.sqrt(C0)
    vmax = +3.5*np.sqrt(C0)

    ax.set(aspect="equal", xticks=[0, 1], yticks=[0, 1])

    if show_obs:
        ax.plot(*obs_locations.T, "ko", ms=4)
        ax.plot(*obs_locations.T, "yo", ms=1)

    field = field.reshape(grid2x.shape)
    if contour:
        ε = 1e-12 # fudge to center colors right below 0
        levels = np.arange(vmin - ε, vmax + .5 + ε, .5)
        return ax.contourf(field, levels=levels, extent=(0, 1, 0, 1), extend="both", cmap=cmap)
    else:
        return ax.pcolormesh(grid2x, grid2y, field, shading="auto", vmin=vmin, vmax=vmax, cmap=cmap)
```

Beware that `contourf` (which works with regularly spaced grids) uses simple bilinear interpolation
between nodes, making the fields appear higher resolution than they actually are.
Therefore, you may want to use `contour=False` instead to view the actual grid "pixels".

```python
@interact(**vg_params)
def sample_2D(power=1.5, transf01="expo", Range=0.3, nugget=1e-2):
    fig, axs = plt.subplots(figsize=(8, 4), nrows=2, ncols=4, sharex=True, sharey=True)
    C = covar_modelled(grid2D, variograms(power, transf01, Range, nugget, sill=1))
    fields = sample_GM(C=C, N=len(axs.ravel()), reg=1e-12, rng=3000)

    for ax, field in zip(axs.ravel(), fields.T):
        cb = plot2d(ax, field, show_obs=False)

    fig.tight_layout()
    colorbar(fig, cb, axs)
    plt.show()
```

## Estimation

Denote $x = X(s)$ the random field at a single location,
$\vect{x}$ the random field at all grid locations,
and $\vect{y}$ the field at the observation locations.

### Problem (2D)

For our estimation/interpolation target ($\vect{x}$) we sample using the default covariance matrix defined above.

```python
truth = sample_GM(C=C, N=1, reg=1e-12).squeeze()
```

For our observations ($\vect{y}$) we randomly pick some grid points:

```python
nObs = 10
rnd.seed(3000)
obs_indices = rnd.choice(len(grid2D), nObs, replace=False)
```

...such that the observation locations and values are simply:

```python
obs_locations = grid2D[obs_indices]
observations = truth[obs_indices]
```

However (unlike classical bilinear/bicubic interpolation),
the following methods work perfectly well with irregular observations,
irregular grids, or different orderings of the flattened grid nodes.

### Methods

The following is a register of fields to include in plot below.

```python
estims = dict(Truth=truth)
```

Each time you (have tried to) implement one of the following methods,
come back up here and re-run this next cell
to visually compare and assess your method.

```python
fig, axs = freshfig(figsize=(8, 4), squeeze=False, nrows=2, ncols=len(estims), sharex=True, sharey=True)
axs[0, 0].set_ylabel("Estimate");
axs[1, 0].set_ylabel("Errors");

for name, ax1, ax2 in zip(estims, *axs):
    x = estims[name]
    ax1.set(title=name)
    cb1 = plot2d(ax1, x)
    cb2 = plot2d(ax2, x - truth, cmap="seismic")

fig.tight_layout()
colorbar(fig, cb1, axs[0]);
colorbar(fig, cb2, axs[1]);
```

Pre-compute some objects that see repeated use.

```python
dists_yy = dist_euclid(obs_locations, obs_locations)
dists_xy = dist_euclid(grid2D, obs_locations)
```

The cells below contain snippets of different spatial interpolation methods
followed by a cell that plots the interpolants.
Complete the code snippets.

#### Exc: Nearest neighbour interpolation

Implement the method [(Wikipedia)](https://en.wikipedia.org/wiki/Nearest-neighbor_interpolation)
by adjusting the code below.

```python
nearest_obs = np.zeros(len(grid2D), dtype=int)  ### FIX THIS ###
estims["Nearest-n."] = observations[nearest_obs]
```

```python
# show_answer('nearest neighbour interp')
```

#### Exc: Inverse distance weighting

Implement the method [(Wikipedia)](https://en.wikipedia.org/wiki/Inverse_distance_weighting).\
*Hint: The `errstate` context silences warnings due to division by 0 (whose special case is treated further down).*

```python
exponent = 3
with np.errstate(invalid='ignore', divide='ignore'):
    weights = np.zeros_like(dists_xy)  ### FIX THIS ###    
estims["Inv-dist."] = weights @ observations # Apply weights
estims["Inv-dist."][obs_indices] = observations # Fix singularities
```

```python
# show_answer('inv-dist weight interp')
```

#### Kriging

Kriging finds the best (minimum) mean square error (MSE), $\Expect (\widehat{x} - x)^2$, among all linear "predictors",
$\widehat{x} = \vect{w}\tr \vect{y}$, that are unbiased (BLUP).

<a name='Exc-–-"simple"-kriging-(SK)'></a>

#### Exc – "simple" kriging (SK)

Suppose $X(s)$ has a mean that is constant in space, $\mu$, and known.
Since it is easy to subtract (and later re-include) $\mu$
from both $x$ and the data $\vect{y}$, we simply assume $\mu = 0$.
Thus, $\Expect \vect{y} = \vect{0}$ and $\Expect \widehat{x} = 0$ for any weights $\vect{w}$,
and so $\widehat{x}$ is already unbiased.
Meanwhile,
$$
\begin{align}
  \text{MSE}
  % \Expect \big( \widehat{x} - x \big)^2
  &= \Expect \big( \vect{w}\tr \vect{y} - x \big)^2 \\
  % &= \Expect \big( \vect{w}\tr \vect{y} \vect{y}\tr \vect{w} - 2 x \vect{y}\tr \vect{w} + x^2 \big) \\
  &= \vect{w}\tr \Expect \big( \vect{y} \vect{y}\tr \big) \vect{w}
     - 2 \Expect \big( x \vect{y}\tr \big) \vect{w}
     + \Expect \big( x^2 \big) \\
  &= \vect{w}\tr \mat{C}_{\vect{y} \vect{y}} \vect{w}
     - 2 \vect{c}_{x \vect{y}}\tr \vect{w}
     + C(0) \,,
\end{align}
$$
where $C(0)$ is the variance ("sill") of the field.
The minimum can be found by setting the derivative with respect to $\vect{w}$ to zero,
yielding $\vect{w}_{\text{SK}} = \mat{C}_{\vect{y} \vect{y}}^{-1} \, \vect{c}_{x \vect{y}}$

<details style="border: 1px solid #aaaaaa; border-radius: 4px; padding: 0.5em 0.5em 0;">
  <summary style="font-weight: normal; font-style: italic; margin: -0.5em -0.5em 0; padding: 0.5em;">
    We have already seen this formula in the
    <a href="T5%20-%20Multivariate%20Kalman%20filter.ipynb#Regression-perspective">regression perspective on the Kalman gain (T5)</a>,
    which also noted the Gauss-Markov theorem which states that it is optimal in the BLUE sense.
    The perspective of radial basis functions (RBF) is also informative (optional reading...)
  </summary>

  The BLUP criterion is similar to the best linear unbiased *estimator* (BLUE),
  and so the equivalence of the formulae may not come as a surprise.
  Still, even though the MSE optimized by the BLUP can be construed as a *variance* (by the unbiasedness constraint),
  it should not be confused with the variance minimized by the BLUE, which targets a *fixed* (but unknown) quantity
  (one conventionally "estimates" fixed effects and "predicts" random effects).
  Indeed, fruitfully applying the BLUE criterion to the kriging setting is difficult.
  For example, the additional randomness (i.e., in the estimand $x$ as well as in the observations $\vect{y}$)
  makes it so that if the mean $\mu$ is known – effectively zero – then the BLUE is simply zero
  (it is unbiased and has zero variance)!
  Or, if $\mu$ is unknown, we can assume $\vect{y} = x + \vect{r}$ in order to define unbiasedness,
  which then requires that the weights sum to one, i.e. $\vect{w}\tr\vect{1} = 1$.
  Then, if $\mat{C}_{\vect{y}\vect{y}}$ is the covariance of $\vect{r}$,
  the BLUE weights become $(\vect{1}\tr \mat{C}_{\vect{y}\vect{y}}^{-1})/(\vect{1}\tr \mat{C}_{\vect{y}\vect{y}}^{-1} \vect{1})$,
  and so $\widehat{x}(s)$ is constant in $s$, i.e. flat.
  Thus we see the need to include the randomness of $x$ along with that of $\vect{y}$.
  This seems contrary to the classical BLUE framing,
  but we have already seen it done for the Kalman gain (by augmenting the observations by the prior mean).
  Even so, claims of BLUE optimality would be limited to cases where we could find
  covariances and operators that fit with the variogram-modelled covariance,
  which is very limiting.

  Now for the "dual" perspective of kriging, which is held by **radial basis function (RBF) interpolation**.
  Ultimately, kriging (simple, ordinary, and universal) provides an estimate of the form
  $\widehat{x} = \vect{y}\tr (\mat{A}_\vect{y}^{-1} \vect{b}_{\vect{y} x})$,
  where $\vect{y}$ is the observations,
  and the subscripts indicate that $\mat{A}_\vect{y}$ depends on the locations of $\vect{y}$,
  while $\vect{b}_{\vect{y} x}$ depends on the locations of $\vect{y}$ and $x$ both.
  In the kriging perspective, $\vect{w} = \mat{A}_\vect{y}^{-1} \vect{b}_{\vect{y} x}$ is seen as the "weights" and get applied to $\vect{y}$:
  once solved for, they are easily re-applied for different values of $\vect{y}$.
  However, if we shift the parentheses (and computational order) onto the first two factors,
  then the linear solver computes $\vect{v} = \mat{A}_\vect{y}^{-1} \vect{y}$,
  fitting $N$ RBFs to $N$ data points, $\vect{y}$.
  Now the "weights", $\vect{v}$, are computed from the observations,
  but applied onto $\vect{b}_{\vect{y} x}$, which is a function of (the location) of $x$.
  As such we have not merely derived an estimate for a single location, $s$,
  but an *interpolant* (that is cheap to evaluate) for all locations.

  Lastly, why is kriging said to be **non-parametric**?
  Heuristically, note that – while the kernel itself is specified through a small number of parameters –
  kriging defines a separate linear regression problem for each and every location of the domain,
  of which there are infinitely (uncountably) many.
  More rigorously, by identifying the variogram with a kernel,
  we can appeal to the theory of reproducing kernel Hilbert spaces (RKHS) to kriging.
  In particular, kriging (GP) can be seen to solve an "empirical risk minimisation problem"
  with negligible measurement error (for interpolation).
  Thus, by the "representer theorem",
  – even though we're only solving for $N$ weights, $\vect{v}$ –
  the fitted surface is not restricted to a finite feature basis,
  but is the minimum in an infinite-dimensional function space.

  - - -
</details>

However, we previously derived this $\widehat{x}$ as the posterior/conditional mean of Gaussian distribution.
This perspective is that of GP regression,
but was also highlighted by [Krige (1951)](#References),
and makes it evident that kriging also provides an uncertainty estimate
(a major benefit compared to the other interpolation methods we tested above),
namely the posterior covariance matrix, $C(0) - \vect{c}_{x \vect{y}}\tr \mat{C}_{\vect{y} \vect{y}}^{-1} \vect{c}_{x \vect{y}}$.

Implement the method [(Wikipedia)](https://en.wikipedia.org/wiki/Kriging#Simple_kriging).  
*Hint: You may use `sla.inv`, but `sla.solve` is better, and `sla.lstsq` is even better.*

```python
C0 = vg(np.inf)
### FIX THIS ###
covar_yy = ...
cross_xy = ...
weights = np.zeros_like(dists_xy.T) # cross_yy "\" covar_yy
estims["S.Kriging"] = observations @ weights
```

```python
# show_answer('Simple kriging')
```

### More kriging (1D)

The above 2D case used the same variogram that generated the truth.
But in practice, we do not know this variogram (whose very existence is a theoretical assumption),
so it must also be estimated.
To that end, let's gain some understanding of the variogram's impact
on the resulting *estimate*. We focus on the 1D problem because its illustrations are clearer.

For our estimation target, we use $x(s) = \sin (s^2)$ for $s \in [0, L]$.

```python
def true(s):
    return np.sin(s**2)
```

The observations are taken at the following (configurable) locations:

```python
def gen_obs_loc(nObs, spacing, L):
    return L/2 + spacing * np.linspace(-L/2, L/2, nObs)
```

Visualisation:

```python
@interact(**vg_params, L=(1, 10, 0.1), nObs=(1, 100, 1), spacing=(0.01, 1), bottom=list(vg_params))
def kriging_1d(L=4, nObs=6, spacing=0.5, transf01="expo", power=1.5, Range=0.3, nugget=0, sill=1):
    # Experiment setup
    sx = np.linspace(0, L, 1001)       # locations of x (interpolation grid)
    sy = gen_obs_loc(nObs, spacing, L) # locations of y (observations points)
    truth = true(sx)
    obsvs = true(sy)

    # Plot setup
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_ylim(-2, 2)
    ax.plot(sx, truth, "-k", label="truth")
    ax.plot(sy, obsvs, 'ok', label='data', ms=12, zorder=9)

    # Kriging setup
    vg = variograms(power, transf01, Range=Range, nugget=nugget, sill=sill)
    dists_yy = dist_euclid(sy, sy)
    dists_xy = dist_euclid(sx, sy)

    with nonchalance():
        mu = 0
        ax.axhline(mu, c="k", lw=0.5)
        interp = mu + simple_kriging(vg, dists_xy, dists_yy, obsvs - mu)
        ax.plot(sx, interp, 'C1', label="S-krig", lw=5)

    ax.legend(loc='lower left', ncols=2)
    plt.show()
```

#### Exc: Impact of variogram on kriged fields

Add the simple kriging interpolant by copy-pasting your solution from above into the function below.
*Hint: Move (and de-indent) code out of the `nonchalance()` context if you want to be
able to view error messages (to debug errors).*

```python
def simple_kriging(vg, dists_xy, dists_yy, observations, mu):
    field_estimate = ...
    return field_estimate
```

- (a) What happens when `Range` $\to 0$ ? What about $\to \infty$? What about intermediate values?
- (b) Try the same with Gauss/Cauchy variograms. What is the main difference?

```python
# show_answer('Interpolant = f(Variogram)', 'a')
```

#### Exc (optional) – Ordinary kriging

Let us do away with the assumption that the mean of the field is known,
all the while retaining the assumption that it is constant in space.
The resulting method is called ordinary kriging.
In this case, unbiasedness of $\widehat{x} = \vect{w}\tr \vect{y}$ requires that the weights sum to one.
This can be imposed on the MSE minimization using a Lagrange multiplier $\lambda$,
yielding the augmented system to solve:
$$ \begin{pmatrix} \mat{C}_{\vect{y} \vect{y}} & \vect{1} \\ \vect{1}\tr & 0 \end{pmatrix} \begin{pmatrix} \vect{w} \\ \lambda \end{pmatrix}
= \begin{pmatrix} \vect{c}_{x \vect{y}} \\ 1 \end{pmatrix} \,.$$

Ordinary kriging can be reproduced by separately kriging the mean and the resulting residual field [(Wackernagel, 2013)](#References).

It is also important to note that the MSE – by its resemblance to intrinsic increments –
can be expressed in terms of variograms even when the covariance function does not exist,
and the same goes for the method of OK. Somewhat surprisingly, the linear system to be solved looks exactly the same, except with covariances replaced by variograms.
In this case, the unbiasedness requirement also manifests as a requirement
that any variance expressed as a linear combination of variograms be positive.
In the RBF perspective, this requirement guarantees the existence and uniqueness of the solution to the OK equations.

- (a) Implement the OK method and add it to the above plotting by adding the following to its code:
  ```python
  with nonchalance():
    interp = ordinary_kriging(vg, dists_xy, dists_yy, obsvs)
    ax.plot(sx, interp, 'C2', label="O-krig", lw=4)
  ```
- (b) Which variogram model(s) does OK "unlock"?
- (c) What is peculiar about the interpolant for each of the new variograms?
- (d) Set `nObs = 1`. What is the difference between the SK and OK interpolants?
- (e) Set `nObs = 2`. What is the difference between the SK and OK interpolants? Explore with a few different `spacing` values.

```python
# show_answer('Ordinary kriging', 'a')
```

#### Exc (optional) – Universal kriging

In addition to the flat/constant *feature* $\vect{1}$,
one can add additional regressors in a similar fashion to how the OK system augments the SK system of equations,
producing universal kriging (UK).
If the features are monomials, then they may well be called *trends*,
and the UK method is called intrinsic kriging (IK).
Their order $+1$ defines the maximum allowable order of the *generalized* covariance function
(for example, `power=3` "variogram" has order 2, requiring order 1 (linear) trends),
again rendering the augmented linear system well-posed.

```python
order = 1
features = [(lambda x: x**p) for p in range(order+1)]
```

- (a) Add the following to the plotting code above, and implement the requisite UK method.
  ```python
    with nonchalance():
        regressors = [f(sy) for f in features]
        regressands = [f(sx) for f in features]
        interp = universal_kriging(vg, dists_xy, dists_yy, obsvs, regressors, regressands)
        ax.plot(sx, interp, 'C3', label="U-krig", lw=3)
  ```
- (b) Add the interpolation method of cubic spine (from scipy; Google it!).\
  Can you make the UK interpolant reproduce this (in the interior)?
  What happens when you vary `Range`?
- (c) Although theoretically suboptimal
  (since it assumes a constant/flat mean, and does not support generalized covariance functions),
  the OK system is still solvable with a cubic variogram.
  Can you find experimental control settings that show the OK estimate does not
  exactly reproduce the spline solution?

```python
# show_answer('Universal kriging', 'a')
```

Further generalizations include co-kriging (vector-valued fields).
Also, since the kriging mean is smoother than any realization (has lower variance,
which favors but does not guarantee spatial/spectral smoothness),
conditional generation (simulation) of fields is frequently used.

## Summary

Random, (geo)spatial fields are often modeled by their (auto-)correlation function or
– more generally – the (semi-)variogram.
Covariances also form the basis of a family of spatial interpolation and approximation
methods known as kriging.

### Next: [T8 - Monte-Carlo & ensembles](T8%20-%20Monte-Carlo%20%26%20cov%20estimation.ipynb)

<a name="References"></a>

### References

<!--

@book{chiles2012geostatistics,
  title={Geostatistics: Modeling Spatial Uncertainty},
  author={Chil{\`e}s, J.P. and Delfiner, P.},
  isbn={9780470317839},
  series={Wiley Series in Probability and Statistics},
  url={https://books.google.no/books?id=tZl07WdjYHgC},
  year={2012},
  publisher={Wiley}
}

@book{wackernagel2013multivariate,
  title={Multivariate Geostatistics: An Introduction with Applications},
  author={Wackernagel, H.},
  isbn={9783662052945},
  year={2013},
  publisher={Springer Berlin Heidelberg}
}

@article{krige1951statistical,
  title={A statistical approach to some basic mine valuation problems on the Witwatersrand},
  author={Krige, Daniel G},
  journal={Journal of the Southern African Institute of Mining and Metallurgy},
  volume={52},
  number={6},
  pages={119--139},
  year={1951},
  publisher={Southern African Institute of Mining and Metallurgy}
}
-->

- **Chilès & Delfiner (2012)**:
  J.P. Chilès and P. Delfiner,
  "Geostatistics: Modeling Spatial Uncertainty", 2nd edition,
  Wiley, 2012.
- **Wackernagel (2013)**:
  H. Wackernagel, "Multivariate Geostatistics: An Introduction with Applications", Springer, 2013.
- **Krige (1951)**:
  D. G. Krige, "A statistical approach to some basic mine valuation problems on the Witwatersrand",
  1951.

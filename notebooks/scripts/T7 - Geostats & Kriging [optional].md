---
jupyter:
  jupytext:
    cell_metadata_filter: -all
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
from resources import show_answer, interact
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rnd
import scipy.linalg as sla
from mpl_tools.misc import nRowCol
from mpl_tools.place import freshfig
plt.ion();
```

# T7 - Spatial statistics ("geostatistics") & Kriging

Covariances between two (or a few) variables is very well,
but if you have not seen it before, the connection between covariances
and geophysical (spatial) fields may not be obvious.
The purpose of this tutorial is to familiarise you with random (spatial) fields
and their estimation.
$
\newcommand{\mat}[1]{{\mathbf{{#1}}}}
\newcommand{\vect}[1]{{\mathbf{#1}}}
\newcommand{\Expect}[0]{\mathbb{E}}
\newcommand{\NormDist}{\mathscr{N}}
$

Set some parameters

```python
rnd.seed(3000)
grid1D = np.linspace(0, 1, 21)
N = 15  # ensemble size
```

## Variograms

Denote *physical* location (i.e. a coordinate for 1D/2D/3D space) by $s$.
A ***random field***, $Z(s)$ is a function taking random values at each point.
Random fields are commonly assumed ***stationary*** (to some degree),
meaning that the dependence of $Z(s_{1})$ and $Z(s_{2})$,
for any two locations $s_1, s_2$, can be described in terms of the distance separating them, $d = \| s_{1} - s_{2} \|$.
For now, we will assume that the mean, $\Expect Z(s)$ is known and constant,
leaving the covariance as the most important descriptor.
But since the field is stationary, the covariance depends only on the distance,
so we can describe the full covariance of the field solely in terms of
"(auto-)***covariance function***", $C(d) = \mathbb{Cov}[Z(s_{1}), Z(s_{2})]$.
<details style="border: 1px solid #aaaaaa; border-radius: 4px; padding: 0.5em 0.5em 0;">
<summary style="font-weight: normal; font-style: italic; margin: -0.5em -0.5em 0; padding: 0.5em;">
In practice, geostatistics usually works with a reformulation of $C(d)$ called "(semi-)<strong>variogram"</strong>, $\gamma(d) = C(0) - C(d)$
... (optional reading 🔍)
</summary>
The variogram can be defined more generally
(allowing for infinite-variance processes, and requiring stationarity of increments, not second-order stationarity)
as half the variance of $Z(s_{1}) - Z(s_{2})$.

- - -
</details>

```python
def variogram(dists, Range=1, kind="Gauss", nugget=0):
    """Compute variogram for distance points `dists`."""
    dists = dists / Range
    if kind == "Spheric":
        gamma = 1.5 * dists - .5 * dists**3
        gamma[dists >= 1] = 1
    elif kind == "Expo":
        dists *= 3  # by convention
        gamma = 1 - np.exp(-dists)
    else:  # "Gauss"
        dists *= 3  # by convention
        gamma = 1 - np.exp(-(dists)**2)
    # Include nugget (discontinuity at 0)
    gamma *= (1-nugget)
    gamma[dists != 0] += nugget
    return gamma
```

#### Plot

```python
@interact(Range=(.01, 4), nugget=(0.0, 1, .1))
def plot_variogram(Range=1, nugget=0):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    for i, kind in enumerate(["Spheric", "Expo", "Gauss"]):
        gamma = variogram(grid1D, Range, kind, nugget=nugget)
        ax.plot(grid1D, gamma, lw=2, color=f"C{i}", label=kind)
        ax.legend(loc="upper left")
    plt.show()
```

In order to apply the variogram, we must first compute distances.
The following is a fairly efficient implementation.

```python
def dist_euclid(A, B):
    """Compute the l2-norm between each point (row) of A and B"""
    diff = A[:, None, :] - B
    d2 = np.sum(diff**2, axis=-1)
    return np.sqrt(d2)
```

Now the full covariance (matrix) between any sets of points can be defined by the following.

```python
def covar(coords, **vg_params):
    dists = dist_euclid(coords, coords)
    return 1 - variogram(dists, **vg_params)
```

```python
fig, ax = freshfig("1D covar")
C = covar(grid1D[:, None], Range=1, kind="Gauss", nugget=1e-3)
ax.matshow(C, cmap="PuOr");
```

## Random fields (1D)

Gaussian random variables (vectors) are fully specified by their mean and covariance.
Once in possession of a covariance matrix, we can use it to sample random variables
by multiplying its Cholesky factor (square root) onto standard normal variables.

```python
def sample_gaussian_fields(coords, **vg_params):
    """Gen. random (Gaussian) fields at `coords` (no structure/ordering required)."""
    C = covar(coords, **vg_params)
    L = sla.cholesky(C)
    fields = L.T @ rnd.randn(len(L.T), N)
    return fields
```

#### Exc

Use the plotting functionality below to
explain the effect of `Range` and `nugget`

```python
fig, ax = freshfig("1D random fields")
fields = sample_gaussian_fields(grid1D[:, None], Range=1, kind="Gauss", nugget=1e-3)
ax.plot(grid1D, fields, lw=2);
```

## Random fields (2D)

The following sets up a 2d grid.

```python
grid2x, grid2y = np.meshgrid(grid1D, grid1D)
grid2x.shape
```

where `grid2y` has the same shape. However, in the following we will "flatten" (a.k.a."(un)ravel", "vectorize", or "string out") this explicitly 2D grid of nodes into a simple list of points in 2D.

```python
grid2D = np.dstack([grid2x, grid2y]).reshape((-1, 2))
grid2D.shape
```

Importantly, none of the following methods actually assume any structure to the list. So we could also work with a completely irregularly spaced set of points. For example, `sample_gaussian_fields` is immediately applicable also to this 2D case.

```python
vg_params = dict(Range=1, kind="Gauss", nugget=1e-4)
fields = sample_gaussian_fields(grid2D, **vg_params)
```

Of course, for plotting purposes, we undo the flattening.

```python
def contour_plot(ax, field, cmap="nipy_spectral", levels=12, has_obs=True):
    field = field.reshape(grid2x.shape)  # undo flattening
    if has_obs:
        ax.plot(*obs_coo.T, "ko", ms=4)
        ax.plot(*obs_coo.T, "yo", ms=1)
    ax.set(aspect="equal", xticks=[0, 1], yticks=[0, 1])
    return ax.contourf(field, levels=levels, extent=(0, 1, 0, 1),
                       cmap=cmap, vmin=vmin, vmax=vmax)

# Fix the color scale for all subsequent `contour_plot`.
# Use `None` to re-compute the color scale for each subplot.
vmin = fields.min()
vmax = fields.max()
```

```python
fig, axs = freshfig(num="2D random fields", figsize=(5, 4),
                    nrows=3, ncols=4, sharex=True, sharey=True)

for ax, field in zip(axs.ravel(), fields.T):
    contour_plot(ax, field, has_obs=False)
```

It might be interesting to inspect the covariance matrix in this 2D case.

```python
C = covar(grid2D, **vg_params)
fig, ax = freshfig("2D covar")
ax.matshow(C, cmap="RdBu", vmin=0, vmax=1);
ax.grid(False)
```

## Estimation problem

For our estimation target we will use one of the above generated random fields.

```python
truth = fields.T[0]
```

For the observations, we pick some random grid locations for simplicity
(even though the methods work also with observations not on grid nodes).

```python
nObs = 10
obs_idx = rnd.randint(0, len(grid2D), nObs)
obs_coo = grid2D[obs_idx]
observations = truth[obs_idx]
```

## Spatial interpolant methods

```python
# Pre-compute re-used objects
dists_yy = dist_euclid(obs_coo, obs_coo)
dists_xy = dist_euclid(grid2D, obs_coo)
```

```python
estims = dict(Truth=truth)
vmin=truth.min()
vmax=truth.max()
```

The cells below contain snippets of different spatial interpolation methods,
followed by a cell that plots the interpolants.
Complete the code snippets.

#### Exc: Nearest neighbour interpolation

Implement the method [(wikipedia)](https://en.wikipedia.org/wiki/Nearest-neighbor_interpolation).  

```python
nearest_obs = np.zeros_like(truth, dtype=int)  ### FIX THIS ###
estims["Nearest-n."] = observations[nearest_obs]
```

```python
# show_answer('nearest neighbour interp')
```

#### Exc: Inverse distance weighting

Implement the method [(wikipedia)](https://en.wikipedia.org/wiki/Inverse_distance_weighting).  
*Hint: You can ignore the `errstate` line below. It is just used to "silence warnings" resulting from division by 0 (whose special case is treated in a cell further down).*

```python
exponent = 3
with np.errstate(invalid='ignore', divide='ignore'):
    weights = np.zeros_like(dists_xy)  ### FIX THIS ###
```

```python
# show_answer('inv-dist weight interp')
```

```python
# Apply weights
estims["Inv-dist."] = weights @ observations
```

```python
# Fix singularities
estims["Inv-dist."][obs_idx] = observations
```

#### Exc: Simple Kriging

Implement the method [(wikipedia)](https://en.wikipedia.org/wiki/Kriging#Simple_kriging).  

*Hint: use `sla.solve` or `sla.inv` (less recommended)*

```python
### ANSWER HERE ###
covar_yy = ...
cross_xy = ...
regression_coefficients = weights ### FIX THIS ### – should be cross_xy / covar_yy
```

```python
# show_answer('Kriging code')
```

```python
estims["Kriging"] = regression_coefficients @ observations
```

### Plot truth, estimates, error

```python
fig, axs = freshfig(num="Estimation problem", figsize=(8, 4), squeeze=False,
                    nrows=2, ncols=len(estims), sharex=True, sharey=True)

for name, ax1, ax2 in zip(estims, *axs):
    ax1.set_title(name)
    c1 = contour_plot(ax1, estims[name])
    c2 = contour_plot(ax2, estims[name] - truth, cmap="RdBu")
fig.tight_layout()
fig.subplots_adjust(right=0.85)
cbar = fig.colorbar(c1, cax=fig.add_axes([0.9, 0.15, 0.03, 0.7]))
axs[1, 0].set_ylabel("Errors");
```

#### Exc: Try different values of `Range`

- Run code to re-compute Kriging estimate.
- What does setting it to `0.1` cause? What about `100`?

```python
@interact(Range=(.01, 40))
def plot_krieged(Range=1):
    vg_params['Range'] = Range
    covar_yy = 1 - variogram(dists_yy, **vg_params)
    cross_xy = 1 - variogram(dists_xy, **vg_params)
    regression_coefficients = sla.solve(covar_yy, cross_xy.T).T

    fig, ax = freshfig(num="Kriging estimates")
    c1 = contour_plot(ax, regression_coefficients @ observations)
    fig.colorbar(c1);
    plt.show()
```

#### Generalizations

- Unknown mean (Ordinary Kriging)
- Co-Kriging (vector-valued fields)
- Trend surfaces (non-stationarity assumptions)

## Summary

The covariances of random fields can sometimes be described by the autocorrelation function,
or equivalently, the (semi-)variogram.
Covariances form the basis of a family of (geo-)spatial interpolation and approximation
methods known as Kriging, which can also be called/interpreted as
**Radial basis function (RBF) interpolation**,
**Gaussian process regression** (GP) regression.

- Kriging is derived by minimizing the variance of linear and unbiased estimators.
- RBF interpolation is derived by the explicit desire to fit
  N functions to N data points (observations).
- GP regression is derived by conditioning (applying Bayes rule)
  to the (supposedly) Gaussian distribution of the random field.

### Next: [T8 - Monte-Carlo & ensembles](T8%20-%20Monte-Carlo%20%26%20ensembles.ipynb)

<a name="References"></a>

### References

<!--

@book{chiles2009geostatistics,
  title={Geostatistics: Modeling Spatial Uncertainty},
  author={Chil{\`e}s, J.P. and Delfiner, P.},
  isbn={9780470317839},
  series={Wiley Series in Probability and Statistics},
  url={https://books.google.no/books?id=tZl07WdjYHgC},
  year={2009},
  publisher={Wiley}
}

@book{wackernagel2013multivariate,
  title={Multivariate Geostatistics: An Introduction with Applications},
  author={Wackernagel, H.},
  isbn={9783662052945},
  year={2013},
  publisher={Springer Berlin Heidelberg}
}

-->

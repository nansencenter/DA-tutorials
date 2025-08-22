# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,scripts//py:light,scripts//md
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
import matplotlib.pyplot as plt
import numpy.random as rnd
import scipy.linalg as sla
from mpl_tools.misc import nRowCol
from mpl_tools.place import freshfig
plt.ion();

# # T6 - Spatial statistics ("geostatistics") & Kriging
#
# Covariances between two (or a few) variables is very well,
# but if you have not seen it before, the connection between covariances
# and geophysical (spatial) fields may not be obvious.
# The purpose of this tutorial is to familiarise you with random (spatial) fields
# and their estimation.
# $
# % ######################################## Loading TeX (MathJax)... Please wait ########################################
# \newcommand{\Reals}{\mathbb{R}} \newcommand{\Expect}[0]{\mathbb{E}} \newcommand{\NormDist}{\mathscr{N}} \newcommand{\DynMod}[0]{\mathscr{M}} \newcommand{\ObsMod}[0]{\mathscr{H}} \newcommand{\mat}[1]{{\mathbf{{#1}}}} \newcommand{\bvec}[1]{{\mathbf{#1}}} \newcommand{\trsign}{{\mathsf{T}}} \newcommand{\tr}{^{\trsign}} \newcommand{\ceq}[0]{\mathrel{≔}} \newcommand{\xDim}[0]{D} \newcommand{\supa}[0]{^\text{a}} \newcommand{\supf}[0]{^\text{f}} \newcommand{\I}[0]{\mat{I}} \newcommand{\K}[0]{\mat{K}} \newcommand{\bP}[0]{\mat{P}} \newcommand{\bH}[0]{\mat{H}} \newcommand{\bF}[0]{\mat{F}} \newcommand{\R}[0]{\mat{R}} \newcommand{\Q}[0]{\mat{Q}} \newcommand{\B}[0]{\mat{B}} \newcommand{\C}[0]{\mat{C}} \newcommand{\Ri}[0]{\R^{-1}} \newcommand{\Bi}[0]{\B^{-1}} \newcommand{\X}[0]{\mat{X}} \newcommand{\A}[0]{\mat{A}} \newcommand{\Y}[0]{\mat{Y}} \newcommand{\E}[0]{\mat{E}} \newcommand{\U}[0]{\mat{U}} \newcommand{\V}[0]{\mat{V}} \newcommand{\x}[0]{\bvec{x}} \newcommand{\y}[0]{\bvec{y}} \newcommand{\z}[0]{\bvec{z}} \newcommand{\q}[0]{\bvec{q}} \newcommand{\br}[0]{\bvec{r}} \newcommand{\bb}[0]{\bvec{b}} \newcommand{\bx}[0]{\bvec{\bar{x}}} \newcommand{\by}[0]{\bvec{\bar{y}}} \newcommand{\barB}[0]{\mat{\bar{B}}} \newcommand{\barP}[0]{\mat{\bar{P}}} \newcommand{\barC}[0]{\mat{\bar{C}}} \newcommand{\barK}[0]{\mat{\bar{K}}} \newcommand{\D}[0]{\mat{D}} \newcommand{\Dobs}[0]{\mat{D}_{\text{obs}}} \newcommand{\Dmod}[0]{\mat{D}_{\text{obs}}} \newcommand{\ones}[0]{\bvec{1}} \newcommand{\AN}[0]{\big( \I_N - \ones \ones\tr / N \big)}
# $

# Set some parameters

rnd.seed(3000)
grid1D = np.linspace(0, 1, 21)
N = 15  # ensemble size


# ## Variograms
# The "Variogram" of a field is essentially `1 - autocovariance`. Thus, it describes the spatial dependence of the field. The mean (1st moment) of a field is usually estimated and described/parametrized with trend lines/surfaces, while higher moments are usually not worth modelling.

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


# #### Plot

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


# In order to apply the variogram, we must first compute distances.
# The following is a fairly efficient implementation.

def dist_euclid(A, B):
    """Compute the l2-norm between each point (row) of A and B"""
    diff = A[:, None, :] - B
    d2 = np.sum(diff**2, axis=-1)
    return np.sqrt(d2)


# Now the full covariance (matrix) between any sets of points can be defined by the following.

def covar(coords, **vg_params):
    dists = dist_euclid(coords, coords)
    return 1 - variogram(dists, **vg_params)


fig, ax = freshfig("1D covar")
C = covar(grid1D[:, None], Range=1, kind="Gauss", nugget=1e-3)
ax.matshow(C, cmap="RdBu");


# ## Random fields (1D)

# Gaussian random variables (vectors) are fully specified by their mean and covariance.
# Once in possession of a covariance matrix, we can use it to sample random variables
# by multiplying its Cholesky factor (square root) onto standard normal variables.

def gaussian_fields(coords, **vg_params):
    """Gen. random (Gaussian) fields at `coords` (no structure/ordering required)."""
    C = covar(coords, **vg_params)
    L = sla.cholesky(C)
    fields = L.T @ rnd.randn(len(L.T), N)
    return fields


# #### Exc
# Use the plotting functionality below to
# explain the effect of `Range` and `nugget`
#

fig, ax = freshfig("1D random fields")
fields = gaussian_fields(grid1D[:, None], Range=1, kind="Gauss", nugget=1e-3)
ax.plot(grid1D, fields, lw=2);

# ## Random fields (2D)
# The following sets up a 2d grid.

grid2x, grid2y = np.meshgrid(grid1D, grid1D)
grid2x.shape

# where `grid2y` has the same shape. However, in the following we will "flatten" (a.k.a."(un)ravel", "vectorize", or "string out") this explicitly 2D grid of nodes into a simple list of points in 2D.

grid2D = np.dstack([grid2x, grid2y]).reshape((-1, 2))
grid2D.shape

# Importantly, none of the following methods actually assume any structure to the list. So we could also work with a completely irregularly spaced set of points. For example, `gaussian_fields` is immediately applicable also to this 2D case.

vg_params = dict(Range=1, kind="Gauss", nugget=1e-4)
fields = gaussian_fields(grid2D, **vg_params)


# Of course, for plotting purposes, we undo the flattening.

# +
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

# +
fig, axs = freshfig(num="2D random fields", figsize=(5, 4),
                    nrows=3, ncols=4, sharex=True, sharey=True)

for ax, field in zip(axs.ravel(), fields.T):
    contour_plot(ax, field, has_obs=False)
# -

# It might be interesting to inspect the covariance matrix in this 2D case.

C = covar(grid2D, **vg_params)
fig, ax = freshfig("2D covar")
ax.matshow(C, cmap="RdBu", vmin=0, vmax=1);
ax.grid(False)

# ## Estimation problem

# For our estimation target we will use one of the above generated random fields.

truth = fields.T[0]

# For the observations, we pick some random grid locations for simplicity
# (even though the methods work also with observations not on grid nodes).

nObs = 10
obs_idx = rnd.randint(0, len(grid2D), nObs)
obs_coo = grid2D[obs_idx]
observations = truth[obs_idx]

# ## Spatial interpolant methods

# Pre-compute re-used objects
dists_yy = dist_euclid(obs_coo, obs_coo)
dists_xy = dist_euclid(grid2D, obs_coo)

estims = dict(Truth=truth)
vmin=truth.min()
vmax=truth.max()


# The cells below contain snippets of different spatial interpolation methods,
# followed by a cell that plots the interpolants.
# Complete the code snippets.

# #### Exc: Nearest neighbour interpolation
# Implement the method [(wikipedia)](https://en.wikipedia.org/wiki/Nearest-neighbor_interpolation).  

nearest_obs = np.zeros_like(truth, dtype=int)  ### FIX THIS ###
estims["Nearest-n."] = observations[nearest_obs]

# +
# show_answer('nearest neighbour interp')
# -

# #### Exc: Inverse distance weighting
# Implement the method [(wikipedia)](https://en.wikipedia.org/wiki/Inverse_distance_weighting).  
# *Hint: You can ignore the `errstate` line below. It is just used to "silence warnings" resulting from division by 0 (whose special case is treated in a cell further down).*

exponent = 3
with np.errstate(invalid='ignore', divide='ignore'):
    weights = np.zeros_like(dists_xy)  ### FIX THIS ###

# +
# show_answer('inv-dist weight interp')
# -

# Apply weights
estims["Inv-dist."] = weights @ observations

# Fix singularities
estims["Inv-dist."][obs_idx] = observations


# #### Exc: Simple Kriging
# Implement the method [(wikipedia)](https://en.wikipedia.org/wiki/Kriging#Simple_kriging).  
#
# *Hint: use `sla.solve` or `sla.inv` (less recommended)*

### ANSWER HERE ###
covar_yy = ...
cross_xy = ...
regression_coefficients = weights ### FIX THIS ### -- should be cross_xy / covar_yy

# +
# show_answer('Kriging code')
# -

estims["Kriging"] = regression_coefficients @ observations


# ### Plot truth, estimates, error

# +
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


# -

# #### Exc: Try different values of `Range`.
# - Run code to re-compute Kriging estimate.
# - What does setting it to `0.1` cause? What about `100`?

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

# #### Generalizations
#
# - Unknown mean (Ordinary Kriging)
# - Co-Kriging (vector-valued fields)
# - Trend surfaces (non-stationarity assumptions)
#

# ## Summary
# The covariances of random fields can sometimes be described by the autocorrelation function,
# or equivalently, the (semi-)variogram.
# Covariances form the basis of a family of (geo-)spatial interpolation and approximation
# methods known as Kriging, which can also be called/interpreted as
# **Radial basis function (RBF) interpolation**,
# **Gaussian process regression** (GP) regression.
#
# - Kriging is derived by minimizing the variance of linear and unbiased estimators.
# - RBF interpolation is derived by the explicit desire to fit
#   N functions to N data points (observations).
# - GP regression is derived by conditioning (applying Bayes rule)
#   to the (supposedly) Gaussian distribution of the random field.
#
# ### Next: [T7 - Chaos & Lorenz](T7%20-%20Chaos%20%26%20Lorenz%20(optional).ipynb)
#
# <a name="References"></a>
#
# ### References

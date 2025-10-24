"""Compare natural (min-bend/curve) i.e. cubic splines with Kriging (simple and ordinary) in 1D."""

import numpy as np
import scipy.linalg as sla
import numpy.random as rnd
from scipy.interpolate import CubicSpline, PchipInterpolator, Akima1DInterpolator
from matplotlib import pyplot as plt
import sys; sys.path.append("notebooks")
from resources import interact
plt.ion()
# plt.style.use("seaborn-v0_8")
plt.style.use('fivethirtyeight')

# import warnings
# np.seterr(all='raise')
# warnings.filterwarnings('error', category=RuntimeWarning)

vg_models = {
    "expo":      lambda d: 1 - np.exp(-d),
    "Gauss":     lambda d: 1 - np.exp(-(d) ** 2),
    "Cauchy":    lambda d: 1 - 1 / (1 + d ** 2),
    "triangular":lambda d: d.clip(max=1),
    "linear":    lambda d: d, # NB: intrinsically stationary, but
    # does not produce valid covariances ⇒ requires ordinary Kriging
    "quadratic": lambda d: d**2, # NB: valid, but degenerate
    "cubic":     lambda d: d ** 3, # NB: Not a valid variogram, but
    # a generalized covariance func. ⇒ requires order-1 intrinsic/universal Kriging 
    # NB: Not to be confused with the "cubic" covariance function.
}

def variograms(model="Gauss", Range=1, nugget=0):
    """Create variogram (function) for the given parameters."""
    def vg(dists):
        by_convention = 3 if model in ["Expo", "Gauss"] else 1
        dists = dists / Range * by_convention
        gamma = vg_models[model](dists)
        gamma[dists != 0] = nugget + (1-nugget) * gamma[dists != 0]
        return gamma
    return vg

reg = 1e-3

def simple_kriging(vg, dists_xy, dists_yy, observations):
    covar_yy = 1 - vg(dists_yy)
    cross_xy = 1 - vg(dists_xy)
    # weights = sla.inv(covar_yy) @ cross_xy.T
    # weights = sla.solve(covar_yy, cross_xy.T)
    weights, *_ = sla.lstsq(covar_yy, cross_xy.T, cond=reg)
    return observations @ weights

def ordinary_kriging(vg, dists_xy, dists_yy, observations):
    n = len(dists_xy)
    d = len(dists_yy)
    A = np.ones((d+1, d+1))
    b = np.ones((d+1, n))

    A[-1, -1] = 0
    A[:-1, :-1] = vg(dists_yy)
    b[:-1] = vg(dists_xy).T

    # weights = sla.solve(A, b)
    weights, *_ = sla.lstsq(A, b, cond=reg)
    return observations @ weights[:-1]

def universal_kriging(vg, dists_xy, dists_yy, observations, regressors, regressands):
    n = len(dists_xy)
    d = len(dists_yy)

    A = np.zeros((d+2, d+2))
    b = np.zeros((d+2, n))

    A[:-2, :-2] = vg(dists_yy)
    A[-2:,:-2] = regressors
    A.T[-2:,:-2] = regressors

    b[:-2] = vg(dists_xy).T
    b[-2:] = regressands

    # weights = sla.solve(A, b)
    weights, *_ = sla.lstsq(A, b, cond=reg)
    return observations @ weights[:-2]

L = 4
nObs = 5
hObs = 0.5
knots = L/2 + hObs * np.linspace(-L/2, L/2, nObs)
dists_yy = np.abs(knots[:, None] - knots[None, :])
for model in list(vg_models):
    vg = variograms(model, Range=1, nugget=0)
    CC = 1 - vg(dists_yy)
    s, V = sla.eig(CC)
    print(model.center(40, "*"))
    # print("\n" + "V:", V, sep="\n")
    assert(np.isclose(s.imag, 0).all())
    print("\n" + "s:", s.real, sep="\n")

    # print("\nProjections")
    # idx_null = np.where(np.isclose(s.real, 0, atol=1e-1))[0]
    # proj_null = V[:, idx_null] @ V[:, idx_null].T
    # for pow in range(4):
    #     print(f"x^{pow}:", knots**pow @ proj_null)

# # Main

@interact(Range=(.01, 3), nugget=(0, 0.99, .01),
          model=list(vg_models),
          L=(1, 5, 0.1), nObs=(1, 100, 1), hObs=(0.01, 1),
          log10_reg = (-20, 0), right=True)
def plot_kriged(Range=1, nugget=0, model="Gauss", L=4, nObs=6, hObs=0.5, log10_reg=-3):
    grid = np.linspace(0, L, 101)
    knots = L/2 + hObs * np.linspace(-L/2, L/2, nObs)

    global reg
    reg = 10.**log10_reg

    def truth(x):
        return np.sin(x**2)

    mu = 0
    true = truth(grid)

    obs = truth(knots)

    # NOTE: Don't need `dist_euclid` in 1D
    dists_yy = np.abs(knots[:, None] - knots[None, :])
    dists_xy = np.abs(grid[:, None] - knots[None, :])

    plt.figure("tmp").clear()
    fig, ax = plt.subplots(num="tmp", figsize=(2, 4))
    ax.axhline(mu, c="k", lw=0.5)
    ax.plot(grid, true, "-k", label="true")
    ax.plot(knots, obs, 'ok', label='data', ms=12, zorder=9)

    if nObs>1:
        ax.plot(grid, CubicSpline(knots, obs, bc_type="natural")(grid), 'C4', label="Natural/Cubic spline")
        #ax.plot(grid, PchipInterpolator(knots, obs)(grid), 'C5', label="PCHIP spline")
        #ax.plot(grid, Akima1DInterpolator(knots, obs)(grid), 'C6', label="Akima spline")

    vg = variograms(model, Range=Range, nugget=nugget)

    interp = mu + simple_kriging(vg, dists_xy, dists_yy, obs-mu)
    ax.plot(grid, interp, '--C1', label="SK", lw=4)

    interp = ordinary_kriging(vg, dists_xy, dists_yy, obs) 
    ax.plot(grid, interp, '--C2', label="OK", lw=3)
    
    if nObs>1:
        regressors = [np.ones(nObs), knots]
        regressands = [np.ones(len(grid)), grid]
        interp = universal_kriging(vg, dists_xy, dists_yy, obs, regressors, regressands)
        ax.plot(grid, interp, '--C3', label="IK", lw=2)

    ax.legend(loc='lower left')
    ax.set_ylim(-2, 2)

# # End

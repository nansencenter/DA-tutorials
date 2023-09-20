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

from resources import show_answer, interact, frame
# %matplotlib inline
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
plt.ion();

# # T7 - Chaos & Lorenz
# ***Chaos***
# is also known as the butterfly effect: "a butterfly that flaps its wings in Brazil can 'cause' a hurricane in Texas".
# As opposed to the opinions of Descartes/Newton/Laplace, chaos effectively means that even in a deterministic (non-stochastic) universe, we can only predict "so far" into the future. This will be illustrated below using two toy-model dynamical systems made by ***Edward Lorenz***.
# $
# % ######################################## Loading TeX (MathJax)... Please wait ########################################
# \newcommand{\Reals}{\mathbb{R}} \newcommand{\Expect}[0]{\mathbb{E}} \newcommand{\NormDist}{\mathcal{N}} \newcommand{\DynMod}[0]{\mathscr{M}} \newcommand{\ObsMod}[0]{\mathscr{H}} \newcommand{\mat}[1]{{\mathbf{{#1}}}} \newcommand{\bvec}[1]{{\mathbf{#1}}} \newcommand{\trsign}{{\mathsf{T}}} \newcommand{\tr}{^{\trsign}} \newcommand{\ceq}[0]{\mathrel{≔}} \newcommand{\xDim}[0]{D} \newcommand{\supa}[0]{^\text{a}} \newcommand{\supf}[0]{^\text{f}} \newcommand{\I}[0]{\mat{I}} \newcommand{\K}[0]{\mat{K}} \newcommand{\bP}[0]{\mat{P}} \newcommand{\bH}[0]{\mat{H}} \newcommand{\bF}[0]{\mat{F}} \newcommand{\R}[0]{\mat{R}} \newcommand{\Q}[0]{\mat{Q}} \newcommand{\B}[0]{\mat{B}} \newcommand{\C}[0]{\mat{C}} \newcommand{\Ri}[0]{\R^{-1}} \newcommand{\Bi}[0]{\B^{-1}} \newcommand{\X}[0]{\mat{X}} \newcommand{\A}[0]{\mat{A}} \newcommand{\Y}[0]{\mat{Y}} \newcommand{\E}[0]{\mat{E}} \newcommand{\U}[0]{\mat{U}} \newcommand{\V}[0]{\mat{V}} \newcommand{\x}[0]{\bvec{x}} \newcommand{\y}[0]{\bvec{y}} \newcommand{\z}[0]{\bvec{z}} \newcommand{\q}[0]{\bvec{q}} \newcommand{\br}[0]{\bvec{r}} \newcommand{\bb}[0]{\bvec{b}} \newcommand{\bx}[0]{\bvec{\bar{x}}} \newcommand{\by}[0]{\bvec{\bar{y}}} \newcommand{\barB}[0]{\mat{\bar{B}}} \newcommand{\barP}[0]{\mat{\bar{P}}} \newcommand{\barC}[0]{\mat{\bar{C}}} \newcommand{\barK}[0]{\mat{\bar{K}}} \newcommand{\D}[0]{\mat{D}} \newcommand{\Dobs}[0]{\mat{D}_{\text{obs}}} \newcommand{\Dmod}[0]{\mat{D}_{\text{obs}}} \newcommand{\ones}[0]{\bvec{1}} \newcommand{\AN}[0]{\big( \I_N - \ones \ones\tr / N \big)}
# $

# ## Dynamical systems
# Dynamical system are systems (sets of equations) whose variables evolve in time (the equations contains time derivatives). As a branch of mathematics, its theory is mainly concerned with understanding the *behaviour* of solutions (trajectories) of the systems.
#
# Below is a function to numerically **integrate**
# (i.e. step-wise evolve the system forward in time) a set of coupled ODEs.
# It relies on `scipy`, but adds some conveniences,
# notably taking advantage of Python's `**kwargs` (key-word argument) feature,
# to define an internal `dxdt` whose only two arguments are
# `x` for the current state, and `t` for time.

# +
from scipy.integrate import odeint
from dapper.mods.integration import rk4
dt = 0.01

def integrate(dxdt, initial_states, final_time, **params):
    # Output shape: `(len(initial_states), nTime, len(x))`
    dxdt_fixed = lambda x, t: dxdt(x, t, **params) # Fix params
    time_steps = np.linspace(0, final_time, 1+int(final_time / dt))
    integrated = []
    ### Replace the following (in the next exercise) ###
    for x0 in initial_states:
        trajectory = odeint(dxdt_fixed, x0, time_steps)
        integrated.append(trajectory)
    return np.array(integrated), time_steps


# -

# In addition, it takes care of looping over `initial_states`,
# computing a solution ("phase space trajectory") for each one,
# so that we can ask it to compute multiple trajectories at once,
# which we call Monte-Carlo simulation, or **ensemble forecasting**.
# But *loops are generally slow in Python*.
# Fortunately, for simple systems,
# we can write our code such that the dynamics get independently (but simultaneously) computed for rows of a *matrix* (rather than a single vector), meaning that each row in the input produces a corresponding row in the output. This in effect leaves `numpy` to do the looping (which it does much quicker than pure Python).
# Alternatively, since each simulation is completely independent of another realisation,
# they can "embarrasingly" easily be parallelized, which is a good option if the system is very costly to simulate.
# The exercise below challenges you to implement the first approach, resulting in much faster visualisation further below.
#
# #### Exc (optional) -- speed-up by vectorisation & parallelisation
# Replace `odeint` in the code above by `rk4` (which does not care about the size/shape of the input, thereby allowing for matrices, i.e. ensembles). Note that the call signature of `rk4` is similar to `odeint`, except that `time_steps` must be replaced by `t` and `dt`. I.e. it only computes a single time step, `t + dt`, so you must loop over `time_steps` yourself. *Hint: `dxdt(x, t, ...)` generally expect axis-0 (i.e. rows) of `x` to be the dimensions of the state vector -- not independent realisations of the states.*

# +
# show_answer('rk4')
# -

# ## The Lorenz (1963) attractor
#
# The [Lorenz-63 dynamical system](https://en.wikipedia.org/wiki/Lorenz_system) can be derived as an extreme simplification of *Rayleigh-Bénard convection*: fluid circulation in a shallow layer of fluid uniformly heated (cooled) from below (above).
# This produces the following 3 *coupled, nonlinear* ordinary differential equations (ODE):
#
# $$
# \begin{aligned}
# \dot{x} & = \sigma(y-x) \\
# \dot{y} & = \rho x - y - xz \\
# \dot{z} & = -\beta z + xy
# \end{aligned}
# \tag{1}
# $$
#
# where the "dot" represents the time derivative, $\frac{d}{dt}$. The state vector is $\x = (x,y,z)$, and the parameters are typically set to $\sigma = 10, \beta=8/3, \rho=28$. The ODEs can be coded as follows (yes, Python supports Unicode, but it might be cumbersome to type out!)

def dxdt63(state, time, σ, β, ρ):
    x, y, z = state
    return np.asarray([σ * (y - x),
                       x * (ρ - z) - y,
                       x * y - β * z])


# The following illustrated the system.

store = ['placeholder']
@interact(        σ=(0.,200), β=(0.,5), ρ=(0.,50),            N=(1,100), ε=(0.01,10), Time=(0.,100), zoom=(.1, 4))
def plot_lorenz63(σ=10,       β=8/3,    ρ=28     , in3D=True, N=2,       ε=0.01,      Time=2.0,      zoom=1):
    rnd.seed(23)
    initial_states = [-6.1, 1.2, 32.5] + ε*rnd.randn(N, 3)
    trajectories, times = integrate(dxdt63, initial_states, Time, σ=σ, β=β, ρ=ρ)
    store[0] = trajectories
    if in3D:
        ax = plt.figure().add_subplot(111, projection='3d')
        for orbit in trajectories:
            line, = ax.plot(*(orbit.T), lw=1, alpha=.5)
            ax.scatter3D(*orbit[-1], s=40, color=line.get_color())
        ax.axis('off')
        frame(trajectories, ax, zoom)
    else:
        fig, axs = plt.subplots(3, sharex=True, figsize=(5, 4))
        for dim, ax, orbits in zip('xyz', axs, trajectories.T):
            start = int(10/dt/zoom)
            ax.plot(times[-start:], orbits[-start:], lw=1, alpha=.5)
            ax.set_ylabel(dim)
        ax.set_xlabel('Time')
    plt.show()


# #### Exc -- Bifurcation hunting
# Classic linear stability analysis involves setting eqn. (1) to zero and considering the eigenvalues (and vectors) of its Jacobian matrix. Here we will go about it mainly by visually inspecting the numerical results of simulations.
# Answer the following (to an approximate degree of precision) by graduallying increasing $\rho$.
# Leave the other model parameters at their defaults, but use `ε`, `N`, `Time` and `zoom` to your advantage.
# - (a) What is the only fixed point for $\rho = 0$?
# - (b) At what (larger) value of $\rho$ does this change?
#   What do you think happened to the original fixed point?
# - (c) At what (larger) value of $\rho$ do we see an oscillating (spiraling) motion?
#   What do you think this entails for the aforementioned eigenvalues?
# - (d) Describe the difference in character of the trajectories between $\rho=10$ and $\rho=20$.
# - (e) At what (larger) values of $\rho$ do we get chaos?
#   In other words, when do the trajectories no longer converge to fixed points (or limit cycles)?
# - (f) Also try $\rho=144$ (edit the code). What is the nature of the trajectories now?
# - (g) *Optional*: Use pen and paper to show that the fixed points of the Lorenz system (1) are
#   indeed the origin as well as the roots of $x^2=\beta z$ with $y=x$,
#   but that the latter two only exist for $\rho > 1$.
#
# In conclusion, while a dynamical system naturally depends on its paramater values (almost by definition), the way in which its behaviour/character depend on it could come as a surprise.

# +
# show_answer("Bifurcations63")
# -

# #### Exc -- Doubling time
# Re-run the animation cell to get default parameter values.
# Visually investigate the system's (i.e. the trajectories') **sensitivity to initial conditions** by moving `Time`, `N` and `ε`. What do you reckon is the "doubling time" of the perturbations? I.e. how long do you think it takes (on average) for two trajectories to grow twice as far apart as they started (alternatives: 0.03, 0.3, 3, 30)? What are the implications for any prediction/forecasting we might attempt?

# +
# show_answer('Guesstimate 63')
# -

# ### Averages
#
# The result actually depends on where in "phase space" the particles started. For example, predictability in the Lorenz system is much shorter when the state is near the center, where the trajectories diverge into the two wings of the butterfly. So to get a universal answer one must average these experiments for many different initial conditions.
# Alternatively, since the above system is [ergodic](https://en.wikipedia.org/wiki/Ergodic_theory#Ergodic_theorems), we could also average a single experiment over a very, very long time, obtaining the same statistics (assuming they have converged). Though not strictly implied, ergodicity is closely related to chaos. It means that
#
# - A trajectory/orbit never quite repeats (the orbit is aperiodic).
# - The tracks of the orbits are sufficiently "dense" that they define a manifold
#   (something that looks like a surface, such as the butterfly wings above,
#   and for which we can speak of properties like derivatives and fractal dimension).
# - Every part (of positive measure) of the manifold can be reached from any other.
# - There is a probability density for the manifold,
#   quantifying the relative amount of time (of an infinite amount)
#   that the system spends in that neighbourhood.
#
# Set `N` and `Time` in the above interactive animation to their upper bounds (might take long to run!).
# Execute the code cell below.
# Do you think the samples behind the histograms are drawn from the same distribution?
# In other words, is the Lorenz system ergodic?

@interact()
def histograms():
    fig, axs = plt.subplots(ncols=3, sharey=True, figsize=(9, 3))
    def hist(ax, sample, lbl):
        ax.hist(sample, density=1, bins=20, label=lbl, alpha=.5)

    trajectories63 = store[0]
    for i, (ax, lbl) in enumerate(zip(axs, "xyz")):
        hist(ax, trajectories63[:, -1, i],            "at final time")
        hist(ax, trajectories63[-1, ::int(.2/dt), i], "of final member")
        ax.set_title(f"Component {lbl}")
    plt.legend();


# The long-run distribution of a system may be called its **climatology**.
# A somewhat rudimentary weather forecasting initialisation (i.e. DA) technique,
# called **optimal interpolation**,
# consists in using the climatology as the prior (as opposed to yesterday's forecast)
# when applying Bayes' rule (in its [Gaussian guise](T3%20-%20Bayesian%20inference.ipynb#Gaussian-Gaussian-Bayes'-rule-(1D))) to the observations of the day.

# ## The Lorenz-96 model
#
# Lorenz-96 is a "spatially 1D" dynamical system of an astoundingly simple design that resemble atmospheric convection,
# including nonlinear terms and chaoticity.
# Each state variable $\x_i$ can be considered some atmospheric quantity at grid point at a fixed latitude of Earth.  The system
# is given by the coupled set of ODEs,
# $$
# \frac{d \x_i}{dt} = (\x_{i+1} − \x_{i-2}) \x_{i-1} − \x_i + F
# \,,
# \quad \quad i \in \{1,\ldots,\xDim\}
# \,,
# $$
# where the subscript indices apply periodically.
#
# This model is not derived from physics but has similar characteristics, such as
# <ul>
#     <li> there is external forcing, determined by a parameter $F$;</li>
#     <li> there is internal dissipation, emulated by the linear term;</li>
#     <li> there is energy-conserving advection, emulated by quadratic terms.</li>
# </ul>
#
# [Further description in the very readable original article](https://www.ecmwf.int/sites/default/files/elibrary/1995/75462-predictability-problem-partly-solved_0.pdf).

# **Exc (optional) -- Conservation of energy:** Show that the "total energy" $\sum_{i=1}^{\xDim} \x_i^2$ is preserved by the quadratic terms in the ODE.  
# *Hint: consider its time derivative.*

# +
# show_answer("Lorenz energy")
# -

# The model is animated below.

# +
def s(vector, n):
    return np.roll(vector, -n)

def dxdt96(x, time, Force):
    return (s(x, 1) - s(x, -2)) * s(x, -1) - x + Force

ylims = -10, 20
# -

store = ["placeholder"]
@interact(        xDim=(4,60,1), N=(1,30), Force=(0,15.), ε=(0.01,3,0.1), Time=(0.05,90,0.04))
def plot_lorenz96(xDim=40,       N=2,      Force=8,       ε=0.01,         Time=3):
    rnd.seed(23)
    initial_states = np.zeros((N, xDim))
    initial_states[:, 0] = ε*(10 + rnd.randn(N))
    trajectories, times = integrate(dxdt96, initial_states, Time, Force=Force)
    store[0] = trajectories

    plt.figure(figsize=(7, 4))
    plt.plot(np.arange(xDim), trajectories[:, -1].T)
    plt.ylim(-10, 20)
    plt.show()

# #### Exc -- Bifurcation hunting 96
# Investigate by moving the sliders (but keep `xDim=40`): Under which settings of the force `F`
#
# - Do the solutions tend to the steady state $\x_i = F$ for all $i$ ?
# - Are the solutions periodic?
# - Is the system chaotic (i.e., the solutions are extremely sensitive to initial conditions,
#   meaning that the predictability horizon is finite) ?
#
# *PS: another way to visualise spatially 1D systems (or cross-sections) over time is the [Hovmöller diagram](https://en.wikipedia.org/wiki/Hovm%C3%B6ller_diagram), here represented for 1 realisation of the simulations.*

@interact()
def Hovmoller():
    plt.contourf(store[0][0], cmap="viridis", vmin=ylims[0], vmax=ylims[1])
    plt.colorbar();
    plt.show()


# +
# show_answer('Bifurcations96', 'a')
# -

# #### Exc (optional) -- Doubling time
# Maximise `N` (for a large sample), minimise `ε` (to approach linear conditions) and set `Time=1` (a reasonable first guess). Compute a rough estimate of the doubling time in the cell below from the data in `store[0]`, which holds the trajectories, and has shape `(N, len(times))`.
# *Hint: The theory for these questions will be described in further detail in the following section.*

# +
# show_answer("doubling time")
# -

# ## The double pendulum

# The [double pendulum](https://en.wikipedia.org/wiki/Double_pendulum) is another classic example of a chaotic system.
# It is a little longer to implement, so we'll just load it from [DAPPER](https://github.com/nansencenter/DAPPER/blob/master/dapper/mods/DoublePendulum/__init__.py).
# Unlike the Lorenz systems, the divergence of its "$f$" flow field is 0,
# so it is conservative, and all of the trajectories preserve their initial energy
# (except for what friction our numerical integration causes).
# Therefore it does not strictly speaking posess an attractor
# nor is it ergodic (but some things might be said upon restriction to the set of initial conditions with equal energy levels?)

# +
from numpy import cos, sin, pi
from dapper.mods.DoublePendulum import L1, L2, x0, dxdt
def x012(x): return (0 , L1*sin(x[0]) , L1*sin(x[0]) + L2*sin(x[2]))
def y012(x): return (0, -L1*cos(x[0]), -L1*cos(x[0]) - L2*cos(x[2]))

x0 = [.9*pi, 0, 0, 0] # Angular pos1, vel1, pos2, vel2
initial_states = x0 + 0.01*np.random.randn(20, 4)
trajectories, times = integrate(lambda x, t: dxdt(x), initial_states, 10)

@interact(k=(0, len(times)-1, 4), N=(1, len(initial_states)))
def plot_pendulum2(k=1, N=2):
    fig, ax = plt.subplots()
    ax.set(xlim=(-2, 2), ylim=(-2, 2), aspect="equal")
    for x in trajectories[:N, k]:
        ax.plot(x012(x), y012(x), '-o')
    plt.show()
# -

# ## Error/perturbation dynamics

# **Exc (optional) -- Perturbation ODE:** Suppose $x(t)$ and $z(t)$ are "twins": they evolve according to the same law $f$:
# $$
# \begin{align}
# \frac{dx}{dt} &= f(x) \\
# \frac{dz}{dt} &= f(z) \,.
# \end{align}
# $$
#
# Define the "error": $\varepsilon(t) = x(t) - z(t)$.  
# Suppose $z(0)$ is close to $x(0)$.  
# Let $F = \frac{df}{dx}(x(t))$.  
#
# * (a) Show that the error evolves according to the ordinary differential equation (ODE)
# $$\frac{d \varepsilon}{dt} \approx F \varepsilon \,.$$

# +
# show_answer("error evolution")
# -

# * (b) Suppose $F$ is constant. Show that the error grows exponentially: $\varepsilon(t) = \varepsilon(0) e^{F t} $.

# +
# show_answer("anti-deriv")
# -

# * (c)
#    * (1) Suppose $F<0$.  
#      What happens to the error?  
#      What does this mean for predictability?
#    * (2) Now suppose $F>0$.  
#      Given that all observations are uncertain (i.e. $R_t>0$, if only ever so slightly),  
#      can we ever hope to estimate $x(t)$ with 0 uncertainty?

# +
# show_answer("predictability cases")
# -

# - (d) What is the doubling time of the error?

# +
# show_answer("doubling time, Lyapunov")
# -

# * (e) Consider the ODE derived above.  
# How might we change it in order to model (i.e. emulate) a saturation of the error at some level?  
# Can you solve this equation?

# +
# show_answer("saturation term")
# -

# * (f) Now suppose $z(t)$ evolves according to $\frac{dz}{dt} = g(z)$, with $g \neq f$.  
# What is now the differential equation governing the evolution of the error, $\varepsilon$?

# +
# show_answer("linear growth")
# -

# ## Summary
# Prediction (forecasting) with these systems is challenging because they are chaotic:
# small errors grow exponentially.
# Therefore there is a limit to how far into the future we can make predictions (skillfully).
# Therefore it is crucial to minimize the initial error as much as possible.
# This is a task of DA (filtering).
#
# Also see this [book on chaos and predictability](https://kuiper2000.github.io/chaos_and_predictability/intro.html).
#
# ### Next: [T8 - Monte-Carlo & ensembles](T8%20-%20Monte-Carlo%20%26%20ensembles.ipynb)

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

# # T1 - Introduction
#
# *Copyright (c) 2020, Patrick N. Raanes
# $
# % ######################################## Loading TeX (MathJax)... Please wait ########################################
# \newcommand{\Reals}{\mathbb{R}} \newcommand{\Expect}[0]{\mathbb{E}} \newcommand{\NormDist}{\mathscr{N}} \newcommand{\DynMod}[0]{\mathscr{M}} \newcommand{\ObsMod}[0]{\mathscr{H}} \newcommand{\mat}[1]{{\mathbf{{#1}}}} \newcommand{\bvec}[1]{{\mathbf{#1}}} \newcommand{\trsign}{{\mathsf{T}}} \newcommand{\tr}{^{\trsign}} \newcommand{\ceq}[0]{\mathrel{≔}} \newcommand{\xDim}[0]{D} \newcommand{\supa}[0]{^\text{a}} \newcommand{\supf}[0]{^\text{f}} \newcommand{\I}[0]{\mat{I}} \newcommand{\K}[0]{\mat{K}} \newcommand{\bP}[0]{\mat{P}} \newcommand{\bH}[0]{\mat{H}} \newcommand{\bF}[0]{\mat{F}} \newcommand{\R}[0]{\mat{R}} \newcommand{\Q}[0]{\mat{Q}} \newcommand{\B}[0]{\mat{B}} \newcommand{\C}[0]{\mat{C}} \newcommand{\Ri}[0]{\R^{-1}} \newcommand{\Bi}[0]{\B^{-1}} \newcommand{\X}[0]{\mat{X}} \newcommand{\A}[0]{\mat{A}} \newcommand{\Y}[0]{\mat{Y}} \newcommand{\E}[0]{\mat{E}} \newcommand{\U}[0]{\mat{U}} \newcommand{\V}[0]{\mat{V}} \newcommand{\x}[0]{\bvec{x}} \newcommand{\y}[0]{\bvec{y}} \newcommand{\z}[0]{\bvec{z}} \newcommand{\q}[0]{\bvec{q}} \newcommand{\br}[0]{\bvec{r}} \newcommand{\bb}[0]{\bvec{b}} \newcommand{\bx}[0]{\bvec{\bar{x}}} \newcommand{\by}[0]{\bvec{\bar{y}}} \newcommand{\barB}[0]{\mat{\bar{B}}} \newcommand{\barP}[0]{\mat{\bar{P}}} \newcommand{\barC}[0]{\mat{\bar{C}}} \newcommand{\barK}[0]{\mat{\bar{K}}} \newcommand{\D}[0]{\mat{D}} \newcommand{\Dobs}[0]{\mat{D}_{\text{obs}}} \newcommand{\Dmod}[0]{\mat{D}_{\text{obs}}} \newcommand{\ones}[0]{\bvec{1}} \newcommand{\AN}[0]{\big( \I_N - \ones \ones\tr / N \big)}
# $
#
# ### Jupyter
#
# The "document" you're currently reading is a *Jupyter notebook*.
# As you can see, it consists of a sequence of **cells**,
# which can be code (Python) or text (markdown).
# For example, try editing the cell below (double-click it)
# to insert your name, and running it.

name = "Batman"
print("Hello world! I'm " + name)
for i, c in enumerate(name):
    print(i, c)

# You will likely be more efficient if you know these **keyboard shortcuts**:
#
# | Navigate                      | Edit              | Exit           | Run                              | Run & go to next                  |
# |-------------------------------|-------------------|----------------|----------------------------------|-----------------------------------|
# | <kbd>↓</kbd> and <kbd>↑</kbd> | <kbd>Enter</kbd>  | <kbd>Esc</kbd> | <kbd>Ctrl</kbd>+<kbd>Enter</kbd> | <kbd>Shift</kbd>+<kbd>Enter</kbd> |
#
# Actually, a notebook connects to a background **session (kernel/runtime/interpreter)** of Python, and all of the code cells (in a given notebook) are connected, meaning that they share variables, functions, and classes. You can start afresh by clicking `restart` somewhere in the top menu bar. The **order** in which you run the cells matters, and from now on,
# <mark><font size="-1">
#     the 1st code cell in each tutorial will be the following, which <em>you must run before others</em>. But if you're on Windows, then you must first delete the line starting with `!wget` (which is only really needed when running on Google Colab).
# </font></mark>

remote = "https://raw.githubusercontent.com/nansencenter/DA-tutorials"
# !wget -qO- {remote}/master/notebooks/resources/colab_bootstrap.sh | bash -s
from resources import show_answer, envisat_video

# ### Python
#
# There is a huge amount of libraries available in **Python**, including the popular `scipy` and `matplotlib` packages, both with the essential `numpy` library at their core. They're usually abbreviated `sp`, `mpl` (and `plt`), and `np`. Try them out by running the following cell.

# +
import numpy as np
import matplotlib.pyplot as plt
plt.ion();

# Use numpy's arrays for vectors and matrices. Example constructions:
a = np.arange(10) # Alternatively: np.array([0,1,2,3,4,5,6,7,8,9])
I = 2*np.eye(10)  # Alternatively: np.diag(2*np.ones(10))

print("Indexing examples:")
print("a        =", a)
print("a[3]     =", a[3])
print("a[0:3]   =", a[0:3])
print("a[:3]    =", a[:3])
print("a[3:]    =", a[3:])
print("a[-1]    =", a[-1])
print("I[:3,:3] =", I[:3,:3], sep="\n")

print("\nLinear algebra examples:")
print("100+a =", 100+a)
print("I@a   =", I@a)
print("I*a   =", I*a, sep="\n")

plt.title("Plotting example")
plt.ylabel("i $x^2$")
for i in range(4):
    plt.plot(i * a**2, label="i = %d"%i)
plt.legend();
# -

# These tutorials require that you are able to understand the above code, but not much beyond that.
# Some exercises will ask you to do some programming, but understanding the pre-written code is also important.
# The interesting parts of the code can all be found in the notebooks themselves
# (as opposed to being hidden away via imports).
# Beware, however, that it is not generally production-ready.
# For example, it overuses global variables, and is lacking in vectorisation,
# generally for the benefit of terseness and simplicity.
#
# ### Dynamical and observational models
#
# What is a ***model***?
# In the broadest sense, a model is a *simplified representation* of something.
# A convenient language for this purpose is mathematics,
# in which case the model consists of a set of equations, often differential.
#
# We will mainly be concerned with models of **dynamical systems**, meaning *stuff that changes in time*.
# The "stuff", denoted $\x_k$ for time $k$, will be referred to as ***state*** variables/vectors.
# Regardless of sophistication or how many PhDs worked on coding it up as a computer simulation program,
# the ***dynamical model*** will henceforth be represented simply as the *function* $\DynMod_k$
# that **predicts** the state at time $k+1$ from $\x_k$.
#
# <details style="border: 1px solid #aaaaaa; border-radius: 4px; padding: 0.5em 0.5em 0;">
# <summary style="font-weight: normal; font-style: italic; margin: -0.5em -0.5em 0; padding: 0.5em;">
#   Examples of models include ... 🔍
# </summary>
#
# - (a) Laws of motion and gravity (Newton, Einstein)
# - (b) Epidemic (SEIR) and predator-prey (Lotka-Volterra)
# - (c) Weather/climate forecasting (Navier–Stokes + thermodynamics + ideal gas law + radiation + cloud microphysics + BCs)
# - (d) Petroleum reservoir flow (Multiphase Darcy's law + ...)
# - (e) Chemical and biological kinetics (Arrhenius, Michaelis-Menten, Mass Action Law)
# - (f) Traffic flow (Lighthill-Whitham-Richards)
# - (g) Sports rating (Elo, Glicko, TrueSkill)
# - (h) Financial pricing (Black-Scholes)
#
# ---
# </details>
#
# **Exc (optional) -- state variables:**  
#
# - For the above model examples above that you are familiar with, list the elements of the state variable.
#
# - Generally speaking, do you think
#   - the ordering of the elements/components matters?
#   - the state vector needs to be of fixed length, or can it change over time?
#   - it is problematic if we include more variables than needed?

# +
# show_answer('state variables')
# -

# Whether describing some fundamental laws of nature,
# or comprising of a set of empirical or statistical relations,
# the "goodness" of a model is usually assessed in terms of (some measure of) skill of prediction,
# as expressed by the following maxim.
#
# > All models are wrong, but some are useful -- [George E. P. Box](https://en.wikipedia.org/wiki/All_models_are_wrong)
#
# **Exc (optional) -- model error:**  
# For each of model examples above, list the shortcomings (below) that seem relevant.
#
# 1. Inaccurate at relatively high speeds
# 1. Extreme events do not conform to statistical assumptions
# 1. Assumes closed systems, ignoring external influences,
#    except for poorly specified boundary conditions and forcings
# 1. Assumes equilibrium or steady-state when systems are inherently dynamic
# 1. Lack of demographic and/or geographic resolution.
# 1. Continuity is an approximation
# 1. Oversimplification of complex interactions and feedbacks
# 1. Incompatibility with quantum dynamics
# 1. Insufficient spatial or temporal resolution upon discretization

# +
# show_answer('model error')
# -

# Taking the quoted advice to heart, in data assimilation it is common to assume that
# the difference between the true evolution and that suggested by the model alone is explained
# by a random (stochastic) noise term, $\q_k$, with a known distribution, i.e.
#
# $\begin{equation}
#   \x_{k+1} = \DynMod_k(\x_k) + \q_k \,. \tag{DynMod}
# \end{equation}$
#
# However, a good model (i.e. $\q \approx 0$) is not enough to ensure good predictions, because
#
# > Garbage in, garbage out (GIGO)
#
# In other words, we also need good initial conditions,
# i.e. a good estimate of $\x_k$.
# This is known as the ***forecast initialisation*** problem.
# At first this might seem obvious and trifling,
# but this illusion is quickly dispelled by considering the case of numerical weather prediction.
# Clearly, in order to launch the numerical simulator (model), $\DynMod_k$,
# to forecast (predict) *tomorrow*'s weather,
# we initially need to know *today*'s state of the atmosphere (wind, pressure, density and temperature)
# at each grid point in the model.
# Yet despite the quantitative explosion of data since the advent of weather satellites in the 1970s,
# most parts of the globe are (at any given moment) unobserved.
# Moreover, the ***measurement/observation data*** available to us, $\y_k$, are not generally a "direct observation"
# of quantities in the state vector, but rather some function, i.e. model $\ObsMod_{\!k}$ thereof
# (in the case of satellite radiances: an integral along the vertical column at some lat/long location,
# or even a more complicated radiative transfer model).
# Finally, since any measurement includes some amount of inaccuracy,
# we include an observation noise, $\varepsilon_k$, in our conception of the measuring process, i.e.
#
# $\begin{equation}
#   \y_k = \ObsMod_{\!k}(\x_k) + \varepsilon_k \,. \tag{ObsMod}
# \end{equation}$
#
# **Exc (optional) -- observation examples:**  
# For each of the above dynamical model examples, suggest 1 or more observation kinds.

# +
# show_answer('obs examples')
# -

# ### Data + Models = ❤️
#
# The above complications make the forecast initialisation problem daunting.
# Fortunately we have another source of information on $\x_k$: yesterday's (or the previous) forecast.
# As we will see, using it as a "prior" in the estimation of $\x_k$ will implicitly
# incorporate the information from *previous* observations, $\y_1, \ldots, \y_{k-1}$,
# on top of the "incoming" one, $\y_k$.
# Thus, model forecasts help in the estimation of $\x_k$, which is otherwise based on the incoming observation, $\y_k$,
# which in turn improve the forecast of $\x_{k+1}$, and so on in a virtuous cycle of improved estimation and prediction (❤️).
#
# **State estimation** (a.k.a. **sequential inference**)
# is the estimation of unknown/uncertain quantities of **dynamical systems**, $\{\x_k\}$,
# based on imprecise (noisy) data/observations, $\{\y_k\}$.
# State estimation is similar to time series estimation and signal processing,
# but focuses on the case where we have a good (skillful) predictive model of the dynamical system,
# and allows for multivariate, partially observed (hidden) states.
#
# The most famous state estimation technique is the ***Kalman filter (KF)***,
# which was developed to steer the Apollo mission rockets to the moon.
# For example, in guidance systems, the *state variable* (vector) consists of at least 6 elements: 3 for the current position and 3 for velocity, whose trajectories we wish to track in time. More sophisticated systems can also include acceleration and/or angular quantities. The *dynamical model* then consists of the fact that displacement is the time integral of the velocity, while the velocity is the integral of acceleration. The noisy *observations* can come from altimetry, sextants, speedometers, compass readings, accelerometers, gyroscopes, or fuel-gauges. The essential point is that we have an *observational model* predicting the observations from the state. For example, the altimeter model is simply the function that selects the $z$ coordinate from the state vector, while the force experienced by an accelerometer can be modelled by Newton's second law of motion, $F = m a$.
#
# <img align="right" width="400" src="./resources/DA_bridges.jpg" alt='DA "bridges" data and models.'/>
#
# In the context of *large* dynamical systems, especially in geoscience (climate, ocean, hydrology, petroleum)
# state estimation is known as **data assimilation** (DA),
# and is thought of as a "bridge" between data and models,
# as illustrated on the right (source: [AICS-Riken](https://aics.riken.jp/en))
# For example, in weather applications, the dynamical model is an atmospheric fluid-mechanical simulator, the state variable consists of the fields of pressure, humidity, and wind quantities discretized on a grid,
# and the observations may come from satellite or weather stations.
#
# But when it was first proposed to apply the KF to DA (specifically, weather forecasting),
# the idea was though ludicrous because of some severe technical challenges in DA (vs. "classic" state estimation):
#
# - size of data and models;
# - nonlinearity of models;
# - sparsity and inhomogeneous-ness of data.
#
# Some of these challenges may be recognized in the video below. Can you spot them?

envisat_video()

# ### The ensemble Kalman filter (EnKF)
#
# The EnKF is a Monte-Carlo formulation of the KF
# that manages (fairly well) to deal with the above challenges in DA.
#
# For those familiar with the method of 4D-Var, **further advantages of the EnKF** include it being:
#
# - Non-invasive: the models are treated as black boxes, and no explicit Jacobian is required.
# - Bayesian:
#   - provides ensemble of possible realities;
#     - arguably the most practical form of "uncertainty quantification";
#     - ideal way to initialize "ensemble forecasts";
#   - uses "flow-dependent" background covariances in the analysis.
# - Embarrassingly parallelizable:
#   - distributed across realizations for model forecasting;
#   - distributed across local domains for observation analysis.
#
# The rest of this tutorial provides an EnKF-centric presentation of DA.
#
# ### DAPPER example
#
# This tutorial builds on the underlying package, DAPPER, made for academic research in DA and its dissemination. For example, the code below is taken from  `DAPPER/example_1.py`. It illustrates DA on a small toy problem. At the end of these tutorials, you should be able to reproduce (from the ground up) this type of experiment.
#
# Run the cells in order and try to interpret the output.
# <mark><font size="-1">
# <em>Don't worry</em> if you can't understand what's going on -- we will discuss it later throughout the tutorials.
# </font></mark>

# +
import dapper as dpr
import dapper.da_methods as da

# Load experiment setup: the hidden Markov model (HMM)
from dapper.mods.Lorenz63.sakov2012 import HMM
HMM.tseq.T = 30  # shorten experiment

# Simulate synthetic truth (xx) and noisy obs (yy)
xx, yy = HMM.simulate()

# Specify a DA method configuration ("xp" is short for "experiment")
# xp = da.OptInterp()
# xp = da.Var3D()
# xp = da.ExtKF(infl=90)
xp = da.EnKF('Sqrt', N=10, infl=1.02, rot=True)
# xp = da.PartFilt(N=100, reg=2.4, NER=0.3)

# Assimilate yy, knowing the HMM; xx is used to assess the performance
xp.assimilate(HMM, xx, yy)

# #### Average the time series of various statistics
# print(xp.stats)  # ⇒ long printout
xp.stats.average_in_time()

print(xp.avrgs.tabulate(['rmse.a', 'rmv.a']))
# -

xp.stats.replay()

# Some more diagnostics
if False:
    import dapper.tools.viz as viz
    viz.plot_rank_histogram(xp.stats)
    viz.plot_err_components(xp.stats)
    viz.plot_hovmoller(xx)

# ### Vocabulary exercises
#
# **Exc -- Word association:**
# Fill in the `x`'s in the table to group the words with similar meaning.
#
# `Sample, Random, Measurements, Monte-Carlo, Observations, Set of draws`
#
# - Ensemble, x, x
# - Stochastic, x, x
# - Data, x, x

# +
# show_answer('thesaurus 1')
# -

# - "The answer" is given from the perspective of DA. Do you agree with it?
# - Can you describe the (important!) nuances between the similar words?
#
# **Exc (optional) -- Word association 2:**
# Also group these words:
#
# `Inverse problems, Operator, Sample point, Transform(ation), Knowledge, Relation, Probability, Mapping, Particle, Sequential, Inversion, Realization, Relative frequency, Information, Iterative, Estimate, Estimation, Single draw, Serial, Regression, Model, Fitting, Uncertainty`
#
# - Statistical inference, x, x, x, x, x
# - Ensemble member, x, x, x, x
# - Quantitative belief, x, x, x, x, x, x
# - Recursive, x, x, x
# - Function, x, x, x, x, x

# +
# show_answer('thesaurus 2')
# -

# **Exc (optional) -- intro discussion:** Prepare to discuss the following questions. Use any tool at your disposal.
#
# - (a) What is a "dynamical system"?
# - (b) What are "state variables"? How do they differ from parameters?
# - (c) What are "prognostic" variables? How do they differ from "diagnostic" variables?
# - (d) What is DA?
# - (e) Is DA a science, an engineering art, or a dark art?
# - (f) What is the point of "Hidden Markov Models"?

# +
# show_answer('Discussion topics 1')
# -

# ### Next: [T2 - Gaussian distribution](T2%20-%20Gaussian%20distribution.ipynb)

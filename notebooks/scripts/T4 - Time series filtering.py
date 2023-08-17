# ---
# jupyter:
#   jupytext:
#     formats: ipynb,scripts//py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

remote = "https://raw.githubusercontent.com/nansencenter/DA-tutorials"
# !wget -qO- {remote}/master/notebooks/resources/colab_bootstrap.sh | bash -s
import resources.workspace as ws

# %matplotlib inline
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
plt.ion();

# # T4 - Time series filtering
# Before we look at the full (multivariate) Kalman filter,
# let's get more familiar with time-dependent (temporal/sequential) problems.
# $
# % ######################################## Loading TeX (MathJax)... Please wait ########################################
# \newcommand{\Reals}{\mathbb{R}} \newcommand{\Expect}[0]{\mathbb{E}} \newcommand{\NormDist}{\mathcal{N}} \newcommand{\DynMod}[0]{\mathscr{M}} \newcommand{\ObsMod}[0]{\mathscr{H}} \newcommand{\mat}[1]{{\mathbf{{#1}}}} \newcommand{\bvec}[1]{{\mathbf{#1}}} \newcommand{\trsign}{{\mathsf{T}}} \newcommand{\tr}{^{\trsign}} \newcommand{\ceq}[0]{\mathrel{≔}} \newcommand{\xDim}[0]{D} \newcommand{\supa}[0]{^\text{a}} \newcommand{\supf}[0]{^\text{f}} \newcommand{\I}[0]{\mat{I}} \newcommand{\K}[0]{\mat{K}} \newcommand{\bP}[0]{\mat{P}} \newcommand{\bH}[0]{\mat{H}} \newcommand{\bF}[0]{\mat{F}} \newcommand{\R}[0]{\mat{R}} \newcommand{\Q}[0]{\mat{Q}} \newcommand{\B}[0]{\mat{B}} \newcommand{\C}[0]{\mat{C}} \newcommand{\Ri}[0]{\R^{-1}} \newcommand{\Bi}[0]{\B^{-1}} \newcommand{\X}[0]{\mat{X}} \newcommand{\A}[0]{\mat{A}} \newcommand{\Y}[0]{\mat{Y}} \newcommand{\E}[0]{\mat{E}} \newcommand{\U}[0]{\mat{U}} \newcommand{\V}[0]{\mat{V}} \newcommand{\x}[0]{\bvec{x}} \newcommand{\y}[0]{\bvec{y}} \newcommand{\z}[0]{\bvec{z}} \newcommand{\q}[0]{\bvec{q}} \newcommand{\br}[0]{\bvec{r}} \newcommand{\bb}[0]{\bvec{b}} \newcommand{\bx}[0]{\bvec{\bar{x}}} \newcommand{\by}[0]{\bvec{\bar{y}}} \newcommand{\barB}[0]{\mat{\bar{B}}} \newcommand{\barP}[0]{\mat{\bar{P}}} \newcommand{\barC}[0]{\mat{\bar{C}}} \newcommand{\barK}[0]{\mat{\bar{K}}} \newcommand{\D}[0]{\mat{D}} \newcommand{\Dobs}[0]{\mat{D}_{\text{obs}}} \newcommand{\Dmod}[0]{\mat{D}_{\text{obs}}} \newcommand{\ones}[0]{\bvec{1}} \newcommand{\AN}[0]{\big( \I_N - \ones \ones\tr / N \big)}
# $

# ## Example problem: AR(1).
# Consider the scalar, stochastic process $\{x_k\}$ generated by
# $$ x_{x+1} = \DynMod_k x_k + q_k \,, \tag{Dyn} $$
# for sequentially increasing time index $k$,
# where $q_k$ is white noise ($q_k$ independent of $q_l$ for $k \neq l$).
# For our present purposes, the dynamical "model",
# $\DynMod_k$ is just some number that we know.
# Merely to alleviate the burden of bookkeeping,
# we henceforth assume that it is constant in time.
# Then $\{x_k\}$ is a so-called order-1 auto-regressive process,
# i.e. [AR(1)](https://en.wikipedia.org/wiki/Autoregressive_model#Example:_An_AR(1)_process).
# Suppose we get observations, $\{y_k\}$, corrupted by noise, as in
# $$ y_k = \ObsMod_k x_k + r_k \,, \tag{Obs} $$
# where the noise, $r_k$, is again independent of everything.
# Moreover, for simplicity,
# assume that the measurement model, $\ObsMod$, is independent of $k$,
# and for each $k$, let
# - $q_k \sim \NormDist(0, Q)$,
# - $r_k \sim \NormDist(0, R)$.
#
# Also assume $x_0 \sim \NormDist(x\supa_0, P\supa_0)$.
# The code below simulates a random realisation of this process.

# +
# Use H=1 so that it makes sense to plot data on same axes as state.
H = 1

# Initial estimate
xa = 0   # mean
Pa = 10  # variance

def simulate(nTime, xa, Pa, M, H, Q, R):
    """Simulate synthetic truth (x) and observations (y)."""
    x = xa + np.sqrt(Pa)*rnd.randn()        # Draw initial condition
    truths = np.zeros(nTime)                # Allocate
    obsrvs = np.zeros(nTime)                # Allocate
    for k in range(nTime):                  # Loop in time
        x = M * x + np.sqrt(Q)*rnd.randn()  # Dynamics
        y = H * x + np.sqrt(R)*rnd.randn()  # Measurement
        truths[k] = x                       # Assign
        obsrvs[k] = y                       # Assign
    return truths, obsrvs


# -

# The following plots the process. *You don't need to read & understand it*,
# but if you find that there are not enough sliders to play around with,
# feel free to alter the code to suit your needs
# (for example, you can comment out the line plotting observations, or `cInterval`).
# *PS: Some of the sliders get activated later.*

@ws.interact(seed=(1, 12), M=(0, 1.03, .01), nTime=(0, 100),
             logR=(-9, 9), logR_bias=(-9, 9),
             logQ=(-9, 9), logQ_bias=(-9, 9))
def exprmt(seed=4, nTime=50, M=0.97, logR=1, logQ=1, analyses_only=False, logR_bias=0, logQ_bias=0):
    R, Q, Q_bias, R_bias = 4.0**np.array([logR, logQ, logQ_bias, logR_bias])

    rnd.seed(seed)
    truths, obsrvs = simulate(nTime, xa, Pa, M, H, Q, R)

    plt.figure(figsize=(9, 6))
    kk = 1 + np.arange(nTime)
    plt.plot(kk, truths, 'k' , label='True state ($x$)')
    plt.plot(kk, obsrvs, 'g*', label='Noisy obs ($y$)', ms=9)

    try:
        estimates, variances = KF(nTime, xa, Pa, M, H, Q*Q_bias, R*R_bias, obsrvs)
        if analyses_only:
            plt.plot(kk, estimates[:, 1], label='Kalman$\supa$ ± 1$\sigma$')
            plt.fill_between(kk, *ws.cInterval(estimates[:, 1], variances[:, 1]), alpha=.2)
        else:
            kk2 = kk.repeat(2)
            plt.plot(kk2, estimates.flatten(), label='Kalman ± 1$\sigma$')
            plt.fill_between(kk2, *ws.cInterval(estimates, variances), alpha=.2)
    except NameError:
        pass

    sigproc = {}
    ### INSERT ANSWER TO EXC "signal processing" HERE ###
    # sigproc['some method'] = ...
    for method, estimate in sigproc.items():
        plt.plot(kk[:len(estimate)], estimate, label=method)

    plt.xlabel('Time index (k)')
    plt.legend(loc='upper left')
    plt.axhline(0, c='k', lw=1, ls='--')
    plt.show()


# **Exc -- AR1 properties:** Answer the following.
# - What does `seed` control?
# - Explain what happens when `M=0`. Also consider $Q \rightarrow 0$.  
#   Can you give a name to this `truth` process,
#   i.e. a link to the relevant Wikipedia page?  
#   What about when `M=1`?  
#   Describe the general nature of the process as `M` changes from 0 to 1.  
#   What about when `M>1`?  
# - What happens when $R \rightarrow 0$ ?
# - What happens when $R \rightarrow \infty$ ?

# +
# ws.show_answer('AR1')
# -

# ## The (univariate) Kalman filter (KF)

# Now we have a random variable that evolves in time, that we can pretend is unknown,
# in order to estimate (or "track") it.
# More specifically, while $x_0$ is unknown,
# we do know that $x_0 \sim \NormDist(x\supa_0, P\supa_0)$, including the parameters.
# We also know that $x_k$ evolves according to eqn. (Dyn).
# Therefore, as shown in the following exercise,
# for $x_1$ we know that
# $x_1 \sim \NormDist(x\supf_1, P\supf_1)$, with
# $$\begin{align}
# x\supf_1 &= \DynMod \, x\supa_0 \tag{5} \\
# P\supf_1 &= \DynMod^2 \, P\supa_0 + Q \tag{6}
# \end{align}$$
#
# #### Exc -- linear algebra of Gaussian random variables
# - (a) Show the linearity of the expectation operator:
#       $\Expect [ \DynMod  x + b ] = \DynMod \Expect[x] + b$, for some constant $b$.
# - (b) Thereby, show that $\mathbb{Var}[ \DynMod  x + b ] = \DynMod^2 \mathbb{Var} [x]$.
# - (c) *Optional*: Now let $z = x + q$, with $x$ and $q$ independent and Guassian.
#       Then the pdf of this sum of random variables, $p_z(z)$, is given by convolution
#       (hopefully this makes intuitive sense, at least in the discrete case):
#       $$ p_z(z) = \int p_x(x) \, p_q(z - x) \, d x \,.$$
#       Show that $z$ is also Gaussian,
#       whose mean and variance are the sum of the means and variances (respectively).  
#       *Hint: you will need the result on [completing the square](T3%20-%20Bayesian%20inference.ipynb#Exc----GG-Bayes),
#       specifically the part that we did not make use of for Bayes' rule.  
#       If you get stuck, you can also view the excellent [3blue1brown](https://www.youtube.com/watch?v=d_qvLDhkg00&t=266s&ab_channel=3Blue1Brown) on the topic.*

# +
# ws.show_answer('Sum of Gaussians', 'a')
# -

# Formulae (5) and (6) are called the **forecast step** of the KF.
# But when $y_1$ becomes available, according to eqn. (Obs),
# then we can update/condition our estimate of $x_1$, i.e. compute the posterior,
# $p(x_1 | y_1) = \mathcal{N}(x_1 \mid x\supa_1, P\supa_1) \,,$
# using the formulae we developed for Bayes' rule with
# [Gaussian distributions](T3%20-%20Bayesian%20inference.ipynb#Gaussian-Gaussian-Bayes'-rule-(1D)).
# We call this the **analysis step** of the KF.
# Note that there are no approximations to these steps (with the linear-Gaussianassumptions we have made).
#
# If $k=1$ is "today", then we can say that "yesterday's forecast became today's prior".
# We can subsequently apply the same two steps again
# to produce forecast and analysis estimates for the next time index $k$.
# Thus, at any point in time, $k$, we compute the exact Bayesian pdf's for $x_k$:
# The analysis step "assimilates" $y_k$ to compute $p(x_k | y_{1:k})$,
# where $y_{1:k} = y_1, \ldots, y_k$,
# while the forecast computes $p(x_{k+1}| y_{1:k})$.
# It is important to appreciate two benefits to this **recursive** procedure
#
# - The recursiveness of the procedure reflects the recursiveness (Markov property) of nature:
#   Both in the problem and our solution, time $k+1$ *builds on* time $k$.
#   It means that we do not have to re-do the entire problem for each $k$.
# - At every time $k$ we only deal with functions of 1 or 2 variables: $x_k$ and $x_{k+1}$.
#   Even if we were doing $k$ computations, this is a significantly smaller domain
#   (in which to quanitify our densities or covariances) than that of the joint pdf $p(x_{1:k} | y_{1:k})$.
#   Ref. [curse of dimensionality](http://localhost:8888/notebooks/notebooks/T3%20-%20Bayesian%20inference.ipynb#Exc-(optional)----Curse-of-dimensionality,-part-1).
#
# Note, however, that our recursive procedure, called ***filtering***, does *not* compute $p(x_l | y_{1:k})$ for any $l<k$. In other words, any filtering estimate only contains *past* information. Updating estimates of the state at any previous time(s) is called ***smoothing***.  However, for the purposes of prediction/forecasting, filtering is all we need: accurate initial conditions (estimates of the present moment).

# #### Exc -- Implementation
# Below is a very rudimentary sequential estimator (not the KF!), essentially just doing "persistance" forecasts, and setting the analysis estimates to the value of the observations (*which is only generally a possibility in this linear, scalar case*). Run its cell to define it, and then re-run the above interactive animation cell. Then
# - Implement the KF properly by replace the forecast and analysis steps below. *Re-run the cell.*
# - Try implementing the analysis step both in the "precision" and "gain" forms.

def KF(nTime, xa, Pa, M, H, Q, R, obsrvs):
    """Kalman filter. PS: (xa, Pa) should be input with *initial* values."""
    ############################
    # TEMPORARY IMPLEMENTATION #
    ############################
    estimates = np.zeros((nTime, 2))
    variances = np.zeros((nTime, 2))
    for k in range(nTime):
        # Forecast step
        xf = xa
        Pf = Pa
        # Analysis update step
        Pa = R / H**2
        xa = obsrvs[k] / H
        # Assign
        estimates[k] = xf, xa
        variances[k] = Pf, Pa
    return estimates, variances


# +
# ws.show_answer('KF1 code')
# -

# #### Exc -- KF behaviour
# - Set `logQ` to its minimum, and `M=1`.  
#   We established in Exc "AR1" that the true states are now constant in time (but unknown).  
#   How does the KF fare in estimating it?  
#   Does its uncertainty variance ever reach 0?
# - What is the KF uncertainty variance in the case of `M=0`?

# +
# ws.show_answer('KF behaviour')
# -

# #### Exc -- Temporal convergence
# In general, $\DynMod$, $\ObsMod$, $Q$, and $R$ depend on time, $k$
# (often to parameterize exogenous/outside factors/forces/conditions),
# and there are no limit values that the KF parameters converge to.
# But, we assumed that they are all stationary.
# In addition, suppose $Q=0$ and $\ObsMod = 1$.
# Show that
#
# - (a) $1/P\supa_k = 1/(\DynMod^2 P\supa_{k-1}) + 1/R$,
#   by combining the forecast and analysis equations for the variance.
# - (b) $1/P\supa_k = 1/P\supa_0 + k/R$, if $\DynMod = 1$.
# - (c) $P\supa_{\infty} = 0$, if $\DynMod = 1$.
# - (d) $P\supa_{\infty} = 0$, if $\DynMod < 1$.  
# - (e) $P\supa_{\infty} = R (1-1/\DynMod^2)$, if $\DynMod > 1$.  
#   *Hint: Look for the fixed point of the recursion of part (a).*

# +
# ws.show_answer('Asymptotic Riccati')
# -

# **Exc (optional) -- Temporal CV, part 2:**
# Now we don't assume that $Q$ is zero. Instead
# - (a) Suppose $\DynMod = 0$. What does $P\supa_k$ equal?
# - (b) Suppose $\DynMod = 1$. Show that $P\supa_\infty$
#       satisfies the quadratic equation: $0 = P^2 + Q P - Q R$.  
#       Thereby, without solving the quadratic equation, show that
#     - (c) $P\supa_\infty \rightarrow R$ (from below) if $Q \rightarrow +\infty$.
#     - (d) $P\supa_\infty \rightarrow \sqrt{ Q R}$ (from above) if $Q \rightarrow 0^+$.

# +
# ws.show_answer('Asymptotes when Q>0')
# -

# #### Exc -- Impact of biases
# Re-run the above interative animation to set the default control values. Answer the following
#
# - `logR_bias`/`logQ_bias` control the (multiplicative) bias in $R$/$Q$ that is fed to the KF.
#   What happens when the KF "thinks" the measurement/dynamical error
#   is (much) smaller than it actually is?
#   What about larger?
# - Re-run the animation to get default values.
#   Set `logQ` to 0, which will make the following behaviour easier to describe.
#   In the code, add 20 to the initial `xa` **given to the KF**.
#   How long does it take for it to recover from this initial bias?
# - Multiply `Pa` **given to the KF** by 0.01. What about now?
# - Remove the previous biases.
#   Instead, multiply `M` **given to the KF** by 2, and observe what happens.
#   Try the same, but dividing `M` by 2.

# +
# ws.show_answer('KF with bias')
# -

# ## Alternative methods
#
# When it comes to (especially univariate) time series analysis,
# the Kalman filter (KF) is not the only game in town.
# For example, **signal processing** offers several alternative filters.
# Indeed, the word "filter" in the KF originates in that domain,
# where it originally referred to the removal of high-frequency noise,
# since this tends to coincide with an improved estimate of the signal.
#
# But for the above problem (which is linear-Gaussian!),
# the KF is guaranteed (on average, in the long run, in terms of mean square error)
# to outperform any other method.
# We will see cases later (of full-blown state estimation)
# where the difference is much clearer,
# and indeed it might not even be clear how to apply signal processing methods.
# However, the KF has an unfair advantage: we are giving it a ton of information
# about the problem (`M, H, R, Q`) that the signal processing methods do not get.
# Therefore, they typically also require a good deal of tuning
# (in practice, so does the KF, since `Q` and `R` are rarely well determined).
# We will not review any signal processing theory here,
# but challenge you to make use of what `scipy` already has to offer.
#  
# #### Exc (optional) -- signal processing
# Run the following cell to import and define some more tools.

import scipy as sp
import scipy.signal as sig
def nrmlz(x):
    return x / x.sum()
def trunc(x, n):
    return np.pad(x[:n], (0, len(x)-n))

# Now try to "filter" the `obsrvs` to produce estimates of `truth`.
# In each case, add your estimate ("filtered signal" in that domain's parlance)
# to the `sigproc` dictionnary in the interactive animation cell,
# with an appropriate name/key (this will automatically include it in the plotting).  
# Use
# - (a) [`sig.wiener`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.wiener.html).  
#       *PS: this is a direct ancestor of the KF*.
# - (b) a moving average, for example [`sig.windows.hamming`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.hamming.html).  
#       *Hint: you may also want to use [`sig.convolve`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.convolve.html#scipy.signal.convolve)*.
# - (c) a low-pass filter using [`np.fft`](https://docs.scipy.org/doc/scipy/reference/fft.html#).  
#       *Hint: you may also want to use the above `trunc` function.*
# - (d) The [`sig.butter`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html) filter.
#       *Hint: apply with [`sig.filtfilt`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.filtfilt.html).*
# - (e) not really a signal processing method: [`sp.interpolate.UniveriateSpline`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.UnivariateSpline.html)
#
# The answers should be considered examples, not the uniquely right way.

# +
# ws.show_answer('signal processing', 'a')
# -

# #### Exc (optional) -- Analytic simplification in the case of an unknown constant
#
# - Note that in case $Q = 0$,
# then $x_{k+1} = \DynMod^k x_0$.  
# - So if $\DynMod = 1$, then $x_k = x_0$, so we are estimating an unknown *constant*,
# and can drop its time index subscript.  
# - For simplicity, assume $\ObsMod = 1$, and $P^a_0 \rightarrow +\infty$.  
# - Then $p(x | y_{1:k}) \propto \exp \big\{- \sum_l \| y_l - x \|^2_R / 2 \big\}
# = \NormDist(x | \bar{y}, R/k )$, which again follows by completing the square.  
# - In words, the (accumulated) posterior mean is the sample average,
#   $\bar{y} = \frac{1}{k}\sum_l y_l$,  
#   and the variance is that of a single observation divided by $k$.
#
# Show that this is the same posterior that the KF recursions produce.  
# *Hint: while this is straightforward for the variance,
# you will probably want to prove the mean using induction.*

# ## Summary
# As a subset of state estimation we can do time series estimation
# [(wherein state-estimation is called state-space approach)](https://www.google.com/search?q="We+now+demonstrate+how+to+put+these+models+into+state+space+form").
# Moreover, DA methods produce uncertainty quantification, something which is usually more obscure with time series analysis methods.
#
# ### Next: [T5 - Multivariate Kalman filter](T5%20-%20Multivariate%20Kalman%20filter.ipynb)
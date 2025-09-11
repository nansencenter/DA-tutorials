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

from resources import show_answer, interact, cInterval
# %matplotlib inline
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
plt.ion();

# # T4 - Time series filtering
#
# Before exploring the full (multivariate) Kalman filter (KF),
# let's first consider scalar but time-dependent (temporal/sequential) problems.
# $
# \newcommand{\Expect}[0]{\mathbb{E}}
# \newcommand{\NormDist}{\mathscr{N}}
# \newcommand{\DynMod}[0]{\mathscr{M}}
# \newcommand{\ObsMod}[0]{\mathscr{H}}
# \newcommand{\mat}[1]{{\mathbf{{#1}}}}
# \newcommand{\vect}[1]{{\mathbf{#1}}}
# \newcommand{\ta}[0]{\text{a}}
# \newcommand{\tf}[0]{\text{f}}
# $
#
# Consider the scalar, stochastic process $\{x_k\}$,
# generated for sequentially increasing time index $k$ by
#
# $$ x_{k+1} = \DynMod_k x_k + q_k \,. \tag{DynMod} $$
#
# For our present purposes, the **dynamical "model"** $\DynMod_k$ is simply a known number.
# Suppose we get observations $\{y_k\}$ as in:
#
# $$ y_k = \ObsMod_k x_k + r_k \,, \tag{ObsMod} $$
#
# The noises and $x_0$ are assumed to be independent of each other and across time
# (i.e., $\varepsilon_k$ is independent of $\varepsilon_l$ for $k \neq l$),
# and Gaussian with known parameters:
# $$x_0 \sim \NormDist(x^\ta_0, P^\ta_0),\quad
# q_k \sim \NormDist(0, Q_k),\quad
# r_k \sim \NormDist(0, R_k) \,.$$
#
# <a name="Example-problem:-AR(1)"></a>
#
# ## Example problem: AR(1)
#
# For simplicity (though the KF does not require these assumptions),
# suppose that $\DynMod_k = \DynMod$, i.e., it is constant in time.
# Then $\{x_k\}$ forms a so-called order-1 auto-regressive process [[Wikipedia](https://en.wikipedia.org/wiki/Autoregressive_model#Example:_An_AR(1)_process)].
# Similarly, we drop the time dependence (subscript $k$) from $\ObsMod_k, Q_k, R_k$.
# The code below simulates a random realization of this process.

# +
# Use H=1 so that it makes sense to plot data on the same axes as the state.
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

# The following code plots the process. *You don't need to read or understand it*.

@interact(seed=(1, 12), M=(0, 1.03, .01), nTime=(0, 100),
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
            plt.plot(kk, estimates[:, 1], label=r'Kalman$^a$ ± 1$\sigma$')
            plt.fill_between(kk, *cInterval(estimates[:, 1], variances[:, 1]), alpha=.2)
        else:
            kk2 = kk.repeat(2)
            plt.plot(kk2, estimates.flatten(), label=r'Kalman ± 1$\sigma$')
            plt.fill_between(kk2, *cInterval(estimates, variances), alpha=.2)
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
#
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
# show_answer('AR1')
# -

# <a name="The-(univariate)-Kalman-filter-(KF)"></a>
#
# ## The (univariate) Kalman filter (KF)
#
# Now we have a random variable that evolves in time, that we can *pretend* is unknown,
# in order to estimate (or "track") it.
# From above,
# $p(x_0) = \NormDist(x_0 | x^\ta_0, P^\ta_0)$ with given parameters.
# We also know that $x_k$ evolves according to eqn. (DynMod).
# Therefore, as shown in the following exercise,
# $p(x_1) = \NormDist(x_1 | x^\tf_1, P^\tf_1)$, with
# $$
# \begin{align}
# x^\tf_k &= \DynMod \, x^\ta_{k-1} \tag{5} \\
# P^\tf_k &= \DynMod^2 \, P^\ta_{k-1} + Q \tag{6}
# \end{align}
# $$
#
# Formulae (5) and (6) are called the **forecast step** of the KF.
# But when $y_1$ becomes available (according to eqn. (ObsMod)),
# we can update/condition our estimate of $x_1$, i.e., compute the posterior,
# $p(x_1 | y_1) = \NormDist(x_1 \mid x^\ta_1, P^\ta_1)$,
# using the formulae we developed for Bayes' rule with
# [Gaussian distributions](T3%20-%20Bayesian%20inference.ipynb#Linear-Gaussian-Bayes'-rule-(1D)).
#
# $$
# \begin{align}
#   P^\ta_k &= 1/(1/P^\tf_k + \ObsMod^2/R) \,, \tag{7} \\\
#   x^\ta_k  &= P^\ta_k (x^\tf/P^\tf_k + \ObsMod y_k/R) \,.  \tag{8}
# \end{align}
# $$
#
# This is called the **analysis step** of the KF.
# We call this the **analysis step** of the KF.
# We can subsequently apply the same two steps again
# to produce forecast and analysis estimates for the next time index, $k+1$.
# Note that if $k$ is a date index, then "yesterday's forecast becomes today's prior".
#
# #### Exc -- linear algebra of Gaussian random variables
#
# - (a) Show the linearity of the expectation operator:
#   $\Expect [ \DynMod  x + b ] = \DynMod \Expect[x] + b$, for some constant $b$.
# - (b) Thereby, show that $\mathbb{Var}[ \DynMod  x + b ] = \DynMod^2 \mathbb{Var} [x]$.
# - (c) *Optional*: Now let $z = x + q$, with $x$ and $q$ independent and Gaussian.
#   Then the pdf of this sum of random variables, $p_z(z)$, is given by convolution
#   (hopefully this makes intuitive sense, at least in the discrete case):
#   $$ p_z(z) = \int p_x(x) \, p_q(z - x) \, d x \,.$$
#   Show that $z$ is also Gaussian,
#   whose mean and variance are the sum of the means and variances (respectively).  
#   *Hint: you will need the result on [completing the square](T3%20-%20Bayesian%20inference.ipynb#Exc----BR-LG1),
#   specifically the part that we did not make use of for Bayes' rule.  
#   If you get stuck, you can also view the excellent [`3blue1brown`](https://www.youtube.com/watch?v=d_qvLDhkg00&t=266s&ab_channel=3Blue1Brown) on the topic.*

# +
# show_answer('Sum of Gaussians', 'a')
# -

# #### The (general) Bayesian filtering recursions
#
# In the case of linearity and Gaussianity,
# the KF of eqns. (5)-(8) computes the *exact* Bayesian pdfs for $x_k$.
# But even without these assumptions,
# a general (abstract) Bayesian **recursive** procedure can still be formulated,
# relying only on the remaining ("hidden Markov model") assumptions.
#
# - The analysis "assimilates" $y_k$ to compute $p(x_k | y_{1:k})$,
#   where $y_{1:k} = y_1, \ldots, y_k$ is shorthand notation.
#   $$
#   p(x_k | y_{1:k}) \propto p(y_k | x_k) \, p(x_k | x_{1:k-1})
#   $$
# - The forecast "propagates" the estimate with its uncertainty
#   to produce $p(x_{k+1}| y_{1:k})$.
#   $$
#   p(x_{k+1} | y_{1:k}) = \int p(x_{k+1} | x_k) \, p(x_k | y_{1:k}) \, d x_k
#   $$
#
# It is important to appreciate the benefits of the recursive form of these computations:
# It reflects the recursiveness (Markov property) of nature:
# Both in the problem and our solution, time $k+1$ *builds on* time $k$,
# so we do not need to re-do the entire problem for each $k$.
# At every time $k$, we only deal with functions of one or two variables: $x_k$ and $x_{k+1}$,
# which is a much smaller space (for quantifying our densities or covariances)
# than that of the joint pdf $p(x_{1:k} | y_{1:k})$.
#
# Note, however, that our recursive procedure, called ***filtering***,
# does *not* compute $p(x_l | y_{1:k})$ for any $l < k$.
# In other words, any filtering estimate only contains *past* information.
# Updating estimates of the state at previous times is called ***smoothing***.
# However, for prediction/forecasting, filtering is all we need:
# accurate initial conditions (estimates of the present moment).
#
# #### Exc -- Implementation
#
# Below is a very rudimentary sequential estimator (not the KF!), which essentially just does "persistence" forecasts and sets the analysis estimates to the value of the observations (*which is only generally possible in this linear, scalar case*). Run its cell to define it, and then re-run the above interactive animation cell. Then:
#
# - Implement the KF properly by replacing the forecast and analysis steps below. *Re-run the cell.*
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
# show_answer('KF1 code')
# -

# #### Exc -- KF behaviour
#
# - Set `logQ` to its minimum, and `M=1`.  
#   We established in Exc "AR1" that the true states are now constant in time (but unknown).  
#   How does the KF fare in estimating it?  
#   Does its uncertainty variance ever reach 0?
# - What is the KF uncertainty variance in the case of `M=0`?

# +
# show_answer('KF behaviour')
# -

# <a name="Exc----Temporal-convergence"></a>
#
# #### Exc -- Temporal convergence
#
# In general, $\DynMod$, $\ObsMod$, $Q$, and $R$ depend on time, $k$
# (often to parameterize exogenous/outside factors/forces/conditions),
# and there are no limit values that the KF parameters converge to.
# But, we assumed that they are all stationary.
# In addition, suppose $Q=0$ and $\ObsMod = 1$.
# Show that
#
# - (a) $1/P^\ta_k = 1/(\DynMod^2 P^\ta_{k-1}) + 1/R$,
#   by combining the forecast and analysis equations for the variance.
# - (b) $1/P^\ta_k = 1/P^\ta_0 + k/R$, if $\DynMod = 1$.
# - (c) $P^\ta_{\infty} = 0$, if $\DynMod = 1$.
# - (d) $P^\ta_{\infty} = 0$, if $\DynMod < 1$.  
# - (e) $P^\ta_{\infty} = R (1-1/\DynMod^2)$, if $\DynMod > 1$.  
#   *Hint: Look for the fixed point of the recursion of part (a).*

# +
# show_answer('Asymptotic Riccati', 'a')
# -

# **Exc (optional) -- Temporal CV, part 2:**
# Now we don't assume that $Q$ is zero. Instead
#
# - (a) Suppose $\DynMod = 0$. What does $P^\ta_k$ equal?
# - (b) Suppose $\DynMod = 1$. Show that $P^\ta_\infty$
#   satisfies the quadratic equation: $0 = P^2 + Q P - Q R$.  
#   Thereby, without solving the quadratic equation, show that
#   - (c) $P^\ta_\infty \rightarrow R$ (from below) if $Q \rightarrow +\infty$.
#   - (d) $P^\ta_\infty \rightarrow \sqrt{ Q R}$ (from above) if $Q \rightarrow 0^+$.

# +
# show_answer('Asymptotes when Q>0')
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
#
# #### Exc -- Impact of biases
#
# Re-run the above interactive animation to set the default control values. Answer the following
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
# show_answer('KF with bias')
# -

# ## Alternative methods
#
# When it comes to (especially univariate) time series analysis,
# the Kalman filter (KF) is not the only option.
# For example, **signal processing** offers several alternative filters.
# Indeed, the word "filter" in the KF comes from that domain,
# where it originally referred to removing high-frequency noise,
# since this often leads to a better estimate of the signal.
# We will not review signal processing theory here,
# but challenge you to make use of what `scipy` already has to offer.
#
# #### Exc (optional) -- signal processing
#
# Run the following cell to import and define some more tools.

import scipy as sp
import scipy.signal as sig
def nrmlz(x):
    return x / x.sum()
def trunc(x, n):
    return np.pad(x[:n], (0, len(x)-n))

# Now try to "filter" the `obsrvs` to produce estimates of `truth`.
# For each method, add your estimate ("filtered signal" in signal processing parlance)
# to the `sigproc` dictionary in the interactive animation cell,
# using an appropriate name/key (this will automatically include it in the plot).
# Use
#
# - (a) [`sig.wiener`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.wiener.html).  
#   *PS: this is a direct ancestor of the KF*.
# - (b) a moving average, for example [`sig.windows.hamming`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.hamming.html).  
#   *Hint: you may also want to use [`sig.convolve`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.convolve.html#scipy.signal.convolve)*.
# - (c) a low-pass filter using [`np.fft`](https://docs.scipy.org/doc/scipy/reference/fft.html#).  
#   *Hint: you may also want to use the above `trunc` function.*
# - (d) The [`sig.butter`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html) filter.
#   *Hint: apply with [`sig.filtfilt`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.filtfilt.html).*
# - (e) not really a signal processing method: [`sp.interpolate.UniveriateSpline`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.UnivariateSpline.html)
#
# The answers should be considered examples, not the uniquely right way.

# +
# show_answer('signal processing', 'a')
# -

# But for the above problem (which is linear-Gaussian!),
# the KF is guaranteed (on average, in the long run, in terms of mean square error)
# to outperform any other method.
# We will see cases later (in full-blown state estimation)
# where the difference is much clearer,
# and indeed it might not even be clear how to apply signal processing methods.
# However, the KF has an unfair advantage: we are giving it a lot of information
# about the problem (`M, H, R, Q`) that the signal processing methods do not have.
# Therefore, those methods typically require a good deal of tuning
# (but in practice, so does the KF, since `Q` and `R` are rarely well determined).
#
# ## Summary
#
# The Kalman filter (KF) can be derived by applying linear-Gaussian assumptions
# to a sequential inference problem.
# Generally, the uncertainty never converges to 0,
# and the performance of the filter depends entirely on
# accurate system parameters (models and error covariance matrices).
#
# As a subset of state estimation (i.e., the KF), we can do classical time series estimation
# [(wherein state-estimation is called the state-space approach)](https://www.google.co.uk/search?q=%22We+now+demonstrate+how+to+put+these+models+into+state+space+form%22&btnG=Search+Books&tbm=bks).
# Moreover, DA methods produce uncertainty quantification, which is usually more obscure with time series analysis methods.
#
# ### Next: [T5 - Multivariate Kalman filter](T5%20-%20Multivariate%20Kalman%20filter.ipynb)
#
# <a name="References"></a>
#
# ### References

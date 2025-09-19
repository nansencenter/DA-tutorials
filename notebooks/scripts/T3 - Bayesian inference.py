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

from resources import show_answer, interact, import_from_nb, get_jointplotter
# %matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
plt.ion();

# # T3 - Bayesian inference
#
# The [previous tutorial](T2%20-%20Gaussian%20distribution.ipynb)
# $
# \newcommand{\Expect}[0]{\mathbb{E}}
# \newcommand{\NormDist}{\mathscr{N}}
# \newcommand{\ObsMod}[0]{\mathscr{H}}
# \newcommand{\mat}[1]{{\mathbf{{#1}}}}
# \newcommand{\vect}[1]{{\mathbf{#1}}}
# \newcommand{\ta}[0]{\text{a}}
# \newcommand{\tf}[0]{\text{f}}
# $
# studied the Gaussian probability density function (pdf), defined in 1D by:
# $$ \large \NormDist(x \mid \mu, \sigma^2) = (2 \pi \sigma^2)^{-1/2} e^{-(x-\mu)^2/2 \sigma^2} \,,\tag{G1} $$
# which we implemented and tested alongside the uniform distribution.

(pdf_G1, pdf_U1, bounds, dx, grid1d, mean_and_var) = import_from_nb("T2", ("pdf_G1", "pdf_U1", "bounds", "dx", "grid1d", "mean_and_var"))
pdfs = dict(N=pdf_G1, U=pdf_U1)


# *We no longer use uppercase to distinguish random variables from their outcomes (an unfortunate consequence of the myriad of notations to keep track of)!*
#
# Now that we have reviewed some probability, we can turn to statistical inference and estimation. In particular, we will focus on
#
# # Bayes' rule
#
# <details style="border: 1px solid #aaaaaa; border-radius: 4px; padding: 0.5em 0.5em 0;">
#   <summary style="font-weight: normal; font-style: italic; margin: -0.5em -0.5em 0; padding: 0.5em;">
#     In the Bayesian approach, uncertain knowledge (i.e. belief) about some unknown ($x$) is quantified through probability ... (optional reading 🔍)
#   </summary>
#
#   For example, what is the temperature at the surface of Mercury (at some given point and time)?
#   Not many people know the answer. Perhaps you say $500^{\circ} C, \, \pm \, 20$.
#   But that's hardly anything compared to you real uncertainty, so you revise that to $\pm \, 1000$.
#   But then you're allowing for temperature below absolute zero, which you of course don't believe is possible.
#   You can continue to refine the description of your uncertainty.
#   Ultimately (in the limit) the complete way to express your belief is as a *distribution*
#   (essentially just a list) of plausibilities for all possibilities.
#   Furthermore, the only coherent way to reason in the presence of such uncertainty
#   is to obey the laws of probability ([Jaynes (2003)](#References)).
#
#   - - -
# </details>
#
# **Bayes' rule** is the fundamental tool for inference: it tells us how to condition/merge/assimilate/update our beliefs based on data/observation ($y$).
# For *continuous* random variables $x$ and $y$, Bayes' rule reads:
# $$
# \large
# \color{red}{\overset{\mbox{Posterior}}{p(\color{black}{x|y})}} = \frac{\color{blue}{\overset{\mbox{  Prior  }}{p(\color{black}{x})}} \, \color{green}{\overset{\mbox{ Likelihood}}{p(\color{black}{y|x})}}}{\color{gray}{\underset{\mbox{Constant wrt. x}}{p(\color{black}{y})}}} \,. \tag{BR} \\[1em]
# $$
#
# **Exc – Bayes' rule derivation:** Derive eqn. (BR) from the definition of [conditional pdfs](https://en.wikipedia.org/wiki/Conditional_probability_distribution#Conditional_continuous_distributions).

# +
# show_answer('symmetry of conjunction')
# -

# It is hard to overstate the simplicity of Bayes' rule, eqn. (BR): it consists merely of scalar multiplication and division.
# However, our goal is to compute the function $p(x|y)$ for **all values of $x$**.
# Thus, upon discretization, eqn. (BR) becomes the multiplication of two *arrays* of values (followed by normalization):

def Bayes_rule(prior_values, lklhd_values, dx):
    prod = prior_values * lklhd_values         # pointwise multiplication
    posterior_values = prod/(np.sum(prod)*dx)  # normalization
    return posterior_values


# #### Exc (optional) – BR normalization
#
# Show that the normalization in `Bayes_rule()` amounts to (approximately) the same as dividing by $p(y)$.

# +
# show_answer('quadrature marginalisation')
# -

# In fact, since $p(y)$ is implicitly known in this way,
# we often omit it, simplifying Bayes' rule (eqn. BR) to
#
# $$ p(x|y) \propto p(x) \, p(y|x) \,.  \tag{BR2} $$
#
# Do we even need to care about $p(y)$ at all? In practice, all we need to know is how much more likely one value of $x$ (or an interval around it) is compared to another.
# Normalization is only necessary because of the *convention* that all densities integrate to $1$.
# However, for large models, we can usually only afford to evaluate $p(y|x)$ at a few points (of $x$), so the integral for $p(y)$ can only be roughly approximated. In such settings, estimating the normalization factor becomes an important question too.
#
# <a name="Interactive-illustration"></a>
#
# ## Interactive illustration
#
# The code below shows Bayes' rule in action.

@interact(y=(*bounds, 1), logR=(-3, 5, .5), prior_kind=list(pdfs), lklhd_kind=list(pdfs))
def Bayes1(y=9.0, logR=1.0, lklhd_kind="N", prior_kind="N"):
    R = 4**logR
    xf = 10
    Pf = 4**2

    # (Ref. later exercise)
    def H(x):
        return 1*x + 0

    x = grid1d
    prior_vals = pdfs[prior_kind](x, xf, Pf)
    lklhd_vals = pdfs[lklhd_kind](y, H(x), R)
    postr_vals = Bayes_rule(prior_vals, lklhd_vals, dx)

    def plot(x, y, c, lbl):
        plt.fill_between(x, y, color=c, alpha=.3, label=lbl)

    plt.figure(figsize=(8, 4))
    plot(x, prior_vals, 'blue'  , f'Prior, {prior_kind}(x | {xf:.4g}, {Pf:.4g})')
    plot(x, lklhd_vals, 'green' , f'Lklhd, {lklhd_kind}({y:<3}| H(x), {R:.4g})')
    plot(x, postr_vals, 'red'   , 'Postr, pointwise: ?(x | %4.2g, %4.2g)' % mean_and_var(postr_vals, x))

    # (Ref. later exercise)
    try:
        H_lin = H(xf)/xf # a simple linear approximation of H(x)
        xa, Pa = Bayes_rule_LG1(xf, Pf, y, H_lin, R)
        label = f'Postr, parametric: N(x | {xa:4.2g}, {Pa:4.2g})'
        postr_vals_G1 = pdf_G1(x, xa, Pa)
        plt.plot(x, postr_vals_G1, 'purple', label=label)
    except NameError:
        pass

    plt.ylim(0, 0.6)
    plt.legend(loc="upper left", prop={'family': 'monospace'})
    plt.show()


# The illustration uses a
#
# - prior $p(x) = \NormDist(x|x^f, P^f)$ with (fixed) mean and variance, $x^f= 10$, $P^f=4^2$
#   (but you can of course change these in the code above)
# - likelihood $p(y|x) = \NormDist(y|x, R)$, whose parameters are set by the interactive sliders.
#
# We are now dealing with three (!) separate distributions,
# which introduces a lot of symbols to keep track of – a necessary evil for later.
#
# **Exc – `Bayes1` properties:** This exercise serves to make you acquainted with how Bayes' rule blends information.
#
# Move the sliders (use arrow keys?) to animate it, and answer the following (with the boolean checkmarks both on and off).
#
# - What happens to the posterior when $R \rightarrow \infty$ ?
# - What happens to the posterior when $R \rightarrow 0$ ?
# - Move $y$ around. What is the posterior's location (mean/mode) when $R$ equals the prior variance?
# - Can you say something universally valid (for any $y$ and $R$) about the height of the posterior pdf?
# - Does the posterior scale (width) depend on $y$?  
#    *Optional*: What does this mean [information-wise](https://en.wikipedia.org/wiki/Differential_entropy#Differential_entropies_for_various_distributions)?
# - Consider the shape (ignoring location & scale) of the posterior. Does it depend on $R$ or $y$?
# - Can you see a shortcut to computing this posterior rather than having to do the pointwise multiplication?
# - For the case of two uniform distributions: What happens when you move the prior and likelihood too far apart? Is the fault of the implementation, the math, or the problem statement?
# - Play around with the grid resolution (see the cell above). What is in your opinion a "sufficient" grid resolution?

# +
# show_answer('Posterior behaviour')
# -

# ## With forward (observation) models
#
# In general, the observation $y$ is not a "direct" measurement of $x$, as above,
# but rather some transformation, i.e. function of $x$,
# which is called **observation/forward model**, $\ObsMod$.
# Examples include:
#
# - $\ObsMod(x) = x + 273$ for a thermometer reporting °C, while $x$ is the temperature in °K.
# - $\ObsMod(x) = 10 x$ for a ruler using mm, while $x$ is stored as cm.
# - $\ObsMod(x) = \log(x)$ for litmus paper (pH measurement), where $x$ is the molar concentration of hydrogen ions.
# - $\ObsMod(x) = |x|$ for bicycle speedometers (measuring rpm, i.e. Hall effect sensors).
# - $\ObsMod(x) = 2 \pi h \, x^2$ if observing inebriation (drunkenness), and the unknown, $x$, is the radius of the beer glasses.
#
# Of course, the linear and logarithmic transformations are hardly worthy of the name "model", since they merely change the scale of measurement, and so could be trivially done away with. But doing so is not necessary, and they will serve to illustrate some important points.
#
# In addition, measurement instruments always (at least for continuous variables) have limited accuracy,
# i.e. there is an **measurement noise/error** corrupting the observation. For simplicity, this noise is usually assumed *additive*, so that the observation, $y$, is related to the true state, $x$, by
# $$
# y = \ObsMod(x) + r \,, \;\; \qquad \tag{Obs}
# $$
# and $r \sim \NormDist(0, R)$ for some variance $R>0$.
# Then the likelihood is $$p(y|x) = \NormDist(y| \ObsMod(x), R) \,. \tag{Lklhd}$$
#
# **Exc (optional) – The likelihood:** Derive the expression (Lklhd) for the likelihood.

# +
# show_answer('Likelihood')
# -

# <a name="Exc----Obs.-model-gallery"></a>
#
# #### Exc – Obs. model gallery
#
# Consider the following observation models.
#
# - (a) $\ObsMod(x) = x + 15$.
# - (b) $\ObsMod(x) = 2 x$.
# - (c) $\ObsMod(x) = (x-5)^2$.
#   - Explain how negative values of $y$ are possible.
# - (d) Try $\ObsMod(x) = |x|$.
#
# In each case, describe how and why you'd expect the likelihood to change (as compared to $\ObsMod(x) = x$).
# Then verify your answer by implementing `H` in the [interactive Bayes' rule](#Interactive-illustration).

# +
# show_answer('Observation models', 'a')
# -

# It is important to appreciate that the likelihood, and its role in Bayes' rule, does not perform any "inversion". It simply quantifies how well each $x$ fits the data, in terms of its weighting. This approach also inherently handles the fact that multiple values of $x$ may be plausible.
#
# **Exc (optional) – "why inverse":** Laplace called "statistical inference" the reasoning of "inverse probability" (1774). You may also have heard of "inverse problems" in reference to similar problems, but without a statistical framing. In view of this, why do you think we use $x$ for the unknown, and $y$ for the known/given data?

# +
# show_answer("what's forward?")
# -

# <a name="Linear-Gaussian-Bayes'-rule-(1D)"></a>
#
# ## Linear-Gaussian Bayes' rule (1D)
#
# To address this computational difficulty, we can take a more analytical ("pen-and-paper") approach: instead of computing the full posterior, we compute only its parameters (mean and (co)variance).
# This is straightforward in the linear-Gaussian case, i.e. when $\ObsMod$ is linear (just a number). For readability, the unknown, $x$, is colored.
#
# - Given the prior of $p(\color{darkorange}{x}) = \NormDist(\color{darkorange}{x} \mid x^\tf, P^\tf)$
# - and a likelihood $p(y|\color{darkorange}{x}) = \NormDist(y \mid \ObsMod \color{darkorange}{x},R)$,  
# - $\implies$ posterior $
#   p(\color{darkorange}{x}|y)
#   = \NormDist(\color{darkorange}{x} \mid x^\ta, P^\ta) \,, $
#   where, in the 1-dimensional/univariate/scalar (multivariate is discussed in [T5](T5%20-%20Multivariate%20Kalman%20filter.ipynb)) case:
#   $$\begin{align}
#     P^\ta &= 1/(1/P^\tf + \ObsMod^2/R) \,, \tag{5} \\\
#     x^\ta &= P^\ta (x^\tf/P^\tf + \ObsMod y/R) \,.  \tag{6}
#   \end{align}$$
#
# The proof is in the following exercise.
#
# #### Exc – BR-LG1
#
# Consider the following identity, where $P^\ta$ and $x^\ta$ are given by eqns. (5) and (6).
# $$
# \frac{(\color{darkorange}{x}-x^\tf)^2}{P^\tf} + \frac{(\ObsMod \color{darkorange}{x}-y)^2}{R} \quad =
# \quad \frac{(\color{darkorange}{x} - x^\ta)^2}{P^\ta} + \frac{(y - \ObsMod x^\tf)^2}{R + P^\tf} \,, \tag{LG1}
# $$
# Notice that the left hand side (LHS) is the sum of *two* squares with $\color{darkorange}{x}$,
# but the RHS only contains *one*.
#
# - (a) Actually derive the first term of the RHS of (LG1), i.e. eqns. (5) and (6).  
#   *Hint: you can simplify the task by first "hiding" $\ObsMod$*
# - (b) *Optional*: Derive the full RHS (i.e. also the second term).
# - (c) Show that $p(\color{darkorange}{x}|y) = \NormDist(\color{darkorange}{x} \mid x^\ta, P^\ta)$
#   using part (a), Bayes' rule (BR2), and the Gaussian pdf (G1).

# +
# show_answer('BR Gauss, a.k.a. completing the square', 'a')
# -

# **Exc – Temperature example:**
# The statement $x = \mu \pm \sigma$ is *sometimes* used
# as a shorthand for $p(x) = \NormDist(x \mid \mu, \sigma^2)$. Suppose
#
# - you think the temperature $x = 20°C \pm 2°C$,
# - a thermometer yields the observation $y = 18°C \pm 2°C$.
#
# Show that your posterior is $p(x|y) = \NormDist(x \mid 19, 2)$

# +
# show_answer('LG BR example')
# -

# The following implements the linear-Gaussian Bayes' rule (eqns. 5 and 6).
# Note that its inputs and outputs are not arrays (as in `Bayes_rule()`), but simply scalar numbers: the means, variances, and $\ObsMod$.

def Bayes_rule_LG1(xf, Pf, y, H, R):
    Pa = 1 / (1/Pf + H**2/R)
    xa = Pa * (xf/Pf + H*y/R)
    return xa, Pa


# #### Exc – Gaussianity as an approximation
#
# - (a) Again, try the various $\ObsMod$ from the [above exercise](#Exc----Obs.-model-gallery) in the [interactive Bayes' rule widget](#Interactive-illustration).  
#   For which $\ObsMod$ does `Bayes_rule_LG1()` reproduce `Bayes_rule()`?
# - (b) For simplicity, revert back to the identity for $\ObsMod$.
#   Then run the cell below, which fits distributions [from `scipy`s library of distributions](https://stackoverflow.com/questions/37559470/)
#   so as to respect `mu` and `sig2` and adds them to the pdfs in the dropdown menus of the widget
#   (upon re-running its cell). Try them out for the likelihood, and answer the following.
#   Which ones
#
#   - Are skewed (or at least asymmetric) ?
#   - Have excess kurtosis (tails heavier than for the Gaussian) ?
#   - Produces a posterior variance that increases with the distance prior-observation ?  
#     *PS: this one (the Student's) is frequently used
#     to increase the robustness of the (ensemble) Kalman filter,
#     or [estimate the "inflation" factor](https://rmets.onlinelibrary.wiley.com/doi/10.1002/qj.3386).*

import scipy.stats as ss
for dist in [
  ss.chi2(df=3),
  ss.beta(a=5, b=2),
  ss.anglit(),
  ss.t(df=3),
]:
    def pdf_fitted(x, mu, sigma2, dist=dist):
        mean, var = dist.stats(moments='mv')
        # mean, var = mean_and_var(dist.pdf(grid1d), grid1d)
        ratio = np.sqrt(var / sigma2)
        u = ratio * (x - mu) + mean
        return ratio * dist.pdf(u)
    pdfs[dist.dist.name + str(dist.kwds)] = pdf_fitted

# **Exc (optional) – Gain algebra:** Show that eqn. (5) can be written as
# $$P^\ta = K R / \ObsMod \,,    \tag{8}$$
# where
# $$K = \frac{\ObsMod P^\tf}{\ObsMod^2 P^\tf + R} \,,    \tag{9}$$
# is called the "Kalman gain".  
# *Hint: again, try to "hide away" $\ObsMod$ among the other objects before proceeding.*
#
# Then shown that eqns. (5) and (6) can be written as
# $$
# \begin{align}
#     P^\ta &= (1-K \ObsMod) P^\tf \,,  \tag{10} \\\
#   x^\ta &= x^\tf + K (y- \ObsMod x^\tf) \tag{11} \,,
# \end{align}
# $$

# +
# show_answer('BR Kalman1 algebra')
# -

# #### Exc (optional) – Gain intuition
#
# Let $\ObsMod = 1$ for simplicity.
#
# - (a) Show that $0 < K < 1$ since $0 < P^\tf, R$.
# - (b) Show that $P^\ta < P^\tf, R$.
# - (c) Show that $x^\ta$ is in the interval $(x^\tf, y)$.
# - (d) Why do you think $K$ is called a "gain"?

# +
# show_answer('KG intuition')
# -

# **Exc – BR with Gain:** Re-define `Bayes_rule_LG1` so to as to use eqns. 9-11. Remember to re-run the cell. Verify that you get the same plots as before.

# +
# show_answer('BR Kalman1 code')
# -

# #### Exc (optional) – optimalities
#
# In contrast to orthodox statistics,
# Bayes' rule (BR) does not attempt to produce a single estimate/value of $x$.
# It merely states how to update our quantitative belief (weighted possibilities) in light of new data.
# Barring any approximations (such as using `Bayes_rule_LG1` outside the linear-Gaussian case),
# the (full) posterior will be **optimal** from the perspective of any [proper scoring rule](https://en.wikipedia.org/wiki/Scoring_rule#Propriety_and_consistency).
#
# *But if you must* pick a single point value estimate $\hat{x}$
# (in order to perform a contingent action, without [robust optimisation](https://en.wikipedia.org/wiki/Robust_optimization)),
# you can **decide** on it by optimising (with respect to $\hat{x}$)
# the expectation (with respect to $x$) of some utility/loss function,
# i.e. $\Expect\, \text{Loss}(x - \hat{x})$.
# For instance, if the posterior pdf happens to be symmetric
# (as in the linear-Gaussian context above),
# and your loss function is convex and symmetric,
# then the mean/median will be optimal [[Lehmann & Casella (1998)](#References), Corollary 7.19].
# More specifically, for any given distribution of $x$,
# the optimal [Bayes estimator](https://en.wikipedia.org/wiki/Bayes_estimator) is:
#
# - the mode if $\text{Loss}(d) = \begin{cases} 0 & \text{if } d = 0 \\ 1 & \text{otherwise} \end{cases}$
# - the median if $\text{Loss}(d) = |d|$
# - the mean if $\text{Loss}(d) = d^2$
#
# The last case (squared-error loss) is most commonly used,
# and the resulting estimator is sometimes called
# the minimum mean-square error (MMSE) estimator.
# *Prove that the MMSE is indeed the mean of the distribution!*

# +
# show_answer('MMSE')
# -

# However, it is not generally easy to find the posterior mean, median, or mode,
# so these optimalities mainly serve to justify a preference for $x^\ta$
# in the linear-Gaussian case.
#
# <details style="border: 1px solid #aaaaaa; border-radius: 4px; padding: 0.5em 0.5em 0;">
#   <summary style="font-weight: normal; font-style: italic; margin: -0.5em -0.5em 0; padding: 0.5em;">
#       It is possible to drop the Gaussianity assumption and still
#       claim optimality for $x^\ta$ of eqns. (6) and (11) as
#       the best (min. variance), linear, unbiased, estimate (BLUE ... 🔍).
#   </summary>
#   The result requires reformulating the prior
#   as "background" zero-mean noise onto the (non-random) $x$,
#   whose outcome was the prior mean, $x^\tf$, and whose covariance is $P^\tf$.
#   Then, by explicit augmentation (i.e. pseudo-obs: $[y, x^\tf]$) one recovers the linear regression problem
#   of the celebrated Gauss-Markov theorem, generalized by Aitken to the case of correlated noise.
#   A similar proof uses an ansatz linear in both $x^\tf$ and $y$ without concatenating them.
#   *PS: this results is also [sometimes](https://en.wikipedia.org/wiki/Minimum_mean_square_error#Linear_MMSE_estimator) reframed as MMSE,
#   causing confusion with the above meaning of the acronym.*
#
#   If Gaussianity is again assumed (but the perspective remains "frequentist"),
#   then one can drop the linearity requirement,
#   yielding the (uniformly) minimum-variance unbiased estimate (UMVUE),
#   as per [Lehmann-Sheffé](https://stats.stackexchange.com/a/398911), or [Cramér-Rao](https://stats.stackexchange.com/a/596307).
#   However, the linearity imposition cannot generally be omitted [Pötscher & Preinerstorfer (2024)](#References).
#
#   - - -
# </details>
#
# All in all, the intuitive idea of **considering the mean of $p(x)$ as the point
# estimate** has good theoretical foundations.
#
# ## Summary
#
# Bayesian inference quantifies uncertainty (in $x$) using probability.
# Bayes' rule tells us how to update this belief based on data or observation ($y$).
# It is simply a reformulation of conditional probability.
# Observation can be "inverted" using Bayes' rule,
# in the sense that all possibilities for $x$ are weighted.
# While technically simple, Bayes' rule requires many pointwise multiplications.
# But if Gaussianity can be assumed, it reduces to just two formulae.
#
# ### Next: [T4 - Filtering & time series](T4%20-%20Time%20series%20filtering.ipynb)
#
# <a name="References"></a>
#
# ### References
#
# <!--
# @book{jaynes2003probability,
#   title={Probability theory: the logic of science},
#   author={Jaynes, Edwin T},
#   year={2003},
#   publisher={Cambridge university press}
# }
#
# @book{lehmann1998theory,
#     title={Theory of Point Estimation},
#     author={Lehmann, Erich Leo and Casella, George},
#     volume={31},
#     year={1998},
#     publisher={Springer}
# }
# @article{potscher2024comment,
#   title={A Comment on:“A Modern Gauss--Markov Theorem”},
#   author={P{\"o}tscher, Benedikt M and Preinerstorfer, David},
#   journal={Econometrica},
#   volume={92},
#   number={3},
#   pages={913--924},
#   year={2024},
#   publisher={Wiley Online Library}
# }
# -->
#
# - **Jaynes (2003)**:
#   Edwin T. Jaynes, "Probability theory: the logic of science", 2003.
# - **Lehmann & Casella (1998)**:
#   "Theory of Point Estimation", 1998.
# - **Pötscher & Preinerstorfer (2024)**:
#   Benedikt M. Pötscher and David Preinerstorfer, "A Comment on: 'A Modern Gauss-Markov Theorem'", Econometrica, 2024.

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

# # T3 - Bayesian inference
# Now that we have reviewed some probability, we can look at statistical inference.
# $
# % ######################################## Loading TeX (MathJax)... Please wait ########################################
# \newcommand{\Reals}{\mathbb{R}} \newcommand{\Expect}[0]{\mathbb{E}} \newcommand{\NormDist}{\mathcal{N}} \newcommand{\DynMod}[0]{\mathscr{M}} \newcommand{\ObsMod}[0]{\mathscr{H}} \newcommand{\mat}[1]{{\mathbf{{#1}}}} \newcommand{\bvec}[1]{{\mathbf{#1}}} \newcommand{\trsign}{{\mathsf{T}}} \newcommand{\tr}{^{\trsign}} \newcommand{\ceq}[0]{\mathrel{≔}} \newcommand{\xDim}[0]{D} \newcommand{\supa}[0]{^\text{a}} \newcommand{\supf}[0]{^\text{f}} \newcommand{\I}[0]{\mat{I}} \newcommand{\K}[0]{\mat{K}} \newcommand{\bP}[0]{\mat{P}} \newcommand{\bH}[0]{\mat{H}} \newcommand{\bF}[0]{\mat{F}} \newcommand{\R}[0]{\mat{R}} \newcommand{\Q}[0]{\mat{Q}} \newcommand{\B}[0]{\mat{B}} \newcommand{\C}[0]{\mat{C}} \newcommand{\Ri}[0]{\R^{-1}} \newcommand{\Bi}[0]{\B^{-1}} \newcommand{\X}[0]{\mat{X}} \newcommand{\A}[0]{\mat{A}} \newcommand{\Y}[0]{\mat{Y}} \newcommand{\E}[0]{\mat{E}} \newcommand{\U}[0]{\mat{U}} \newcommand{\V}[0]{\mat{V}} \newcommand{\x}[0]{\bvec{x}} \newcommand{\y}[0]{\bvec{y}} \newcommand{\z}[0]{\bvec{z}} \newcommand{\q}[0]{\bvec{q}} \newcommand{\br}[0]{\bvec{r}} \newcommand{\bb}[0]{\bvec{b}} \newcommand{\bx}[0]{\bvec{\bar{x}}} \newcommand{\by}[0]{\bvec{\bar{y}}} \newcommand{\barB}[0]{\mat{\bar{B}}} \newcommand{\barP}[0]{\mat{\bar{P}}} \newcommand{\barC}[0]{\mat{\bar{C}}} \newcommand{\barK}[0]{\mat{\bar{K}}} \newcommand{\D}[0]{\mat{D}} \newcommand{\Dobs}[0]{\mat{D}_{\text{obs}}} \newcommand{\Dmod}[0]{\mat{D}_{\text{obs}}} \newcommand{\ones}[0]{\bvec{1}} \newcommand{\AN}[0]{\big( \I_N - \ones \ones\tr / N \big)}
# $

remote = "https://raw.githubusercontent.com/nansencenter/DA-tutorials"
# !wget -qO- {remote}/master/notebooks/resources/colab_bootstrap.sh | bash -s
import resources.workspace as ws

# %matplotlib inline
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
plt.ion();

# The [previous tutorial](T2%20-%20Gaussian%20distribution.ipynb)
# studied the Gaussian probability density function (pdf), defined by:
#
# $$\begin{align}
# \mathcal{N}(x \mid \mu, \sigma^2) &= (2 \pi \sigma^2)^{-1/2} e^{-(x-\mu)^2/2 \sigma^2} \,,\tag{G1} \\
# \NormDist(\x \mid  \mathbf{\mu}, \mathbf{\Sigma})
# &=
# |2 \pi \mathbf{\Sigma}|^{-1/2} \, \exp\Big(-\frac{1}{2}\|\x-\mathbf{\mu}\|^2_\mathbf{\Sigma} \Big) \,, \tag{GM}
# \end{align}$$
# In this tutorial, we grab implentations straight from `scipy.stats`.

# +
def pdf_G1(x, meanval, variance):
    return sp.stats.norm.pdf(x, loc=meanval, scale=np.sqrt(variance))

def pdf_GM(points, mu, Sigma):
    diff = points - mu  # ensures both get broadcast
    zero = np.zeros(len(Sigma))
    return sp.stats.multivariate_normal(zero, Sigma).pdf(diff)


# -

# In addition,
# the following implements the the [uniform](https://en.wikipedia.org/wiki/Uniform_distribution_(continuous))
# (or "flat" or "box") pdf.

def pdf_U1(x, meanval, variance):
    # pdfx = sp.stats.uniform(loc=lower, scale=(upper-lower)).pdf(x)
    lower = meanval - np.sqrt(3*variance)
    upper = meanval + np.sqrt(3*variance)
    height = 1/(upper - lower)
    pdfx = height * np.ones_like(x)
    pdfx[x<lower] = 0
    pdfx[x>upper] = 0
    return pdfx

# We'll be playing with these distribution on the following numerical grid.

bounds = -15, 15
grid1d = np.linspace(*bounds, 201)
grid2d = np.dstack(np.meshgrid(grid1d, grid1d))
dx = grid1d[1] - grid1d[0]

# This will help illustrate:
#
# # Bayes' rule
# In the Bayesian approach, knowledge and uncertainty about the unknown ($x$)
# is quantified through probability.
# And **Bayes' rule** is how we do inference: it says how to condition/merge/assimilate/update this belief based on data/observation ($y$).
# For *continuous* "random variables", $x$ and $y$, it reads:
#
# $$\begin{align}
# p(x|y) &= \frac{p(x) \, p(y|x)}{p(y)} \, , \tag{BR} \\[1em]
# \text{i.e.} \qquad \texttt{posterior}\,\text{[pdf of $x$ given $y$]}
# \; &= \;
# \frac{\texttt{prior}\,\text{[pdf of $x$]}
# \; \times \;
# \texttt{likelihood}\,\text{[pdf of $y$ given $x$]}}
# {\texttt{normalisation}\,\text{[pdf of $y$]}} \, ,
# \end{align}
# $$

# Note that, in contrast to (the frequent aim of) classical statistics,
# Bayes' rule in itself makes no attempt at producing only a single estimate
# (but the topic is briefly discussed [further below](#Exc-(optional)----optimality-of-the-mean)).
# It merely states how quantitative belief (weighted possibilities) should be updated in view of new data.

# **Exc -- Bayes' rule derivation:** Derive eqn. (BR) from the definition of [conditional pdf's](https://en.wikipedia.org/wiki/Conditional_probability_distribution#Conditional_continuous_distributions).

# +
# ws.show_answer('symmetry of conditioning')
# -

# Bayes' rule, eqn. (BR), involves functions (the densities), but applies for any/all values of $x$ (and $y$).
# Thus, upon discretisation, eqn. (BR) becomes the multiplication of two arrays of values,
# followed by a normalisation (explained [below](#Exc-(optional)----BR-normalization)).
# It is hard to overstate how simple this principle is.

def Bayes_rule(prior_values, lklhd_values, dx):
    prod = prior_values * lklhd_values         # pointwise multiplication
    posterior_values = prod/(np.sum(prod)*dx)  # normalization
    return posterior_values


# #### Exc (optional) -- BR normalization
# Show that the normalization in `Bayes_rule()` amounts to (approximately) the same as dividing by $p(y)$.

# +
# ws.show_answer('quadrature marginalisation')
# -

# In fact, since $p(y)$ is thusly implicitly known,
# we often don't bother to write it down, simplifying Bayes' rule (eqn. BR) to
# $$\begin{align}
# p(x|y) \propto p(x) \, p(y|x) \, .  \tag{BR2}
# \end{align}$$
# Actually, do we even need to care about $p(y)$ at all? All we really need to know is how much more likely some value of $x$ (or an interval around it) is compared to any other $x$.
# The normalisation is only necessary because of the *convention* that all densities integrate to $1$.
# However, for large models, we usually can only afford to evaluate $p(y|x)$ at a few points (of $x$), so that the integral for $p(y)$ can only be roughly approximated. In such settings, estimation of the normalisation factor becomes an important question too.
#
# The code below shows Bayes' rule in action.

@ws.interact(y=(*bounds, 1), logR=(-9, 9), top=['y', 'logR'])
def Bayes1(y=9.0, logR=1.0, prior_is_G=True, lklhd_is_G=True):
    R = 4**logR
    xf = 0
    Pf = 1
    
    # (See exercise below)
    def H(x):
        return 1*x + 0

    x = grid1d
    prior_vals = pdf_G1(x, xf, Pf)  if prior_is_G else pdf_U1(x, xf, Pf)
    lklhd_vals = pdf_G1(y, H(x), R) if lklhd_is_G else pdf_U1(y, H(x), R)
    postr_vals = Bayes_rule(prior_vals, lklhd_vals, dx)

    def plot(x, y, c, lbl):
        plt.fill_between(x, y, color=c, alpha=.3, label=lbl)

    plt.figure(figsize=(8, 4))
    plot(x, prior_vals, 'blue'  , f'Prior, N(x | {xf:.4g}, {Pf:.4g})')
    plot(x, lklhd_vals, 'green' , f'Lklhd, N({y} | x, {R:.4g})')
    plot(x, postr_vals, 'red'   , f'Postr, pointwise')

    try:
        # (See exercise below)
        xa, Pa = Bayes_rule_G1(xf, Pf, y, R)
        label = f'Postr, parametric\nN(x | {xa:.4g}, {Pa:.4g})'
        postr_vals_G1 = pdf_G1(x, xa, Pa)
        plt.plot(x, postr_vals_G1, 'purple', label=label)
    except NameError:
        pass

    plt.ylim(0, 0.6)
    plt.legend(loc="upper right", prop={'family': 'monospace'})
    plt.show()


# **Exc -- Bayes1 properties:** This exercise serves to make you acquainted with how Bayes' rule blends information.  
#  Move the sliders (use arrow keys?) to animate it, and answer the following (with the boolean checkmarks both on and off).
#  * What happens to the posterior when $R \rightarrow \infty$ ?
#  * What happens to the posterior when $R \rightarrow 0$ ?
#  * Move $y$ around. What is the posterior's location (mean/mode) when $R$ equals the prior variance?
#  * Can you say something universally valid (for any $y$ and $R$) about the height of the posterior pdf?
#  * Does the posterior scale (width) depend on $y$?  
#    *Optional*: What does this mean [information-wise](https://en.wikipedia.org/wiki/Differential_entropy#Differential_entropies_for_various_distributions)?
#  * Consider the shape (ignoring location & scale) of the posterior. Does it depend on $R$ or $y$?
#  * Can you see a shortcut to computing this posterior rather than having to do the pointwise multiplication?
#  * For the case of two uniform distributions: What happens when you move the prior and likelihood too far apart? Is the fault of the implementation, the math, or the problem statement?
#  * Play around with the grid resolution (see the cell above). What is in your opinion a "sufficient" grid resolution?

# +
# ws.show_answer('Posterior behaviour')
# -

# ## With forward (observation) models
#
# Suppose the observation, $y$, is related to the true state, $x$,
#   via some "observation (forward) model", $\ObsMod$:
#   \begin{align*}
#   y &= \ObsMod(x) + r \, , \;\; \qquad \tag{Obs}
#   \end{align*}
#   where the corrupting additive noise has law $r \sim \NormDist(0, R)$ for some variance $R>0$.
# Then the likelihood is $$p(y|x) = \NormDist(y| \ObsMod(x), R) \,. \tag{Lklhd}$$
#
# **Exc (optional) -- The likelihood:** Derive the expression (Lklhd) for the likelihood.

# +
# ws.show_answer('Likelihood')
# -


# #### Exc -- Obs. model gallery
# Go back to the interactive illustration of Bayes' rule above.
# Change `H` to implement the following observation models, $\ObsMod$.
# In each case,
# - Explain the impact (shape, position, variance) on the likelihood (and thereby posterior).  
#   *PS: Note that in each case, the likelihood can be expressed via eqn. (Lklhd).*
# - Consider to what extent it is reasonable to say that $\ObsMod$ gets "inverted".  
#   *PS: it might be helpful to let $R \rightarrow 0$.*
#
# Try
#
# - (a) $\ObsMod(x) = x + 15$.
# - (b) $\ObsMod(x) = 2 x$.
#     *PS: The word "magnifying" might come to mind.*
#     - Does the likelihood integrate (in $x$) to 1? Should we care (also see [above](#Exc-(optional)----BR-normalization)) ?
# - (c) $\ObsMod(x) = x^2/4$. *PS: We're now doing "nonlinear regression"*.
#     - Is the resulting posterior Gaussian?
# - (d) Try $\ObsMod(x) = |x|$.
#     - Is the resulting posterior Gaussian?

# +
# ws.show_answer('Observation models', 'a')
# -

# **Exc (optional) -- "why inverse":** Laplace called "statistical inference" the reasoning of "inverse probability" (1774). You may also have heard of "inverse problems" in reference to similar problems, but without a statistical framing. In view of this, why do you think we use $x$ for the unknown, and $y$ for the known/given data?

# +
# ws.show_answer("what's forward?")
# -

# ## Multivariate Bayes (interlude)
# The following illustrates Bayes' rule in the 2D, i.e. multivariate, case.

# +
def H(x, kind="x"):
    y = x
    if   kind == "x^2": y = x**2
    elif kind == "x_1": y = x[:1]
    elif kind == "mean(x)": y = x[:1] + x[1:]
    elif kind == "diff(x)": y = x[:1] - x[1:]
    elif kind == "prod(x)": y = x[:1] * x[1:]
    # OR np.mean/prod with `keepdims=True`
    return y
H.kinds = ["(x_1, x_2)", "x^2", "x_1", "mean(x)", "diff(x)", "prod(x)"]

v = dict(orientation="vertical"),
@ws.interact(top=[['corr_B', 'corr_R']], bottom=[['y1', 'R1']], right=['variant', ['y2', 'R2']],
             corr_R=(-.999, .999, .01), y1=bounds,     R1=(0.01, 36, 0.2),
             corr_B=(-.999, .999, .01), y2=bounds + v, R2=(0.01, 36, 0.2) + v, variant=H.kinds)
def Bayes2(  corr_R=.6,                 y1=1,          R1=4**2,
             corr_B=.6,                 y2=-12,        R2=1,                   variant=H.kinds[0]):
    # Prior
    mu = np.zeros(2)
    B = 25 * np.array([[1, corr_B],
                       [corr_B, 1]])
    # Likelihood
    cov_R = np.sqrt(R1*R2)*corr_R
    R = np.array([[R1, cov_R],
                  [cov_R, R2]])
    y = np.array([y1, y2])

    # Restrict dimensionality to that of output of H
    if len(H(mu, variant)) == 1:
        i = slice(None, 1)
    else:
        i = slice(None)

    # Compute BR
    x = grid2d
    lklhd = pdf_GM(y[i], H(x.T, variant)[i].T, R[i, i])
    prior = pdf_GM(x, mu, B)
    postr = Bayes_rule(prior, lklhd, dx**2)

    ax, jplot = ws.get_jointplotter(grid1d)
    contours = [jplot(prior, 'blue'),
                jplot(lklhd, 'green'),
                jplot(postr, 'red', linewidths=2)]
    ax.legend(contours, ['prior', 'lklhd', 'postr'], loc="upper left")
    ax.set_title(r"$\mathscr{H}(x_1, x_2) = " + variant + "$")
    plt.show()


# -

# #### Exc (optional) -- Multivariate observation models
# - (a) Does the posterior (pdf) generally lie "between" the prior and likelihood?
# - (b) Try the different observation models in the dropdown menu, corresponding to
#     - $\ObsMod(\x) = \x$.
#     - $\ObsMod(\x) = (\x_1^2, \x_2^2)$.
#     - $\ObsMod(\x) = \x_1$.  
#       Why do `y2`, `R2` and `corr_R` become inactive? Also for the next items.
#     - $\ObsMod(\x) = \frac{1}{\xDim} \sum_{i=1}^\xDim \x_i$.
#     - $\ObsMod(\x) = \x_2 - \x_1$.
#     - $\ObsMod(\x) = \x_1 \x_2$.
#   
#   As [above](#Exc----Obs.-model-gallery), in each case,
#     - Explain the impact on the likelihood (and thereby posterior).  
#     - Consider to what extent it is reasonable to say that $\ObsMod$ gets "inverted".
#     - For those of the above models that are linear,
#       find the (possibly rectangular) matrix $\bH$ such that $\ObsMod(\x) = \bH \x$.

# +
# ws.show_answer('Multivariate Observations')
# -

# As simple as it is, the amount of computations done by `Bayes_rule` quickly becomes a difficulty in higher dimensions. This is hammered home in the following exercise.
#
# #### Exc (optional) -- Curse of dimensionality, part 1
#
#  * (a) How many point-multiplications are needed on a grid with $N$ points in $\xDim$ dimensions? Imagine an $\xDim$-dimensional cube where each side has a grid with $N$ points on it.
#    *PS: Of course, if the likelihood contains an actual model $\ObsMod(x)$ as well, its evaluations (computations) could be significantly more costly than the point-multiplications of Bayes' rule itself.*
#  * (b) Suppose we model 5 physical quantities [for example: velocity (u, v, w), pressure, and humidity fields] at each grid point/node for a discretized atmosphere of Earth. Assume the resolution is $1^\circ$ for latitude (110km), $1^\circ$ for longitude, and that we only use $3$ vertical layers. How many variables, $\xDim$, are there in total? This is the ***dimensionality*** of the unknown.
#  * (c) Suppose each variable is has a pdf represented with a grid using only $N=20$ points. How many multiplications are necessary to calculate Bayes rule (jointly) for all variables on our Earth model?

# +
# ws.show_answer('nD-space is big', 'a')
# -

# ## Gaussian-Gaussian Bayes' rule (1D)
#
# In response to this computational difficulty, we try to be smart and do something more analytical ("pen-and-paper"): we only compute the parameters (mean and (co)variance) of the posterior pdf.
#
# This is doable and quite simple in the Gaussian-Gaussian case, when $\ObsMod$ is linear (i.e. just a number):  
# - With a prior $p(x) = \mathcal{N}(x \mid x\supf, P\supf)$ and  
# - a likelihood $p(y|x) = \mathcal{N}(y \mid \ObsMod x,R)$,  
# - the posterior is
# $
# p(x|y)
# = \mathcal{N}(x \mid x\supa, P\supa) \,,
# $
# where, in the 1-dimensional/univariate/scalar (multivariate is discussed in [T5](T5%20-%20Kalman%20filter%20(multivariate).ipynb)) case:
#
# $$\begin{align}
#     P\supa &= 1/(1/P\supf + \ObsMod^2/R) \, , \tag{5} \\\
#   x\supa &= P\supa (x\supf/P\supf + \ObsMod y/R) \, .  \tag{6}
# \end{align}$$
#
# *There are a lot of sub/super-scripts. Please take a moment to somewhat digest the formulae.*
#
# #### Exc -- GG Bayes
# Consider the following identity, where $P\supa$ and $x\supa$ are given by eqns. (5) and (6).
# $$\frac{(x-x\supf)^2}{P\supf} + \frac{(x-y)^2}{R} \quad
# =\quad \frac{(x - x\supa)^2}{P\supa} + \frac{(y - x\supf)^2}{R + P\supf} \,, \tag{S2}$$
# Notice that the left hand side (LHS) is the sum of two squares with $x$,
# but the RHS only contains one square with $x$.
# - (a) Derive the first term of the RHS, i.e. eqns. (5) and (6).
# - (b) *Optional*: Derive the full RHS (i.e. also the second term).
# - (c) Derive $p(x|y) = \mathcal{N}(x \mid x\supa, P\supa)$ from eqns. (5) and (6)
#   using part (a), Bayes' rule (BR2), and the Gaussian pdf (G1).  
#   *Hint: you can temporarily "hide" $\ObsMod$ by astutely multiplying by $1$ somewhere.*

# +
# ws.show_answer('BR Gauss, a.k.a. completing the square', 'a')
# -

# **Exc -- Temperature example:**
# The statement $x = \mu \pm \sigma$ is *sometimes* used
# as a shorthand for $p(x) = \mathcal{N}(x \mid \mu, \sigma^2)$. Suppose
# - you think the temperature $x = 20°C \pm 2°C$,
# - a thermometer yields the observation $y = 18°C \pm 2°C$.
#
# Show that your posterior is $p(x|y) = \mathcal{N}(x \mid 19, 2)$

# +
# ws.show_answer('GG BR example')
# -

# The following implements a Gaussian-Gaussian Bayes' rule (eqns 5 and 6).
# *Note that its inputs and outputs are not discretised density values (as for `Bayes_rule()`), but simply 2 numbers: the mean and the variance.*

def Bayes_rule_G1(xf, Pf, y, R):
    Pa = 1 / (1/Pf + 1/R)
    xa = Pa * (xf/Pf + y/R)
    return xa, Pa


# #### Exc -- Gaussianity as an approximation
# Restore $y = x$ and re-run/execute the interactive animation code cell up above.
# - (a) Under what conditions does `Bayes_rule_G1()` provide a good approximation to `Bayes_rule()`?
# - (b) *Optional*. Try using one or more of the other [distributions readily available in `scipy`](https://stackoverflow.com/questions/37559470/) in the above animation.
#
# **Exc -- Gain algebra:** Show that eqn. (5) can be written as
# $$P\supa = K R \,,    \tag{8}$$
# where
# $$K = \frac{\ObsMod P\supf}{\ObsMod^2 P\supf + R} \,,    \tag{9}$$
# is called the "Kalman gain".  
# *Hint: again, try to "hide away" $\ObsMod$ among the other objects before proceeding.*
#
# Then shown that eqns (5) and (6) can be written as
# $$\begin{align}
#     P\supa &= (1-K \ObsMod) P\supf \, ,  \tag{10} \\\
#   x\supa &= x\supf + K (y- \ObsMod x\supf) \tag{11} \, ,
# \end{align}$$

# +
# ws.show_answer('BR Kalman1 algebra')
# -

# #### Exc (optional) -- Gain intuition
# - (a) Show that $0 < K < 1$ since $0 < P\supf, R$.
# - (b) Show that $P\supa < P\supf, R$.
# - (c) Show that $x\supa \in (x\supf, y)$.
# - (d) Why do you think $K$ is called a "gain"?

# +
# ws.show_answer('KG intuition')
# -

# **Exc -- BR with Gain:** Re-define `Bayes_rule_G1` so to as to use eqns. 9-11. Remember to re-run the cell. Verify that you get the same plots as before.

# +
# ws.show_answer('BR Kalman1 code')
# -

# #### Exc (optional) -- optimality of the mean
# *If you must* pick a single point value for your estimate (for example, an action to be taken), you can **decide** on it by optimising (with respect to the estimate) the expected value of some utility/loss function [[ref](https://en.wikipedia.org/wiki/Bayes_estimator)].
# - For example, if the density of $X$ is symmetric,
#    and $\text{Loss}$ is convex and symmetric,
#    then $\Expect[\text{Loss}(X - \theta)]$ is minimized
#    by the mean, $\Expect[X]$, which also coincides with the median.
#    <!-- See Corollary 7.19 of Lehmann, Casella -->
# - (a) Show that, for the expected *squared* loss, $\Expect[(X - \theta)^2]$,
#   the minimum is the mean for *any distribution*.
#   *Hint: insert $0 = \,?\, - \,?$.*
# - (b) Show that linearity can replace Gaussianity in the 1st bullet point.
#   *PS: this gives rise to various optimality claims of the Kalman filter,
#   such as it being the best linear-unibased estimator (BLUE).*
#
# In summary, the intuitive idea of **considering the mean of $p(x)$ as the point estimate** has good theoretical foundations.
#
# ### Next: [T4 - Filtering & time series](T4%20-%20Filtering%20%26%20time%20series.ipynb)

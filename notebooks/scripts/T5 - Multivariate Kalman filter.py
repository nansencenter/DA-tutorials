# ---
# jupyter:
#   jupytext:
#     formats: ipynb,scripts//py:light
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
from resources import show_answer, interact, cInterval

# %matplotlib inline
import numpy as np
import numpy.random as rnd
from scipy.linalg import inv
import matplotlib.pyplot as plt
plt.ion();


# # T5 - The Kalman filter (KF) -- multivariate
# Dealing with vectors and matrices is a lot like plain numbers. But some things get more complicated.
# $
# % ######################################## Loading TeX (MathJax)... Please wait ########################################
# \newcommand{\Reals}{\mathbb{R}} \newcommand{\Expect}[0]{\mathbb{E}} \newcommand{\NormDist}{\mathcal{N}} \newcommand{\DynMod}[0]{\mathscr{M}} \newcommand{\ObsMod}[0]{\mathscr{H}} \newcommand{\mat}[1]{{\mathbf{{#1}}}} \newcommand{\bvec}[1]{{\mathbf{#1}}} \newcommand{\trsign}{{\mathsf{T}}} \newcommand{\tr}{^{\trsign}} \newcommand{\ceq}[0]{\mathrel{≔}} \newcommand{\xDim}[0]{D} \newcommand{\supa}[0]{^\text{a}} \newcommand{\supf}[0]{^\text{f}} \newcommand{\I}[0]{\mat{I}} \newcommand{\K}[0]{\mat{K}} \newcommand{\bP}[0]{\mat{P}} \newcommand{\bH}[0]{\mat{H}} \newcommand{\bF}[0]{\mat{F}} \newcommand{\R}[0]{\mat{R}} \newcommand{\Q}[0]{\mat{Q}} \newcommand{\B}[0]{\mat{B}} \newcommand{\C}[0]{\mat{C}} \newcommand{\Ri}[0]{\R^{-1}} \newcommand{\Bi}[0]{\B^{-1}} \newcommand{\X}[0]{\mat{X}} \newcommand{\A}[0]{\mat{A}} \newcommand{\Y}[0]{\mat{Y}} \newcommand{\E}[0]{\mat{E}} \newcommand{\U}[0]{\mat{U}} \newcommand{\V}[0]{\mat{V}} \newcommand{\x}[0]{\bvec{x}} \newcommand{\y}[0]{\bvec{y}} \newcommand{\z}[0]{\bvec{z}} \newcommand{\q}[0]{\bvec{q}} \newcommand{\br}[0]{\bvec{r}} \newcommand{\bb}[0]{\bvec{b}} \newcommand{\bx}[0]{\bvec{\bar{x}}} \newcommand{\by}[0]{\bvec{\bar{y}}} \newcommand{\barB}[0]{\mat{\bar{B}}} \newcommand{\barP}[0]{\mat{\bar{P}}} \newcommand{\barC}[0]{\mat{\bar{C}}} \newcommand{\barK}[0]{\mat{\bar{K}}} \newcommand{\D}[0]{\mat{D}} \newcommand{\Dobs}[0]{\mat{D}_{\text{obs}}} \newcommand{\Dmod}[0]{\mat{D}_{\text{obs}}} \newcommand{\ones}[0]{\bvec{1}} \newcommand{\AN}[0]{\big( \I_N - \ones \ones\tr / N \big)}
# $

# ## Another time series problem, now multivariate
#
# Recall the AR(1) process from the previous tutorial: $x_{k+1} = \DynMod x_k + q_k$.
# - It could result from discretizing [exponential decay](https://en.wikipedia.org/wiki/Exponential_decay):
#   $\frac{d x}{d t} = - \beta x \,,$ for some $\beta \geq 0$, and
#   adding some white noise, $\frac{d q}{d t}$.
# - Discretisation
#   - using explicit-Euler produces $\DynMod = (1 - \beta\, \Delta t)$,
#   - using implicit-Euler produces $\DynMod = 1/(1 + \beta\, \Delta t)$.
#   - such that $x_{k+1}$ equals the analytic solution requires $\DynMod = e^{- \beta\, \Delta t}$.
#   - *PS: note that the 1-st order Taylor expansion of each scheme is the same.*
# - Recall that $\{x_k\}$ became a (noisy) constant (horizontal) line when $\DynMod = 1$,
#   which makes sense since then $\beta = 0$.
# - Similarly, a straight (sloping) line would result from
#   $\frac{d^2 x}{d t^2} = 0 \,.$
#
# To make matters more interesting we're now going to consider the $\xDim$-th order model:
#   $\displaystyle \frac{d^{\xDim} x}{d t^\xDim} = 0 \,.$
# - This can be rewritten as a 1-st order *vector* (i.e. coupled system of) ODE:
#   $\frac{d x_i}{d t} = x_{i+1} \,,$ and $x_{{\xDim}+1} = 0$   
#   where the subscript $i$ is now instead the *index* of the state vector element.
# - Again we include noise, $\frac{d q_i}{d t}$,
#   and damping (exponential decay), $- \beta x_i$, to each component.
# - In total, $ \frac{d x_i}{d t} = x_{i+1} - \beta x_i + \frac{d q_i}{d t} \, .$
# - Discretizing with time step $\Delta t=1$ produces
#   $ x_{k+1, i} = x_{k, i+1} + 0.9 x_{k, i} + q_{k, i}\,,$  
#   i.e. $\beta = 0.1$ or $\beta = -\log(0.9)$ depending on which scheme was used.
#
# Thus, $\x_{k+1} = \DynMod \x_k + \q_k$, with $\DynMod$ the matrix specified below.

# +
xDim = 4 # state (x) length, also model order
M = 0.9*np.eye(xDim) + np.diag(np.ones(xDim-1), 1)
print("M =", M, sep="\n")

nTime = 100
Q = 0.01**2 * np.diag(1+np.arange(xDim))
# -

# #### Observing system
# Note that will generate a $\xDim$-dimensional time series.
# But we will only observe the 1st (`0`th in Python) element/component of the state vector.
# We say that the other components are **hidden**.

# +
H = np.zeros((1, xDim))
H[0, 0] = 1.0
print("H =", H)

R = 30**2 * np.identity(1)
# -

# #### Simulation
# The following simulates a synthetic truth (x) time series and observations (y).
# In particular, note the use of `@` for matrix/vector algebra, in place of `*` as in the [scalar case of the previous tutorial](T4%20-%20Time%20series%20filtering.ipynb#Example-problem:-AR(1)).

# +
rnd.seed(4)

# Initial condition
xa = np.zeros(xDim)
Pa = 0.1**2 * np.diag(np.arange(xDim))
x = xa + np.sqrt(Pa) @ rnd.randn(xDim)

truths = np.zeros((nTime, xDim))
obsrvs = np.zeros((nTime, len(H)))
for k in range(nTime):
    x = M @ x + np.sqrt(Q) @ rnd.randn(xDim)
    y = H @ x + np.sqrt(R) @ rnd.randn(1)
    truths[k] = x
    obsrvs[k] = y

for i, x in enumerate(truths.T):
    magnification = (i+1)**4  # for illustration purposes
    plt.plot(magnification*x, label=fr"${magnification}\,x_{i}$")
plt.legend();
# -
# ## The KF forecast step
#
# The forecast step (and its derivation) remains essentially unchanged from the [univariate case](T4%20-%20Time%20series%20filtering.ipynb#The-(univariate)-Kalman-filter-(KF)).
# The only difference is that $\DynMod$ is now a *matrix*, as well as the use of the transpose ${}^T$ in the covariance equation:
# $\begin{align}
# \x\supf_k
# &= \DynMod_{k-1} \x\supa_{k-1} \,, \tag{1a} \\\
# \bP\supf_k
# &= \DynMod_{k-1} \bP\supa_{k-1} \DynMod_{k-1}^T + \Q_{k-1} \,. \tag{1b}
# \end{align}$
#
# ## The KF analysis step
#
# It may be shown that the prior $p(\x) = \NormDist(\x \mid \x\supf,\bP\supf)$
# and likelihood $p(\y|\x) = \NormDist(\y \mid \ObsMod \x,\R)$,
# yield the posterior:
# \begin{align}
# p(\x|\y)
# &= \NormDist(\x \mid \x\supa, \bP\supa) \tag{4}
# \,,
# \end{align}
# where the posterior/analysis mean (vector) and covariance (matrix) are given by:
# \begin{align}
# 			\bP\supa &= \big(\ObsMod\tr \Ri \ObsMod + (\bP\supf)^{-1}\big)^{-1} \,, \tag{5} \\
# 			\x\supa &= \bP\supa\left[\ObsMod\tr \Ri \y + (\bP\supf)^{-1} \x\supf\right] \tag{6} \,,
# \end{align}
# *PS: all of the above could be subscripted by a single time index ($k$), but that seems unnecessary.*
#
# **Exc (optional) -- The 'precision' form of the KF:** Prove eqns (4-6).  
# *Hint: similar to the [univariate case](T3%20-%20Bayesian%20inference.ipynb#Exc----GG-Bayes), the main part lies in "completing the square" in $\x$.*

# +
# show_answer('KF precision')
# -

# ## Implementation & illustration

estims = np.zeros((nTime, 2, xDim))
covars = np.zeros((nTime, 2, xDim, xDim))
for k in range(nTime):
    # Forecast step
    xf = M@xa
    Pf = M@Pa@M.T + Q
    # Analysis update step
    y = obsrvs[k]
    Pa = inv( inv(Pf) + H.T@inv(R)@H )
    xa = Pa @ ( inv(Pf)@xf + H.T@inv(R)@y )
    # Assign
    estims[k] = xf, xa
    covars[k] = Pf, Pa

# Using `inv` is very bad practice, since it is not numerically stable.
# You generally want to use `scipy.linalg.solve` instead, or a more fine-grained matrix decomposition routine.
# But that is not possible here, since we have no "right hand side" to solve for in the formula for `Pa`.
# We'll address this point later.
#
# <mark><font size="-1">
# *Caution!: Because of our haphazard use of global variables, re-running the KF (without re-running the truth-generating cell) will take as initial condition the endpoint of the previous run.*
# </font></mark>
#
# Use the following to plot the result.

fig, axs = plt.subplots(figsize=(10, 6), nrows=xDim, sharex=True)
for i, (ax, truth, estim) in enumerate(zip(axs, truths.T, estims.T)):
    kk = 1 + np.arange(nTime)
    kk2 = kk.repeat(2)
    ax.plot(kk, truth, c='k')
    ax.plot(kk2, estim.T.flatten())
    ax.fill_between(kk2, *cInterval(estim.T, covars[..., i, i]), alpha=.2)
    if i == 0 and H[0, 0] == 1 and np.sum(np.abs(H)) == 1:
        ax.plot(kk, obsrvs, '.')
    ax.set_ylabel(f"$x_{i}$")
    ax.set_xlim([0, nTime])


# It is also about time that we have a look at how the state uncertainty/error covariance matrix looks. After all, it is the covariance that is used by the KF to update the hidden state estimates. But since the scales are so different between the components, it is more suitable to consider the correlation matrix. Run the cell below, and note that
# - It converges in time to a fixed value, as we might expect from T4.
# - There are no negative correlations in this case, which is perhaps a bit boring.

@interact(k=(1, nTime))
def plot_correlation_matrix(k=1, analysis=True):
    Pf, Pa = covars[k-1]
    covmat = Pa if analysis else Pf
    stds = np.sqrt(np.diag(covmat))
    corrmat = covmat / np.outer(stds, stds)
    plt.matshow(corrmat, cmap='coolwarm', vmin=-1, vmax=+1)
    plt.grid(False)
    plt.colorbar(shrink=0.5)
    plt.show()

# ## Woodbury and the Kalman gain
# The KF formulae, as specified above, can be pretty expensive...
#
# #### Exc (optional) -- flops and MBs
# Suppose the length of $\x$ is $\xDim$ and denote its covariance matrix by $\bP$.
#  * (a) What's the size of $\bP$?
#  * (b) How many "flops" (approximately, i.e. up to leading order) are required  
#    to compute the "precision form" of the KF update equation, eqn (5) ?  
#    *Hint: Assume the computationally demanding part is the [Cholesky decomposition](https://en.wikipedia.org/wiki/Cholesky_decomposition#Computation).*
#  * (c) How much memory (bytes) is required to hold its covariance matrix $\bP$ ?
#  * (d) How many megabytes (MB) is that if $\xDim$ is a million,
#    as in our [$1^\circ$ (110km) resolution Earth atmosphere model](T3%20-%20Bayesian%20inference.ipynb#Exc-(optional)----Curse-of-dimensionality,-part-1).
#  * (e) How many times more MB or flops are needed if you double the resolution (in all 3 dimensions) ?

# +
# show_answer('nD-covars are big')
# -

# This is one of the principal reasons why basic extended KF is infeasible for DA. In the following we derive the "gain" form of the KF analysis update, which should help at least a little bit.
#
# #### Exc -- The "Woodbury" matrix inversion identity
# The following is known as the Sherman-Morrison-Woodbury lemma/identity,
# $$\begin{align}
#     \bP = \left( \B^{-1} + \V\tr \R^{-1} \U \right)^{-1}
#     =
#     \B - \B \V\tr \left( \R + \U \B \V\tr \right)^{-1} \U \B \,,
#     \tag{W}
# \end{align}$$
# which holds for any (suitably shaped matrices)
# $\B$, $\R$, $\V,\U$ *such that the above exists*.
#
# Prove the identity. *Hint: don't derive it, just prove it!*

# +
# show_answer('Woodbury')
# -

# #### Exc (optional) -- Matrix shape compatibility
# - Show that $\B$ and $\R$ must be square.
# - Show that $\U$ and $\V$ are not necessarily square, but must have the same dimensions.
# - Show that $\B$ and $\R$ are not necessarily of equal size.
#

# The above exercise makes it clear that the Woodbury identity may be used to compute $\bP$ by inverting matrices of the size of $\R$ rather than the size of $\B$.
# Of course, if $\R$ is bigger than $\B$, then the identity is useful the other way around.

# #### Exc (optional) -- Corollary 1
# Prove that, for any symmetric, positive-definite
# ([SPD](https://en.wikipedia.org/wiki/Definiteness_of_a_matrix#Properties))
# matrices $\R$ and $\B$, and any matrix $\ObsMod$,
# $$\begin{align}
#  	\left(\ObsMod\tr \R^{-1} \ObsMod + \B^{-1}\right)^{-1}
#     &=
#     \B - \B \ObsMod\tr \left( \R + \ObsMod \B \ObsMod\tr \right)^{-1} \ObsMod \B \tag{C1}
#     \,.
# \end{align}$$

# +
# show_answer('inv(SPD + SPD)')
# -

# #### Exc (optional) -- Corollary 2
# Prove that, for the same matrices as for Corollary C1,
# $$\begin{align}
# 	\left(\ObsMod\tr \R^{-1} \ObsMod + \B^{-1}\right)^{-1}\ObsMod\tr \R^{-1}
#     &= \B \ObsMod\tr \left( \R + \ObsMod \B \ObsMod\tr \right)^{-1}
#     \tag{C2}
#     \, .
# \end{align}$$

# +
# show_answer('Woodbury C2')
# -

# #### Exc -- The "Gain" form of the KF
# Now, let's go back to the KF, eqns (5) and (6). Since $\bP\supf$ and $\R$ are covariance matrices, they are symmetric-positive. In addition, we will assume that they are full-rank, making them SPD and invertible.  
#
# Define the Kalman gain by:
#  $$\begin{align}
#     \K &= \bP\supf \ObsMod\tr \big(\ObsMod \bP\supf \ObsMod\tr + \R\big)^{-1} \,. \tag{K1}
# \end{align}$$
#  * (a) Apply (C1) to eqn (5) to obtain the Kalman gain form of analysis/posterior covariance matrix:
# $$\begin{align}
#     \bP\supa &= [\I_{\xDim} - \K \ObsMod]\bP\supf \,. \tag{8}
# \end{align}$$
#
# * (b) Apply (C2)  to (5) to obtain the identity
# $$\begin{align}
#     \K &= \bP\supa \ObsMod\tr \R^{-1}  \,. \tag{K2}
# \end{align}$$
#
# * (c) Show that $\bP\supa (\bP\supf)^{-1} = [\I_{\xDim} - \K \ObsMod]$.
# * (d) Use (b) and (c) to obtain the Kalman gain form of analysis/posterior covariance
# $$\begin{align}
#      \x\supa &= \x\supf + \K\left[\y - \ObsMod \x\supf\right] \, . \tag{9}
# \end{align}$$
# Together, eqns (8) and (9) define the Kalman gain form of the KF update.
# Note that the inversion (eqn 7) involved is of the size of $\R$, while in eqn (5) it is of the size of $\bP\supf$.
#
# #### Exc -- KF implemented with gain
# Implement the Kalman gain form in place of the precision form of the KF, including
# - Use `scipy.linalg.solve`.
# - Re-run all cells.
# - Verify that you get the same result as before.

# ## Summary
# We have derived two forms of the multivariate KF analysis update step: the
# "precision matrix" form, and the "Kalman gain" form. The latter is especially
# practical when the number of observations is smaller than the length of the
# state vector. Still, the best is yet to come: the ability to handle very
# large and chaotic systems
# (which are more fun than stochastically driven signals such as above).
#
# ### Next: [T6 - Spatial statistics ("geostatistics") & Kriging](T6%20-%20Geostats%20%26%20Kriging%20(optional).ipynb)

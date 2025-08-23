---
jupyter:
  jupytext:
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
from resources import show_answer, interact, import_from_nb, get_jointplotter
%matplotlib inline
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
plt.ion();
```

# T3 - Bayesian inference

$
% ######################################## Loading TeX (MathJax)... Please wait ########################################
\newcommand{\Reals}{\mathbb{R}} \newcommand{\Expect}[0]{\mathbb{E}} \newcommand{\NormDist}{\mathscr{N}} \newcommand{\DynMod}[0]{\mathscr{M}} \newcommand{\ObsMod}[0]{\mathscr{H}} \newcommand{\mat}[1]{{\mathbf{{#1}}}} \newcommand{\bvec}[1]{{\mathbf{#1}}} \newcommand{\trsign}{{\mathsf{T}}} \newcommand{\tr}{^{\trsign}} \newcommand{\ceq}[0]{\mathrel{≔}} \newcommand{\xDim}[0]{D} \newcommand{\supa}[0]{^\text{a}} \newcommand{\supf}[0]{^\text{f}} \newcommand{\I}[0]{\mat{I}} \newcommand{\K}[0]{\mat{K}} \newcommand{\bP}[0]{\mat{P}} \newcommand{\bH}[0]{\mat{H}} \newcommand{\bF}[0]{\mat{F}} \newcommand{\R}[0]{\mat{R}} \newcommand{\Q}[0]{\mat{Q}} \newcommand{\B}[0]{\mat{B}} \newcommand{\C}[0]{\mat{C}} \newcommand{\Ri}[0]{\R^{-1}} \newcommand{\Bi}[0]{\B^{-1}} \newcommand{\X}[0]{\mat{X}} \newcommand{\A}[0]{\mat{A}} \newcommand{\Y}[0]{\mat{Y}} \newcommand{\E}[0]{\mat{E}} \newcommand{\U}[0]{\mat{U}} \newcommand{\V}[0]{\mat{V}} \newcommand{\x}[0]{\bvec{x}} \newcommand{\y}[0]{\bvec{y}} \newcommand{\z}[0]{\bvec{z}} \newcommand{\q}[0]{\bvec{q}} \newcommand{\br}[0]{\bvec{r}} \newcommand{\bb}[0]{\bvec{b}} \newcommand{\bx}[0]{\bvec{\bar{x}}} \newcommand{\by}[0]{\bvec{\bar{y}}} \newcommand{\barB}[0]{\mat{\bar{B}}} \newcommand{\barP}[0]{\mat{\bar{P}}} \newcommand{\barC}[0]{\mat{\bar{C}}} \newcommand{\barK}[0]{\mat{\bar{K}}} \newcommand{\D}[0]{\mat{D}} \newcommand{\Dobs}[0]{\mat{D}_{\text{obs}}} \newcommand{\Dmod}[0]{\mat{D}_{\text{obs}}} \newcommand{\ones}[0]{\bvec{1}} \newcommand{\AN}[0]{\big( \I_N - \ones \ones\tr / N \big)}
$
The [previous tutorial](T2%20-%20Gaussian%20distribution.ipynb)
studied the Gaussian probability density function (pdf), defined in 1D by:
$$ \large \NormDist(x \mid \mu, \sigma^2) = (2 \pi \sigma^2)^{-1/2} e^{-(x-\mu)^2/2 \sigma^2} \,,\tag{G1} $$
which we implemented and tested alongside the uniform distribution.

```python
(pdf_G1, pdf_U1, bounds, dx, grid1d) = import_from_nb("T2", ("pdf_G1", "pdf_U1", "bounds", "dx", "grid1d"))
pdfs = dict(N=pdf_G1, U=pdf_U1)
```

Now that we have reviewed some probability, we can look at statistical inference and estimation, in particular

# Bayes' rule

<details style="border: 1px solid #aaaaaa; border-radius: 4px; padding: 0.5em 0.5em 0;">
  <summary style="font-weight: normal; font-style: italic; margin: -0.5em -0.5em 0; padding: 0.5em;">
    In the Bayesian approach, knowledge and uncertainty about some unknown ($x$) is quantified through probability... (optional reading 🔍🤓)
  </summary>

  For example, what is the temperature at the surface of Mercury (at some given point and time)?
  Not many people know the answer. Perhaps you say $500^{\circ} C, \, \pm \, 20$.
  But that's hardly anything compared to you real uncertainty, so you revise that to $\pm \, 1000$.
  But then you're allowing for temperature below absolute zero, which you of course don't believe is possible.
  You can continue to refine the description of your uncertainty.
  Ultimately (in the limit) the complete way to express your belief is as a *distribution*
  (essentially just a list) of plausibilities for all possibilities.
  Furthermore, the only coherent way to reason in the presence of such uncertainty
  is to obey the laws of probability ([Jaynes (2003)](#References)).

  - - -
</details>

And **Bayes' rule** is how we do inference: it says how to condition/merge/assimilate/update this belief based on data/observation ($y$).
For *continuous* random variables, $x$ and $y$, it reads:
$$
\large
\color{red}{\overset{\mbox{Posterior}}{p(\color{black}{x|y})}} = \frac{\color{blue}{\overset{\mbox{  Prior  }}{p(\color{black}{x})}} \, \color{green}{\overset{\mbox{ Likelihood}}{p(\color{black}{y|x})}}}{\color{gray}{\underset{\mbox{Constant wrt. x}}{p(\color{black}{y})}}} \,. \tag{BR} \\[1em]
$$
Note that, in contrast to orthodox statistics,
Bayes' rule (BR) itself makes no attempt at producing only a single estimate/value
It merely states how quantitative belief (weighted possibilities) should be updated in view of new data.

**Exc -- Bayes' rule derivation:** Derive eqn. (BR) from the definition of [conditional pdf's](https://en.wikipedia.org/wiki/Conditional_probability_distribution#Conditional_continuous_distributions).

```python
# show_answer('symmetry of conjunction')
```

It is hard to overstate how simple Bayes' rule, eqn. (BR), is, consisting merely of scalar multiplication and division.
However, we want to compute the function $p(x|y)$ for **all values of $x$**.
Thus, upon discretization, eqn. (BR) becomes the multiplication of two *arrays* of values (followed by a normalisation):

```python
def Bayes_rule(prior_values, lklhd_values, dx):
    prod = prior_values * lklhd_values         # pointwise multiplication
    posterior_values = prod/(np.sum(prod)*dx)  # normalization
    return posterior_values
```

#### Exc (optional) -- BR normalization

Show that the normalization in `Bayes_rule()` amounts to (approximately) the same as dividing by $p(y)$.

```python
# show_answer('quadrature marginalisation')
```

In fact, since $p(y)$ is thusly implicitly known,
we often don't bother to write it down, simplifying Bayes' rule (eqn. BR) to

$$ p(x|y) \propto p(x) \, p(y|x) \,.  \tag{BR2} $$

Actually, do we even need to care about $p(y)$ at all? All we really need to know is how much more likely some value of $x$ (or an interval around it) is compared to any other $x$.
The normalisation is only necessary because of the *convention* that all densities integrate to $1$.
However, for large models, we usually can only afford to evaluate $p(y|x)$ at a few points (of $x$), so that the integral for $p(y)$ can only be roughly approximated. In such settings, estimation of the normalisation factor becomes an important question too.

## Interactive illustration

The code below shows Bayes' rule in action.

```python
@interact(y=(*bounds, 1), logR=(-3, 5, .5), prior_kind=list(pdfs), lklhd_kind=list(pdfs))
def Bayes1(y=9.0, logR=1.0, lklhd_kind="N", prior_kind="N"):
    R = 4**logR
    xf = 10
    Pf = 4**2

    # (See exercise below)
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
    plot(x, lklhd_vals, 'green' , f'Lklhd, {lklhd_kind}({y} | x, {R:.4g})')
    plot(x, postr_vals, 'red'   , f'Postr, pointwise')

    try:
        # (See exercise below)
        H_lin = H(xf)/xf # a simple linear approximation of H(x)
        xa, Pa = Bayes_rule_G1(xf, Pf, y, H_lin, R)
        label = f'Postr, parametric\nN(x | {xa:.4g}, {Pa:.4g})'
        postr_vals_G1 = pdf_G1(x, xa, Pa)
        plt.plot(x, postr_vals_G1, 'purple', label=label)
    except NameError:
        pass

    plt.ylim(0, 0.6)
    plt.legend(loc="upper left", prop={'family': 'monospace'})
    plt.show()
```

The illustration uses a

- prior $p(x) = \NormDist(x|x^f, P^f)$ with (fixed) mean and variance, $x^f= 10$, $P^f=4^2$.
- likelihood $p(y|x) = \NormDist(y|x, R)$, whose parameters are set by the interactive sliders.

We are now dealing with 3 (!) separate distributions,
giving us a lot of symbols to keep straight in our head -- a necessary evil for later.

**Exc -- `Bayes1` properties:** This exercise serves to make you acquainted with how Bayes' rule blends information.

Move the sliders (use arrow keys?) to animate it, and answer the following (with the boolean checkmarks both on and off).

- What happens to the posterior when $R \rightarrow \infty$ ?
- What happens to the posterior when $R \rightarrow 0$ ?
- Move $y$ around. What is the posterior's location (mean/mode) when $R$ equals the prior variance?
- Can you say something universally valid (for any $y$ and $R$) about the height of the posterior pdf?
- Does the posterior scale (width) depend on $y$?  
   *Optional*: What does this mean [information-wise](https://en.wikipedia.org/wiki/Differential_entropy#Differential_entropies_for_various_distributions)?
- Consider the shape (ignoring location & scale) of the posterior. Does it depend on $R$ or $y$?
- Can you see a shortcut to computing this posterior rather than having to do the pointwise multiplication?
- For the case of two uniform distributions: What happens when you move the prior and likelihood too far apart? Is the fault of the implementation, the math, or the problem statement?
- Play around with the grid resolution (see the cell above). What is in your opinion a "sufficient" grid resolution?

```python
# show_answer('Posterior behaviour')
```

## With forward (observation) models

In general, the observation $y$ is not a "direct" measurement of $x$, as above,
but rather some transformation, i.e. function of $x$,
which is called **observation/forward model**, $\ObsMod$.
Examples include:

- $\ObsMod(x) = x + 273$ for a thermometer reporting °C, while $x$ is the temperature in °K.
- $\ObsMod(x) = 10 x$ for a ruler using mm, while $x$ is stored as cm.
- $\ObsMod(x) = \log(x)$ for litmus paper (pH measurement), where $x$ is the molar concentration of hydrogen ions.
- $\ObsMod(x) = |x|$ for bicycle speedometers (measuring rpm, i.e. Hall effect sensors).
- $\ObsMod(x) = 2 \pi h \, x^2$ if observing inebriation (drunkenness), and the unknown, $x$, is the radius of the beer glasses.

Of course, the linear and logarithmic transformations are hardly worthy of the name "model", since they merely change the scale of measurement, and so could be trivially done away with. But doing so is not necessary, and they will serve to illustrate some important points.

In addition, measurement instruments always (at least for continuous variables) have limited accuracy,
i.e. there is an **measurement noise/error** corrupting the observation. For simplicity, this noise is usually assumed *additive*, so that the observation, $y$, is related to the true state, $x$, by
$$
y = \ObsMod(x) + \varepsilon \,, \;\; \qquad \tag{Obs}
$$
and $\varepsilon \sim \NormDist(0, R)$ for some variance $R>0$.
Then the likelihood is $$p(y|x) = \NormDist(y| \ObsMod(x), R) \,. \tag{Lklhd}$$

**Exc (optional) -- The likelihood:** Derive the expression (Lklhd) for the likelihood.

```python
# show_answer('Likelihood')
```

#### Exc -- Obs. model gallery

Go back to the interactive illustration of Bayes' rule above.
Change `H` to implement the following observation models, $\ObsMod$.
In each case,

- Explain the impact on the likelihood (and thereby posterior): shape (e.g. Gaussian?), position, variance.
- Consider to what extent it is reasonable to say that $\ObsMod$ gets "inverted".  
  *PS: it might be helpful to let $R \rightarrow 0$.*

Try

- (a) $\ObsMod(x) = x + 15$.
- (b) $\ObsMod(x) = 2 x$.
- (c) $\ObsMod(x) = (x-5)^2$.
  - Explain how negative values of $y$ are possible.
- (d) Try $\ObsMod(x) = |x|$.

```python
# show_answer('Observation models', 'a')
```

It is important to appreciate that the likelihood and its role in Bayes' rule, does no "inversion". It simply quantifies how well $x$ fits to the data in terms of its weighting. This approach also inherently handles the fact that multiple values of $x$ may be plausible.

**Exc (optional) -- "why inverse":** Laplace called "statistical inference" the reasoning of "inverse probability" (1774). You may also have heard of "inverse problems" in reference to similar problems, but without a statistical framing. In view of this, why do you think we use $x$ for the unknown, and $y$ for the known/given data?

```python
# show_answer("what's forward?")
```

<a name="Gaussian-Gaussian-Bayes'-rule-(1D)"></a>

## Gaussian-Gaussian Bayes' rule (1D)

In response to this computational difficulty, we try to be smart and do something more analytical ("pen-and-paper"): we only compute the parameters (mean and (co)variance) of the posterior pdf.

This is doable and quite simple in the Gaussian-Gaussian case, when $\ObsMod$ is linear (i.e. just a number):  

- Given the prior of $p(x) = \NormDist(x \mid x\supf, P\supf)$
- and a likelihood $p(y|x) = \NormDist(y \mid \ObsMod x,R)$,  
- $\implies$ posterior
$
p(x|y)
= \NormDist(x \mid x\supa, P\supa) \,,
$
where, in the 1-dimensional/univariate/scalar (multivariate is discussed in [T5](T5%20-%20Multivariate%20Kalman%20filter.ipynb)) case:

Consider the following identity, where $P\supa$ and $x\supa$ are given by eqns. (5) and (6).
$$
\frac{(x-x\supf)^2}{P\supf} + \frac{(\ObsMod x-y)^2}{R} \quad =
\quad \frac{(x - x\supa)^2}{P\supa} + \frac{(y - \ObsMod x\supf)^2}{R + P\supf} \,, \tag{S2}
$$
Notice that the left hand side (LHS) is the sum of two squares with $x$,
but the RHS only contains one square with $x$.

- (a) Actually derive the first term of the RHS, i.e. eqns. (5) and (6).  
  *Hint: you can simplify the task by first "hiding" $\ObsMod$*
- (b) *Optional*: Derive the full RHS (i.e. also the second term).
- (c) Derive $p(x|y) = \NormDist(x \mid x\supa, P\supa)$ from eqns. (5) and (6)
  using part (a), Bayes' rule (BR2), and the Gaussian pdf (G1).

```python
# show_answer('BR Gauss, a.k.a. completing the square', 'a')
```

**Exc -- Temperature example:**
The statement $x = \mu \pm \sigma$ is *sometimes* used
as a shorthand for $p(x) = \NormDist(x \mid \mu, \sigma^2)$. Suppose

- you think the temperature $x = 20°C \pm 2°C$,
- a thermometer yields the observation $y = 18°C \pm 2°C$.

Show that your posterior is $p(x|y) = \NormDist(x \mid 19, 2)$

```python
# show_answer('GG BR example')
```

The following implements a Gaussian-Gaussian Bayes' rule (eqns. 5 and 6).
Note that its inputs and outputs are not discretized density values (as for `Bayes_rule()`), but simply 5 numbers: the means, variances and $\ObsMod$.

```python
def Bayes_rule_G1(xf, Pf, y, H, R):
    Pa = 1 / (1/Pf + H**2/R)
    xa = Pa * (xf/Pf + H*y/R)
    return xa, Pa
```

#### Exc -- Gaussianity as an approximation

Re-run/execute the interactive animation code cell up above.

- (a) Under what conditions does `Bayes_rule_G1()` provide a good approximation to `Bayes_rule()`?
- (b) Try using one or more of the other [distributions readily available in `scipy`](https://stackoverflow.com/questions/37559470/) in the above animation by inserting them in `pdfs`.

**Exc (optional) -- Gain algebra:** Show that eqn. (5) can be written as
$$P\supa = K R / \ObsMod \,,    \tag{8}$$
where
$$K = \frac{\ObsMod P\supf}{\ObsMod^2 P\supf + R} \,,    \tag{9}$$
is called the "Kalman gain".  
*Hint: again, try to "hide away" $\ObsMod$ among the other objects before proceeding.*

Then shown that eqns. (5) and (6) can be written as
$$
\begin{align}
    P\supa &= (1-K \ObsMod) P\supf \,,  \tag{10} \\\
  x\supa &= x\supf + K (y- \ObsMod x\supf) \tag{11} \,,
\end{align}
$$

```python
# show_answer('BR Kalman1 algebra')
```

#### Exc (optional) -- Gain intuition

Let $\ObsMod = 1$ for simplicity.

- (a) Show that $0 < K < 1$ since $0 < P\supf, R$.
- (b) Show that $P\supa < P\supf, R$.
- (c) Show that $x\supa \in (x\supf, y)$.
- (d) Why do you think $K$ is called a "gain"?

```python
# show_answer('KG intuition')
```

**Exc -- BR with Gain:** Re-define `Bayes_rule_G1` so to as to use eqns. 9-11. Remember to re-run the cell. Verify that you get the same plots as before.

```python
# show_answer('BR Kalman1 code')
```

#### Exc (optional) -- optimality of the mean

*If you must* pick a single point value for your estimate (for example, an action to be taken), you can **decide** on it by optimising (with respect to the estimate) the expected value of some utility/loss function [[ref](https://en.wikipedia.org/wiki/Bayes_estimator)].

- For example, if the density of $X$ is symmetric,
   and $\text{Loss}$ is convex and symmetric,
   then $\Expect[\text{Loss}(X - \theta)]$ is minimized
   by the mean, $\Expect[X]$, which also coincides with the median.
   <!-- See Corollary 7.19 of Lehmann, Casella -->
- (a) Show that, for the expected *squared* loss, $\Expect[(X - \theta)^2]$,
  the minimum is the mean for *any distribution*.
  *Hint: insert $0 = \,?\, - \,?$.*
- (b) Show that linearity can replace Gaussianity in the 1st bullet point.
  *PS: this gives rise to various optimality claims of the Kalman filter,
  such as it being the best linear-unbiased estimator (BLUE).*

In summary, the intuitive idea of **considering the mean of $p(x)$ as the point estimate** has good theoretical foundations.

## Summary

Bayesian inference quantifies uncertainty (in $x$) using the notion of probability.
Bayes' rule says how to condition/merge/assimilate/update this belief based on data/observation ($y$).
It is simply a re-formulation of the notion of conditional probability.
Observation can be "inverted" using Bayes' rule,
in the sense that all possibilities for $x$ are weighted.
While technically simple, Bayes' rule becomes expensive to compute in high dimensions,
but if Gaussianity can be assumed then it reduces to only 2 formulae.

### Next: [T4 - Filtering & time series](T4%20-%20Time%20series%20filtering.ipynb)

<a name="References"></a>

### References

- **Jaynes (2003)**:
  Edwin T. Jaynes, "Probability theory: the logic of science", 2003.
